"""Decode uncompressed RIFF WAV without assuming integer PCM."""
import math
import struct
from pathlib import Path


class InvalidAudio(ValueError):
    """The model returned an empty, silent or structurally broken waveform."""


def read_wav(path: Path) -> tuple[list[float], dict]:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise InvalidAudio("invalid_wav_header")
    end = struct.unpack_from("<I", data, 4)[0] + 8
    if end != len(data):
        raise InvalidAudio("truncated_or_inconsistent_riff")
    fmt, raw = None, None
    pos = 12
    while pos < end:
        if pos + 8 > end:
            raise InvalidAudio("truncated_chunk_header")
        name, size = struct.unpack_from("<4sI", data, pos)
        pos += 8
        if pos + size > end:
            raise InvalidAudio("truncated_chunk")
        chunk = data[pos:pos + size]
        if name == b"fmt ":
            fmt = chunk
        elif name == b"data":
            if raw is not None:
                raise InvalidAudio("multiple_data_chunks")
            raw = chunk
        pos += size + size % 2
    if fmt is None or len(fmt) < 16 or not raw:
        raise InvalidAudio("missing_format_or_empty_audio")
    tag, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == 0xfffe:
        if len(fmt) < 40 or fmt[26:40] != bytes.fromhex("000000001000800000aa00389b71"):
            raise InvalidAudio("unsupported_extensible_wav")
        tag = struct.unpack_from("<H", fmt, 24)[0]
    width = bits // 8
    if tag not in (1, 3) or bits not in ((8, 16, 24, 32) if tag == 1 else (32, 64)):
        raise InvalidAudio("unsupported_wav_encoding")
    if channels < 1 or rate < 1 or align != channels * width or byte_rate != rate * align or len(raw) % align:
        raise InvalidAudio("invalid_or_truncated_frames")
    if tag == 3:
        values = [v[0] for v in struct.iter_unpack("<f" if bits == 32 else "<d", raw)]
    elif bits == 8:
        values = [(v - 128) / 128 for v in raw]
    else:
        values = [int.from_bytes(raw[i:i + width], "little", signed=True) / (2 ** (bits - 1)) for i in range(0, len(raw), width)]
    if not all(math.isfinite(v) for v in values):
        raise InvalidAudio("non_finite_audio")
    if not any(v != 0 for v in values):
        raise InvalidAudio("all_silence")
    # Retain per-channel energy: opposite-phase channels must not be labelled silence.
    rms = math.sqrt(sum(v * v for v in values) / len(values))
    mono = [sum(values[i:i + channels]) / channels for i in range(0, len(values), channels)]
    return mono, {"sample_rate": rate, "channels": channels, "duration_s": len(mono) / rate,
                  "encoding": "float" if tag == 3 else "pcm", "bits": bits, "rms": rms}


def validate_audio(path: Path, settings: dict | None = None) -> dict:
    _, meta = read_wav(path)
    threshold = (settings or {}).get("silence_rms_max")
    if threshold is not None:
        if not isinstance(threshold, (int, float)) or not 0 <= threshold < 1:
            raise ValueError("invalid silence_rms_max")
        if meta["rms"] <= threshold:
            raise InvalidAudio("below_configured_silence_threshold")
    return meta
