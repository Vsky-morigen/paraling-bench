"""MFA CLI adapter. Character tokens preserve measured Chinese character boundaries."""
import json
import math
import subprocess
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from .audio import read_wav
from .core import normalize_text


def parse_alignment(raw: dict, text: str, duration: float) -> list[dict]:
    tiers = raw.get('tiers', {})
    if isinstance(tiers, dict):
        tiers = [dict(value, name=name) for name, value in tiers.items()]
    words = [t for t in tiers if t.get('name', '').split(' - ')[-1] == 'words']
    if len(words) != 1:
        raise ValueError('MFA output must contain exactly one words tier')
    chars = []
    previous = 0.0
    for start, end, label in words[0].get('entries', []):
        label = str(label).strip()
        if label in {'', '<eps>'}:
            continue
        if len(label) != 1 or normalize_text(label) != label:
            raise ValueError(f'MFA non-character/OOV token: {label!r}; cannot invent character boundaries')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)) or not previous <= start < end <= duration + 0.001:
            raise ValueError('MFA invalid/nonmonotonic timestamps')
        chars.append({'char': label, 'start': start, 'end': min(end, duration)})
        previous = end
    if ''.join(c['char'] for c in chars) != text:
        raise ValueError('MFA alignment does not cover the actual ASR transcript')
    return chars


def align_mfa(settings: dict, request: dict) -> dict:
    text = normalize_text(request.get('spoken_text') or '')
    if not text:
        raise ValueError('ASR text unavailable for MFA alignment')
    if any(not '\u4e00' <= c <= '\u9fff' for c in text):
        raise ValueError('MFA character alignment currently requires Chinese characters; resolve numbers/foreign words in ASR transcript first')
    audio = Path(request['audio_path']).resolve()
    _, meta = read_wav(audio)
    work = Path(settings.get('work_dir', 'runs/mfa')).resolve() / uuid.uuid4().hex
    work.mkdir(parents=True)
    transcript = work / 'speech.lab'
    transcript.write_text(' '.join(text), encoding='utf-8')
    output = work / 'alignment.json'
    prefix = settings.get('command', ['mfa'])
    if not isinstance(prefix, list) or not prefix or not all(isinstance(x, str) for x in prefix):
        raise ValueError('align.command must be a nonempty argument list')
    with TemporaryDirectory(prefix='temp-', dir=work) as scratch:
        command = prefix + ['align_one', str(audio), str(transcript), settings['dictionary'],
                            settings['acoustic_model'], str(output), '--output_format', 'json',
                            '--no_tokenization', '--temporary_directory', scratch]
        (work / 'command.json').write_text(json.dumps(command, ensure_ascii=False), encoding='utf-8')
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', timeout=settings.get('timeout_s', 600))
        except (OSError, subprocess.TimeoutExpired) as exc:
            (work / 'error.txt').write_text(str(exc), encoding='utf-8')
            raise RuntimeError(f"{request.get('id', 'unknown')} align/MFA failed; artifacts: {work}: {exc}") from exc
    (work / 'stdout.log').write_text(result.stdout, encoding='utf-8')
    (work / 'stderr.log').write_text(result.stderr, encoding='utf-8')
    if result.returncode or not output.is_file():
        raise RuntimeError(f"{request.get('id', 'unknown')} align/MFA exit={result.returncode}; logs: {work}")
    raw = json.loads(output.read_text(encoding='utf-8-sig'))
    chars = parse_alignment(raw, text, meta['duration_s'])
    return {'status': 'ok', 'chars': chars, 'text': text, 'raw': raw,
            'model': settings['acoustic_model'], 'dictionary': settings['dictionary'],
            'provider': 'mfa', 'artifacts': str(work)}
