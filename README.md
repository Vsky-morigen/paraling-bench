# Paraling Bench

50 题中文副语言表达评测。需要 Python 3.10+。以下命令均在仓库根目录执行。

## 安装

```cmd
python -m pip install -e ".[analysis]"
```

使用 Montreal Forced Aligner（MFA）取得台词的逐字时间戳。安装 Conda 后，在可运行 `conda` 的终端执行：

```cmd
conda create -n paraling-mfa -c conda-forge montreal-forced-aligner
conda run -n paraling-mfa mfa model download dictionary mandarin_china_mfa
conda run -n paraling-mfa mfa model download acoustic mandarin_mfa
```

如果 MFA 已装在其他环境，调整 `settings.example.json` 的 `align.command`。AST 声音事件检测器默认用 CPU，首次使用会下载模型权重。

## 配置评测服务

在当前 CMD 会话配置密钥：

```cmd
set "OPENAI_API_KEY=你的 OpenAI API 密钥"
set "GEMINI_API_KEY=你的 Gemini API 密钥"
```

| `settings.example.json` 配置 | 用途 |
| --- | --- |
| `asr` | OpenAI 音频转写 |
| `judge` | OpenAI 文本裁判 |
| `jev.url` | DuplexJev 服务根地址，程序追加 `/v1/decide`。服务需要 Bearer Token 时，在 `jev.api_key_env` 填入对应的环境变量名。 |
| `audio_judge` | Gemini 音频判断（`gemini-3.5-flash-lite`）。按题型提供一路证据，也可在只有一路有效证据时补判。 |
| `arbitration` | OpenAI 音频仲裁（`gpt-audio-1.5`）。两路有效证据冲突、两路均为未知，或原始两路已包含音频裁判且一路未知时独立听判。 |

填写 `run.example.json` 的 `model_id` 和 `versions`，记录被测模型和各评测服务的实际版本。音频判断和最终仲裁使用不同模型。评测会向配置中的外部服务发送音频或转写。

## 接入被测模型

**使用已有音频。** 将每题的 WAV 放入 `predictions/`，文件名与题号一致，例如 `predictions/Q01.wav`。可通过 `settings.example.json` 的 `generate.audio_dir` 更改目录。支持 PCM 和 float32/float64 WAV。

**运行时生成音频。** 将 `run.example.json` 的 `generate` 替换成生成程序的命令数组，例如 `["python", "my_generator.py"]`。每道题启动一次程序。程序从标准输入读取一个 UTF-8 JSON 对象。Python 可用 `json.load(sys.stdin.buffer)` 读取：

```json
{
  "id": "Q01",
  "instruction": "任务说明",
  "scenario": "场景描述",
  "script": "需要朗读的台词",
  "direction": "表演要求",
  "parameters": {}
}
```

`parameters` 来自运行配置的 `generation_parameters`。由生成程序将这些字段转换为被测模型的接口格式。把音频保存为 WAV，向标准输出写入一个 JSON 对象，例如 `{"status":"ok","audio_path":"predictions/Q01.wav"}`。相对路径以仓库根目录为准。拒答返回 `{"status":"refused"}`，运行日志写入标准错误。

## 运行与查看结果

简易版只检查声音事件有无。复杂版按题额外检查位置和次数，哭声只检查有无和题目要求的位置。两种模式采用相同的韵律检查。

```cmd
python -m paraling_bench.cli run --dataset data/questions.jsonl --specs data/checks.json --config run.example.json --output runs/simple --mode simple
python -m paraling_bench.cli run --dataset data/questions.jsonl --specs data/checks.json --config run.example.json --output runs/complex --mode complex
```

默认运行全部 50 题。仅运行指定题号时，加 `--ids Q01,Q21`。新评测使用新目录。恢复同一运行时，使用原命令并加上 `--resume`。

- `results.jsonl` 保存逐题评分。各题目录保存音频、转写、评测证据和错误。
- `summary.json` 汇总整体、类别和任务模式的分数与覆盖率。内容分按字错误率计算。内容、表达完成度、恰当性各 0–5 分，三项等权换算为百分制。
- 恰当性必须给出数值分。证据不足时，裁判根据可用转写、情景和证据评分，并说明局限。所有信息均不足时给 2.5 分。
- 生成失败、拒答、超时、全静音及损坏的 WAV 计 0 分。评测服务故障或证据不足时，相关分数留空。存在未完成项时，`total_mean` 留空，`available_mean` 只统计已完成项，请同时查看 `coverage` 和 `incomplete`。

使用已保存的证据重新汇总分数：

```cmd
python -m paraling_bench.cli score --dataset data/questions.jsonl --specs data/checks.json --run runs/complex --output runs/rescored
```
