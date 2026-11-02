# Paraling Bench

50题中文副语言表达评测，支持导入音频、ASR转写、JEV识别和文本裁判评分。

## 安装

Python 3.10以上，在项目根目录运行：

```powershell
python -m pip install -e .
```

## 配置

编辑 `settings.example.json`：

| 配置 | 填写内容 |
|---|---|
| generate.audio_dir | 待评测音频目录，默认predictions |
| asr.base_url / model | ASR服务地址和模型名 |
| judge.base_url / model | 文本裁判服务地址和模型名 |
| jev.url | DuplexJEV gateway地址，默认http://localhost:8020 |

ASR使用 `/audio/transcriptions`，裁判使用 `/chat/completions`；base_url包含版本前缀（如/v1），不含具体接口路径。裁判不支持JSON模式时设 `json_mode=false`，不接受temperature时设为null。

通过环境变量提供密钥：

```powershell
$env:ASR_API_KEY = "你的ASR密钥"
$env:JUDGE_API_KEY = "你的裁判密钥"
```

在 `run.example.json` 填写model_id、工具versions和生成参数；各阶段命令最后一个参数为其使用的配置文件路径。

JEV的questions定义识别问题、选项和结果映射。对应能力通过正反例核验后，将calibrated设为true；未核验时保留false，对应证据记为unknown。默认仅配置六类声音是否出现；重音、语速、拖音、停顿及位置次数需补充相应识别问题。

## 准备数据

- `data/questions.jsonl`：题号、分类、情景、台词、读法。
- `data/checks.json`：按题号配置必须有、可选、禁止出现及核心/附加检查项。
- 修改读法后同步检查评分项；正式题库须包含Q01至Q50，10类各5题。
- 将已生成的PCM WAV音频放入predictions，命名为Q01.wav至Q50.wav。

从Excel重新导出题库：

```powershell
python -m paraling_bench.cli prepare v1_50.xlsx data/questions.new.jsonl
```

## 运行

先评测少量题：

```powershell
python -m paraling_bench.cli run --dataset data/questions.jsonl --specs data/checks.json --config run.example.json --output runs/first --ids Q01,Q29,Q39,Q40
```

去掉 `--ids` 运行全部50题。续跑增加 `--resume`，配置、数据及源码必须与原运行一致；更换配置或工具版本使用新输出目录。

## 结果

输出目录中每题保存audio.wav、audio_meta.json、ASR/JEV原始结果、evidence.json、裁判响应和result.json。根目录保存manifest.json、results.jsonl和summary.json。

评分包含台词内容content、表达完成delivery、使用恰当性appropriateness，各0至2分。证据不足为null；三个维度完整时，总分为三项之和除6再乘100。summary提供分类、题型、各维度均分、有效题数及覆盖率。工具错误不计模型0分。

## 本地模型选项

Paraformer：

```powershell
python -m pip install -e ".[funasr]"
```

将run配置中asr命令最后一个参数改为 `settings.paraformer.example.json`，默认使用CPU上的paraformer-zh。首次运行下载模型。

faster-whisper：安装 `python -m pip install -e ".[asr]"`；asr配置使用provider=faster_whisper、model=small、device=cpu、compute_type=int8。

Ollama：judge配置使用provider=ollama、url=http://localhost:11434、model填写已安装模型名。

## 测试

```powershell
python -m unittest discover -s tests -v
python -m paraling_bench.cli run --dataset data/questions.jsonl --specs data/checks.json --config tests/mock_config.json --output runs/demo --ids Q01,Q29,Q39,Q40
```

mock配置使用模拟音频和预设响应，只用于检查流程。
