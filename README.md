# Paraling Bench

50 题中文副语言表达评测。Python 3.10+，在项目根目录执行命令。

## 安装

```powershell
python -m pip install -e .
# 实际使用本地强制对齐、声学分析和 AST 时安装：
python -m pip install -e ".[analysis]"
```

WhisperX 使用中文强制对齐模型；AST 使用 AudioSet 分类模型。首次使用需要下载权重，CPU 可运行，耗时取决于设备。模型依赖和服务的版本需填入运行配置。

## 准备音频与配置

将模型生成的 WAV 放入 `predictions/`，命名为 `Q01.wav` 至 `Q50.wav`。支持整数 PCM 和 float32/float64 WAV。全零静音、空音频、非有限采样和文件结构截断记为模型失败；低音量阈值可在 `run.example.json` 的 `audio_validation.silence_rms_max` 配置，默认只检查完全静音。文件完好但台词没说完，通过字错误率扣分，不能仅凭 WAV 头判断。

`data/questions.jsonl` 保存题面，`data/checks.json` 保存逐题评分要求、JEV 问题和声学定位目标。修改读法时同步修改这些配置。正式题库必须保留 50 个唯一题号。

编辑 `settings.example.json`：

| 配置 | 用途 |
|---|---|
| generate.audio_dir | 输入音频目录 |
| asr.base_url / model | OpenAI 兼容 `/audio/transcriptions` 转写接口 |
| judge.base_url / model | OpenAI 兼容 `/chat/completions` 文本裁判 |
| jev.url | DuplexJev 服务根地址，程序追加 `/v1/decide` |
| align.model / device | WhisperX 中文对齐模型；null 使用其默认模型 |
| acoustic.rules | 逐题测量指标、上下界及校准标记 |
| events.model / revision | AST 权重名称和固定版本 |
| events.features | 六类声音的标签、阈值和校准标记 |
| audio_judge.model | 支持音频输入的 Gemini 模型名称 |

```powershell
$env:ASR_API_KEY = "你的ASR密钥"
$env:JUDGE_API_KEY = "你的文本裁判密钥"
$env:GEMINI_API_KEY = "你的Gemini密钥"
```

填写 `run.example.json` 中的被测模型标识、服务版本和生成参数。各阶段都是 JSON stdin/stdout 命令，可以换成自己的调用程序。生成器返回 `status=ok` 和 `audio_path`；拒答返回 `status=refused`。生成失败和超时不自动重试，记 0 分；音频中明确说出模型拒答时，文本裁判须返回 `generation_status=refused` 并引用实际转写原文；角色台词中的拒绝不算模型拒答。无法仅靠 WAV 校验识别语义拒答。

## 校准与证据

| 题型 | 第一路 | 第二路 |
|---|---|---|
| 重音、拖音 | JEV | 强制对齐后测音高、能量、时长 |
| 语速、停顿 | 强制对齐后测字/秒、词间间隔 | Gemini 音频判断 |
| 六类声音事件 | JEV | AST 滑窗检测 |

两路针对同一检查项一致才直接采用。明确冲突才调用 Gemini 仲裁；仲裁无法判断，或任一路缺失/未校准，保留 unknown。语速/停顿仲裁是另一次 Gemini 调用，不是新增的独立模型。原始回答、各选项概率、测量值和冲突处理均落盘。

声学规则使用 `predicates`，每项指定 `metric` 与 `min`/`max`；所有条件都满足判 pass。只有上下界已填写且 `calibrated=true` 才产生判定。指标包括 `chars_per_second`（含句内停顿）、`articulation_chars_per_second`（对齐字时长总和）、目标字的时长/能量/音高比值和目标词前后停顿。缺失目标、无声基频或对齐不完整保留 unknown。对齐使用实际 ASR 文本，避免把遗漏台词强行对齐进音频。

配置中的声学阈值和 AST 阈值默认为 null，需用独立正反样本校准后填写；未提供通用“快慢”或重音阈值。AST 的标签必须与选用权重一致，时间区间为滑窗支持范围，不是精确事件边界。修改窗口或步长后需要重新校准。

JEV 每题带中文语言标记并保留缺失选项；重音、拖音、笑声、咳嗽、叹气按使用说明启用，哭声、倒吸气、清嗓默认待校准。位置答案保存在逐题结果中。JEV 配置的每个问题通过 `check_id` 和 `mapping` 关联检查项；同一项的所有问题通过才算通过。

表达检查已拆成存在、位置、次数、长短或强度；“得意、无奈、自然”等情绪只进入恰当性，不在表达完成重复扣分。每条物理检查独立取得两路证据。JEV 位置题可通过 `check_mappings` 同时提供存在和位置证据；次数、长短、强度问题使用独立 `calibration_key`，默认待校准。

AST 将连续阳性窗口合并成候选事件段，输出 `bursts`、候选段数、持续时间、事件与说话段能量比及 `position_verdict`。候选段数不天然等于真实咳嗽次数，窗口时长也不是精确事件时长：每条 `events.check_rules` 必须独立验证后设置 `calibrated=true`，长短/强度还需填写上下界。位置区间跨越词边界时保留 unknown。单次要求的数量为1，多次要求为至少2；不为“几声”假设上限。存在性校准不能替代次数和时间定位校准。

## 运行

```powershell
python -m paraling_bench.cli run --dataset data/questions.jsonl --specs data/checks.json --config run.example.json --output runs/first --ids Q01,Q29,Q39,Q40
```

去掉 `--ids` 运行全部 50 题。续跑加 `--resume`；数据、配置和源码必须一致。不同实验使用新目录。

## 分数与输出

- 内容：`5 × max(0, 1 − CER)`，CER 为字错误率；统一全半角、忽略标点和空白，保留汉字、数字及拟声词。数字中文写法与阿拉伯数字不自动等价。
- 表达完成、恰当性：分别采用 0–5 分，允许小数。表达完成从核心未实现到全部到位；恰当性从严重违背情景到自然贴合。裁判提示包含逐档标准。核心失败表达分置 0；证据不足置 null。
- 三项等权，总分为三项之和 / 15 × 100。拒答、生成超时、静音及结构损坏的音频三项均 0，保留在统计分母。
- 评测工具故障和证据不足不冒充模型失败。存在未评分项时正式 `total_mean` 为 null；`available_mean` 仅是已评分子集均值，必须同时看 `coverage`、`denominator`、`incomplete` 和状态计数。

每题保存音频及校验值、ASR/对齐/JEV/AST/Gemini 原始结果、证据、文本裁判结果和最终评分。根目录保存配置/数据/源码校验值、`results.jsonl` 和 `summary.json`。外部服务及未固定版本的权重不保证逐次复现。

基于已保存证据与裁判结果重新计算分数，不调用模型：

```powershell
python -m paraling_bench.cli score --dataset data/questions.jsonl --specs data/checks.json --run runs/first --output runs/rescored
```

评分版本 2.1 与旧版 0–2 分、失败处理及统计口径不同，结果不能直接比较。旧运行目录保留，升级后使用新目录；旧证据缺少第二路，不能直接转成新评分结果。

## 本地 ASR 选项

Paraformer：安装 `python -m pip install -e ".[funasr]"`，把运行配置的 ASR 配置路径改为 `settings.paraformer.example.json`。faster-whisper：安装 `python -m pip install -e ".[asr]"`，使用 `provider=faster_whisper`。文本裁判还支持 Ollama；默认示例为 API。

## 离线验证

```powershell
python -m unittest discover -s tests -v
python -m paraling_bench.cli run --dataset data/questions.jsonl --specs data/checks.json --config tests/mock_config.json --output runs/demo
```

模拟路径使用合成音和固定结果，不下载权重、不调用收费接口，也不验证模型识别质量。

评分 2.1 允许小数，整数档位作为判分锚点。2.0 的已保存整数评分可用 score 命令重新计算，数值保留。重新调用裁判可能得到不同的小数评分，比较实验时应统一评分版本。
