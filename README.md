# 非语言表达评测 v0.1

依据本目录AGENTS.md及飞书50题版本实现。输入情景、台词、读法，评台词内容、表达完成、使用恰当性。保留原Excel；data/questions.jsonl为完整50题快照，data/checks.json只拆解现有读法，运行前可审查并冻结。

## 安装

Python 3.10以上，在本目录运行：

```powershell
python -m pip install -e .
python -m pip install -e ".[asr]"
```

第二条仅在使用本地faster-whisper时需要。首次使用会下载模型。CPU int8可运行，但速度取决于机器。默认ASR和文本裁判均调用远程兼容API；费用取决于所选服务，代码不承诺免费。保留Ollama及本地ASR选项。

## 最小验证（无外网、无付费调用）

```powershell
python -m unittest discover -s tests -v
python -m paraling_bench.cli run --dataset data/questions.jsonl --specs data/checks.json --config tests/mock_config.json --output runs/demo --ids Q01,Q29,Q39,Q40
```

模拟服务产生静音与预设响应，只测试程序。manifest明确标记simulation，不可作为模型成绩。重复命令须增加--resume或更换输出目录。

## 真实评测

1. 将被测模型已生成的PCM WAV放入predictions/Q01.wav等；缺音频记录工具错误，不能算0分。
2. 编辑settings.example.json：ASR与裁判的base_url、model、api_key_env；JEV服务地址及已校准识别问题。
3. 编辑run.example.json中的被测模型和工具版本。配置不存密钥，鉴权仅通过api_key_env引用环境变量。
4. 运行：

```powershell
python -m paraling_bench.cli run --dataset data/questions.jsonl --specs data/checks.json --config run.example.json --output runs/first --ids Q01,Q29,Q39,Q40
```

去掉--ids运行全50题。先用少量题核对接口、识别能力，再正式运行。没有被测模型时，音频导入就是生成阶段；日后只需替换generate命令。所有阶段命令为参数数组，不通过shell执行。

## 接口与证据

每个适配程序从stdin读JSON，stdout只输出JSON。generate接收id/instruction/scenario/script/direction/parameters，返回audio_path或明确的status=model_failure；asr/jev仅接收audio_path，不接收场景或评分答案。judge接收item/evaluation_spec/evidence。

ASR返回status、spoken_text。JEV返回status、facts，每条fact含唯一id、feature、observed（present/absent/unknown）、description；不可提供的位置、次数、时长保持null。裁判输出格式见adapters.py中的schema。裁判必须引用声音证据；程序复算delivery和总分。

当前JEV默认问题只覆盖六种声音事件，calibrated=false意味着答案仅供诊断，不能直接作为正式评分证据。请通过正反例核验后将对应问题设为true。概率不是已经校准的正确率，未设置任意阈值。重音、语速、拖音、停顿及事件位置次数需要补充经过验证的问题配置；未支持项为unknown，相关成绩null。不能因ASR没识别出文字就认定没有笑声。

普通ASR不能可靠区分所有拟声词和非语言事件。本版本不全局删除“哈哈”“咳咳”；遇到转写与声音冲突应由裁判标null。尚未实现经过时间对齐验证的事件感知转写归一，不能宣称已彻底解决转写误扣分。

## 评分和输出

三维0/1/2/null。delivery：core失败为0，否则未知为null，否则附加要求失败为1，否则2。optional可靠未出现时允许通过；prohibited出现时核心失败。content和appropriateness由裁判给出并校验证据。总分仅三维完整时为100乘三维和除6。

结果包含逐题JSON、原音频及SHA256、识别原始响应、证据、裁判响应、summary.json、results.jsonl。汇总有每类、每维、题型、有效题数和覆盖率。不同模型只在共同可评分题集上比较；目前不提供自动跨运行排行榜。

接口错误最多重试一次；成功音频固定后不因低分重新生成。续跑验证题库、检查项、配置和源码哈希，已完成音频校验哈希。更换裁判版本请新建运行目录，generate指向旧运行音频目录即可复用音频。当前不跨目录自动复用识别缓存。

## 官方接口依据

- DuplexJEV: https://github.com/adventists-ai/duplexjev/blob/main/duplexjev/remote.py
  使用POST /v1/decide，audio_b64、format、lang、questions，读取answers。支持自建gateway。官方https://api.adventists.cn/duplexjev为限流试用服务，默认配置采用localhost；如使用试用服务，音频会上传该服务。
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Ollama: https://docs.ollama.com/api/chat

接口依据核对于2026-11-02。真实服务尚未运行验证；本地模型的识别与评分质量须另行校准。JEV权重与托管服务许可请按官方说明确认。


## API接入（2026-11-02更新）

默认settings.example.json已经改为ASR和文本裁判均通过API调用。替换YOUR_ASR_HOST、YOUR_JUDGE_HOST和模型名，并在PowerShell设置ASR_API_KEY、JUDGE_API_KEY环境变量。密钥不写入JSON。两者可以使用不同供应商。兼容协议不意味着所有供应商都支持；需确认对应服务提供下列接口。

- 裁判：POST {base_url}/chat/completions，JSON messages；读取choices[0].message.content中的JSON对象。
- ASR：POST {base_url}/audio/transcriptions，以multipart上传file、model、language和response_format=json；读取text。不发送参考台词，避免提示词泄漏污染识别。
- base_url须包含供应商要求的版本前缀（如/v1），不重复填写具体路由。不支持JSON模式的裁判可设json_mode=false，不接受temperature的模型可设为null。协议错误、拒答、非法JSON进入现有错误/复判流程，不编造分数。
- 原Ollama仍可配置provider=ollama、url、model；默认示例不再使用Ollama。

若希望按NVV-SuperBench的中文ASR路线，使用settings.paraformer.example.json中的ASR配置（provider=funasr、model=paraformer-zh），将run配置中asr命令最后一个参数改为该文件，并安装：

```powershell
python -m pip install -e ".[funasr]"
```

这是本地模型，首次运行下载权重，推理无需ASR服务费用。默认CPU；GPU需匹配torch与torchaudio。原faster-whisper通过provider=faster_whisper继续可用。

调研依据：NVV-SuperBench中文使用paraformer-zh、英文使用Whisper-large-v3；NV-Bench使用专门的Multilingual-NVASR进行文字与事件联合转写。它们没有统一采用云ASR API。本项目保留JEV识别路线，不将普通ASR宣称为NVASR。

来源：
- https://github.com/lmxue/NVV-SuperBench/tree/main/evaluation/objective_evaluation/wer_cer
- https://github.com/AmphionTeam/NV-Bench
- https://github.com/modelscope/FunASR

本次API适配通过模拟HTTP响应验证；没有使用真实密钥调用付费服务，也未下载Paraformer权重。ASR和裁判服务替换后请更新run配置中的versions，并新建运行目录。
