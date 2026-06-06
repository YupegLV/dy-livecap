# dy-livecap

抖音直播录制 + ASR 转写 + LLM 智能总结工具。

发送一个直播间链接，自动完成：录制直播 → 语音转文字 → AI 生成结构化总结。支持 QQ 机器人交互和多轮追问。

## 功能

- **直播录制** — 输入抖音分享链接，自动解析直播间并录制直播流（m3u8/flv）
- **ASR 转写** — 支持 Qwen3-ASR-Flash（DashScope）和讯飞两个语音识别引擎
- **LLM 总结** — 基于转写文本生成结构化直播总结，支持超长文本 Map-Reduce 分段总结
- **多模态总结** — 自动抽取关键帧截图，结合画面+语音进行图文总结
- **弹幕收集** — 通过 WebSocket 抓取直播间弹幕，纳入总结分析
- **QQ 机器人** — 基于 QQ 开放平台（[q.qq.com](https://q.qq.com)）注册机器人，在 QQ 私聊中发送直播链接即可触发录制，完成后自动回送总结
- **多轮追问** — 基于直播上下文连续提问，支持指定任务 ID 追问不同直播
- **任务管理** — 异步录制任务，支持状态查询和失败恢复

## 工作流程

```
抖音分享链接 → 解析直播间 → 录制直播流 → 提取音频 → ASR 转写 → LLM 总结 → 输出结果
                              ↓
                         抽取关键帧截图（多模态）
                              ↓
                         收集弹幕（可选）
```

## 快速开始

### 环境要求

- Python 3.12+
- ffmpeg（需加入 PATH）

### 安装

```bash
git clone <repo-url>
cd douyin-zhibo
pip install -r requirements.txt
```

### 配置

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 编辑 `.env`，填入 API 密钥：

```env
# ASR（二选一）
DASHSCOPE_API_KEY=sk-xxx              # Qwen ASR（推荐）
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_ASR_MODEL=qwen3-asr-flash

# LLM 总结
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat
```

3. 按需修改 `config.yaml`：

```yaml
record:
  format: mp4
  audio_format: wav

asr:
  provider: qwen    # qwen | xunfei

llm:
  max_tokens: 8192
  multimodal: true   # 开启关键帧图文总结

qq:                  # QQ 机器人模式才需要
  # QQ_BOT_APPID / QQ_BOT_SECRET 通过 .env 设置
```

## 使用方式

### QQ 机器人

1. 在 [QQ 开放平台](https://q.qq.com) 创建机器人，获取 AppID 和 Secret
2. 在 `.env` 中配置：

```env
QQ_BOT_APPID=your_appid
QQ_BOT_SECRET=your_secret
```

3. 启动机器人：

```bash
python bot_main.py
```

4. 在 QQ 中给机器人发消息：

| 消息 | 效果 |
|------|------|
| 直接发送直播分享链接 | 默认录制 1 小时并总结 |
| `帮我录这个直播30分钟 [链接]` | 录制 30 分钟 |
| `看看这个直播在干嘛 [链接]` | 快速查看 10 分钟 |
| `录1小时，重点看技术内容 [链接]` | 按需求重点总结 |
| `刚才直播讲了什么` | 多轮追问（最近完成的任务） |
| `换个角度总结` | 重新组织总结 |
| `任务20260602_144803讲了什么技术` | 指定任务 ID 进行追问 |
| `恢复 20260602_144803` | 恢复失败任务 |
| `帮助` | 查看使用说明 |

### Docker

```bash
# 构建镜像
docker compose build

# 启动（QQ Bot 模式）
docker compose up -d
```

## 输出文件

每次录制会在 `output/` 下创建以时间戳命名的目录：

```
output/20260602_144803/
├── live_record.mp4      # 录制视频
├── audio.wav            # 提取的音频
├── asr_text.txt         # ASR 转写文本
├── summary.md           # LLM 生成的总结
├── danmaku.txt          # 弹幕记录（如有）
└── keyframes/           # 关键帧截图
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    └── ...
```

## 项目结构

```
├── main.py              # 命令行入口
├── bot_main.py          # QQ 机器人入口
├── config.yaml          # 配置文件
├── .env.example         # 环境变量模板
├── requirements.txt     # Python 依赖
├── Dockerfile
├── docker-compose.yml
└── src/
    ├── live_fetcher.py  # 直播链接解析 & 流地址获取
    ├── recorder.py      # FFmpeg 直播录制
    ├── asr/             # ASR 语音转写
    │   ├── base.py
    │   ├── qwen_asr.py  # Qwen ASR（DashScope）
    │   └── xunfei.py    # 讯飞 ASR
    ├── summarizer.py    # LLM 总结（含 Map-Reduce & 多模态）
    ├── llm_agent.py     # LLM 意图解析 & 多轮对话
    ├── danmaku.py       # 弹幕收集（WebSocket）
    ├── qq_bot.py        # QQ Bot 封装
    ├── task_manager.py  # 异步任务管理
    └── utils.py         # 工具函数
```

## 依赖说明

| 依赖 | 用途 |
|------|------|
| openai | LLM API 调用（OpenAI 兼容接口） |
| requests | HTTP 请求（直播流获取） |
| websocket-client | 弹幕 WebSocket 连接 |
| pyyaml | 配置文件解析 |
| rich | 命令行美化输出 |
| qq-botpy | QQ 机器人 SDK |
| pydub | 音频处理 |

## 注意事项

- 录制依赖 **ffmpeg**，请确保已安装并加入系统 PATH
- 部分直播间获取流地址需要配置抖音 Cookie，可在 `config.yaml` 的 `douyin.cookie` 中设置
- LLM 总结使用 OpenAI 兼容 API，支持 DeepSeek、Qwen 等任何兼容接口
- 多模态总结需要 LLM 模型支持视觉输入（图片理解）
- 弹幕收集为实验性功能，稳定性取决于抖音接口变化
