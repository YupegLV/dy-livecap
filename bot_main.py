"""QQ 机器人入口

整合 QQ Bot + LLM Agent + TaskManager，实现：
用户在 QQ 中发直播链接 → LLM 解析参数 → 执行录制 → 发回总结
"""

import asyncio
import os
import sys

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, create_output_dir, save_summary
from src.live_fetcher import resolve_share_url, get_live_stream_url
from src.recorder import LiveRecorder
from src.asr import QwenASR
from src.summarizer import LiveSummarizer
from src.llm_agent import LLMAgent, HELP_TEXT
from src.task_manager import TaskManager, Task, TaskStatus
from src.qq_bot import QQBotManager


def run_record_task(task: Task) -> dict:
    """在独立线程中执行录制任务。

    Returns:
        完成结果 dict: {summary, keyframes, anchor_name, output_dir}
    """
    config = load_config()

    # Step 1: 解析链接
    print(f"[Task {task.id}] 解析链接: {task.url}")
    try:
        room_id = resolve_share_url(task.url)
        print(f"[Task {task.id}] room_id: {room_id}")
    except ValueError as e:
        raise RuntimeError(f"链接解析失败: {e}")

    # Step 2: 获取直播流
    print(f"[Task {task.id}] 获取直播流地址...")
    cookie = config.get("douyin", {}).get("cookie", "")
    live_info = get_live_stream_url(room_id, cookie)

    anchor_name = live_info.get("anchor_name", "未知")
    print(f"[Task {task.id}] 主播: {anchor_name}")

    if not live_info["is_living"]:
        raise RuntimeError(f"直播间 [{anchor_name}] 当前未开播")

    stream_url = live_info["stream_url"] or live_info["flv_url"]
    if not stream_url:
        raise RuntimeError("无法获取直播流地址")

    task.progress = f"正在录制「{anchor_name}」..."

    # Step 3: 录制
    print(f"[Task {task.id}] 开始录制 {task.duration} 分钟...")
    output_dir = create_output_dir(name=task.id)
    task.output_dir = output_dir

    recorder = LiveRecorder(output_dir)
    record_config = config.get("record", {})
    result = recorder.record(
        stream_url=stream_url,
        duration=task.duration,
        video_format=record_config.get("format", "mp4"),
        audio_format=record_config.get("audio_format", "wav"),
    )

    if not result.get("audio_path") and not result.get("video_path"):
        raise RuntimeError("录制失败，没有生成任何文件")

    # 下播检测
    early_stop = result.get("early_stop", False)
    actual_duration = result.get("actual_duration", task.duration * 60) // 60
    if early_stop:
        print(f"[Task {task.id}] 主播提前下播，实际录制 {actual_duration} 分钟")

    task.progress = "录制完成，正在转写..."

    # Step 4: ASR
    print(f"[Task {task.id}] ASR 转写...")
    audio_path = result.get("audio_path", "")
    if not audio_path:
        raise RuntimeError("没有音频文件")

    asr_config = config.get("asr", {})
    qwen_cfg = asr_config.get("qwen", {})
    asr = QwenASR(
        api_key=qwen_cfg.get("api_key", ""),
        model=qwen_cfg.get("model", "qwen3-asr-flash"),
    )

    asr_text = asr.transcribe(audio_path)

    # 保存 ASR 文本
    asr_path = os.path.join(output_dir, "asr_text.txt")
    with open(asr_path, "w", encoding="utf-8") as f:
        f.write(asr_text)

    task.progress = "转写完成，正在生成总结..."

    # Step 5: LLM 总结
    print(f"[Task {task.id}] LLM 总结...")
    llm_config = config.get("llm", {})
    summarizer = LiveSummarizer(
        base_url=llm_config.get("base_url", ""),
        api_key=llm_config.get("api_key", ""),
        model=llm_config.get("model", ""),
        max_tokens=llm_config.get("max_tokens", 8192),
        multimodal=llm_config.get("multimodal", False),
    )

    keyframes = result.get("keyframes", [])
    summary = summarizer.summarize(
        asr_text=asr_text,
        anchor_name=anchor_name,
        duration=actual_duration,
        keyframes=keyframes,
        user_request=task.user_request,
    )

    # 保存总结
    save_summary(summary, output_dir)
    print(f"[Task {task.id}] 完成！输出目录: {output_dir}")

    return {
        "summary": summary,
        "asr_text": asr_text,
        "keyframes": keyframes,
        "anchor_name": anchor_name,
        "output_dir": output_dir,
        "early_stop": early_stop,
        "actual_duration": actual_duration,
    }


def recover_task(task: Task) -> dict:
    """恢复失败任务，从 ASR 步骤重新执行。

    从 task.output_dir 中找到已录制的音频，重新执行 ASR → LLM 总结。
    """
    config = load_config()
    output_dir = task.output_dir

    # 如果 task.output_dir 为空，尝试用 task_id 作为目录名查找
    if not output_dir or not os.path.isdir(output_dir):
        alt_dir = os.path.join("output", task.id)
        if os.path.isdir(alt_dir):
            output_dir = alt_dir
            task.output_dir = output_dir
        else:
            raise RuntimeError(f"任务 {task.id} 的输出目录不存在，且无法自动定位")

    # 查找音频文件（支持多种格式）
    audio_path = ""
    for audio_name in ["audio.wav", "audio.mp3", "audio.flac"]:
        candidate = os.path.join(output_dir, audio_name)
        if os.path.exists(candidate):
            audio_path = candidate
            break

    if not audio_path:
        # 尝试从视频重新提取
        audio_path = os.path.join(output_dir, "audio.wav")
        video_path = ""
        for name in ["live_record.mp4", "live_record.ts"]:
            p = os.path.join(output_dir, name)
            if os.path.exists(p):
                video_path = p
                break

        if video_path:
            print(f"[Recover {task.id}] 从视频重新提取音频: {video_path}")
            recorder = LiveRecorder(output_dir)
            audio_path = recorder._extract_audio_from_video(video_path, audio_path)
            if not audio_path:
                raise RuntimeError("音频提取失败")
        else:
            raise RuntimeError(f"找不到音频或视频文件: {output_dir}")

    task.progress = "恢复中，正在转写..."

    # ASR
    print(f"[Recover {task.id}] ASR 转写...")
    asr_config = config.get("asr", {})
    qwen_cfg = asr_config.get("qwen", {})
    asr = QwenASR(
        api_key=qwen_cfg.get("api_key", ""),
        model=qwen_cfg.get("model", "qwen3-asr-flash"),
    )
    asr_text = asr.transcribe(audio_path)

    # 保存 ASR 文本
    asr_path = os.path.join(output_dir, "asr_text.txt")
    with open(asr_path, "w", encoding="utf-8") as f:
        f.write(asr_text)

    task.progress = "转写完成，正在生成总结..."

    # LLM 总结
    print(f"[Recover {task.id}] LLM 总结...")
    llm_config = config.get("llm", {})
    summarizer = LiveSummarizer(
        base_url=llm_config.get("base_url", ""),
        api_key=llm_config.get("api_key", ""),
        model=llm_config.get("model", ""),
        max_tokens=llm_config.get("max_tokens", 8192),
        multimodal=llm_config.get("multimodal", False),
    )

    # 查找关键帧
    keyframes = []
    keyframes_dir = os.path.join(output_dir, "keyframes")
    if os.path.isdir(keyframes_dir):
        keyframes = sorted([
            os.path.join(keyframes_dir, f)
            for f in os.listdir(keyframes_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ])

    summary = summarizer.summarize(
        asr_text=asr_text,
        keyframes=keyframes,
        user_request=task.user_request,
    )

    save_summary(summary, output_dir)
    print(f"[Recover {task.id}] 恢复完成！输出目录: {output_dir}")

    return {
        "summary": summary,
        "asr_text": asr_text,
        "keyframes": keyframes,
        "anchor_name": "",
        "output_dir": output_dir,
        "early_stop": False,
        "actual_duration": task.duration,
    }


class BotApp:
    """QQ 机器人应用。"""

    def __init__(self):
        self.config = load_config()
        self.qq_bot: QQBotManager | None = None
        self.agent: LLMAgent | None = None
        self.task_manager: TaskManager | None = None

    def run(self):
        """启动机器人。"""
        # 初始化 LLM Agent
        llm_cfg = self.config.get("llm", {})
        self.agent = LLMAgent(
            base_url=llm_cfg.get("base_url", ""),
            api_key=llm_cfg.get("api_key", ""),
            model=llm_cfg.get("model", ""),
        )

        # 初始化 TaskManager
        self.task_manager = TaskManager(on_complete=self._on_task_complete)

        # 初始化 QQ Bot
        qq_cfg = self.config.get("qq", {})
        appid = qq_cfg.get("appid", "")
        secret = qq_cfg.get("secret", "")

        if not appid or not secret:
            print("错误：请配置 QQ_BOT_APPID 和 QQ_BOT_SECRET 环境变量")
            print("在 QQ 开放平台 (q.qq.com) 创建机器人后获取")
            sys.exit(1)

        self.qq_bot = QQBotManager(appid, secret)
        self.qq_bot.set_message_handler(self._handle_message)
        self.qq_bot.start()

        print("=" * 50)
        print("直播监控 QQ 机器人已启动")
        print("在 QQ 中给机器人发消息即可使用")
        print("=" * 50)

        # 主线程保持运行
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在退出...")

    async def _handle_message(self, content: str, user_openid: str,
                               msg_id: str, api):
        """处理收到的 QQ 消息。"""
        print(f"[Bot] 收到消息: {content[:100]}")

        # LLM 解析意图
        intent = self.agent.parse_intent(content)
        print(f"[Bot] 意图: {intent}")

        if intent["action"] == "record":
            url = intent["url"]
            duration = intent["duration"]
            user_request = intent.get("user_request", "")

            # 提交录制任务
            task_id = self.task_manager.submit(
                url=url,
                duration=duration,
                user_openid=user_openid,
                run_fn=run_record_task,
                user_request=user_request,
            )

            reply = f"开始录制直播，时长 {duration} 分钟。\n任务ID: {task_id}\n录制完成后会自动发送总结。"
            await self.qq_bot.send_text(user_openid, reply, msg_id)

        elif intent["action"] == "recover":
            task_id = intent.get("task_id", "")
            output_dir = os.path.join("output", task_id)
            if not os.path.isdir(output_dir):
                await self.qq_bot.send_text(
                    user_openid, f"找不到任务 {task_id}，请确认任务ID是否正确。", msg_id
                )
                return

            # 先尝试从内存中找到已有 Task，否则创建临时 Task
            task = self.task_manager.get_task(task_id)
            if not task:
                task = Task(
                    id=task_id,
                    url="",
                    duration=60,
                    user_openid=user_openid,
                    output_dir=output_dir,
                )
                with self.task_manager._lock:
                    self.task_manager._tasks[task_id] = task

            self.task_manager.recover(task_id, recover_task)
            reply = f"正在恢复任务 {task_id}，从 ASR 步骤重新执行..."
            await self.qq_bot.send_text(user_openid, reply, msg_id)

        elif intent["action"] == "help":
            await self.qq_bot.send_text(user_openid, HELP_TEXT, msg_id)

        elif intent["action"] == "chat":
            # 多轮对话：优先按用户指定的 task_id 查找，否则取最近完成的任务
            task_id = intent.get("task_id")
            if task_id:
                task = self.task_manager.get_user_task(user_openid, task_id)
                if not task:
                    await self.qq_bot.send_text(
                        user_openid, f"找不到任务 {task_id}，请确认任务ID是否正确。", msg_id
                    )
                    return
                if task.status != TaskStatus.COMPLETED:
                    await self.qq_bot.send_text(
                        user_openid, f"任务 {task_id} 尚未完成，无法进行追问。", msg_id
                    )
                    return
            else:
                task = self.task_manager.get_user_latest_completed_task(user_openid)
            if not task:
                await self.qq_bot.send_text(
                    user_openid,
                    "你还没有完成的直播录制任务。发送直播链接开始录制吧！",
                    msg_id,
                )
                return

            result = task.result
            asr_text = result.get("asr_text", "")
            previous_summary = result.get("summary", "")
            anchor_name = result.get("anchor_name", "")
            actual_duration = result.get("actual_duration", task.duration)

            # 如果 result 中没有 asr_text，从文件读取
            if not asr_text and task.output_dir:
                asr_path = os.path.join(task.output_dir, "asr_text.txt")
                if os.path.exists(asr_path):
                    with open(asr_path, "r", encoding="utf-8") as f:
                        asr_text = f.read()

            user_message = intent.get("user_request", content)

            # 调用 LLM 多轮对话
            answer = self.agent.chat_with_context(
                user_message=user_message,
                asr_text=asr_text,
                previous_summary=previous_summary,
                chat_history=task.chat_history,
                anchor_name=anchor_name,
                duration=actual_duration,
            )

            # 更新对话历史
            task.chat_history.append({"role": "user", "content": user_message})
            task.chat_history.append({"role": "assistant", "content": answer})
            # 只保留最近 20 条对话（10 轮）
            if len(task.chat_history) > 20:
                task.chat_history = task.chat_history[-20:]

            # 发送回答
            reply = f"「{anchor_name}」直播追问：\n\n{answer}" if anchor_name else answer
            await self.qq_bot.send_text(user_openid, reply, msg_id)

        elif intent["action"] == "query":
            # 查询用户任务状态
            tasks = self.task_manager.get_user_tasks(user_openid)
            if not tasks:
                await self.qq_bot.send_text(
                    user_openid, "你当前没有进行中的任务。发送直播链接开始录制！", msg_id
                )
            else:
                lines = []
                for t in tasks[-3:]:  # 只显示最近3个
                    status_text = {
                        TaskStatus.PENDING: "等待中",
                        TaskStatus.RUNNING: f"执行中 - {t.progress}",
                        TaskStatus.COMPLETED: "已完成",
                        TaskStatus.FAILED: f"失败 - {t.error}",
                    }.get(t.status, str(t.status))
                    lines.append(f"任务 {t.id}: {t.url[:30]}... | {status_text}")
                await self.qq_bot.send_text(user_openid, "\n".join(lines), msg_id)

        else:
            await self.qq_bot.send_text(
                user_openid,
                "你好！我是直播监控助手。\n发送抖音直播链接即可开始录制和总结。\n发送\"帮助\"查看使用说明。",
                msg_id,
            )

    def _on_task_complete(self, task: Task):
        """任务完成回调（在任务线程中调用，用 sync 方法发消息）。"""
        if not self.qq_bot:
            return

        user_openid = task.user_openid

        if task.status == TaskStatus.COMPLETED:
            result = task.result
            summary = result.get("summary", "总结生成失败")
            keyframes = result.get("keyframes", [])
            anchor_name = result.get("anchor_name", "")
            early_stop = result.get("early_stop", False)
            actual_duration = result.get("actual_duration", task.duration)

            header = f"「{anchor_name}」直播总结已完成！"
            if early_stop:
                header += f"\n（主播提前下播，实际录制 {actual_duration} 分钟，计划 {task.duration} 分钟）"
            header += "\n\n"
            self.qq_bot.send_result_sync(
                user_openid, header + summary, keyframes
            )

        elif task.status == TaskStatus.FAILED:
            self.qq_bot.send_text_sync(
                user_openid, f"录制任务失败: {task.error}"
            )


def main():
    app = BotApp()
    app.run()


if __name__ == "__main__":
    main()
