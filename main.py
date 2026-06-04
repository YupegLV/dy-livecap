"""抖音直播监控 + LLM 总结工具

用法：
    python main.py --url "https://v.douyin.com/xxx" --duration 60
    python main.py --url "https://v.douyin.com/xxx" --duration 30 --record-only
    python main.py --summarize ./output/20240101_120000/
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from src.utils import load_config, create_output_dir, save_summary
from src.live_fetcher import resolve_share_url, get_live_stream_url
from src.recorder import LiveRecorder
from src.asr import XunfeiASR, QwenASR
from src.summarizer import LiveSummarizer

console = Console(force_terminal=True)


def _safe(text: str) -> str:
    """移除终端无法显示的字符（如 emoji），防止 GBK 编码崩溃。"""
    if not text:
        return text
    return text.encode("gbk", errors="replace").decode("gbk")


def cmd_record_and_summarize(args):
    """录制直播并总结。"""
    config = load_config(args.config)

    # Step 1: 解析分享链接，获取 room_id
    console.print("[bold blue]Step 1/5: 解析直播间链接...[/bold blue]")
    try:
        room_id = resolve_share_url(args.url)
        console.print(f"  room_id: {room_id}")
    except ValueError as e:
        console.print(f"[red]链接解析失败: {e}[/red]")
        sys.exit(1)

    # Step 2: 获取直播流地址
    console.print("[bold blue]Step 2/5: 获取直播流地址...[/bold blue]")
    cookie = config.get("douyin", {}).get("cookie", "")
    live_info = get_live_stream_url(room_id, cookie)

    console.print(f"  主播: {_safe(live_info['anchor_name']) or '未知'}")
    console.print(f"  标题: {_safe(live_info['room_title']) or '未知'}")
    console.print(f"  在播: {'是' if live_info['is_living'] else '否'}")

    if not live_info["is_living"]:
        console.print("[red]该直播间当前未开播，无法录制[/red]")
        sys.exit(1)

    stream_url = live_info["stream_url"] or live_info["flv_url"]
    if not stream_url:
        console.print("[red]无法获取直播流地址，可能需要配置 cookie[/red]")
        sys.exit(1)

    console.print(f"  流地址: {stream_url[:80]}...")

    # 创建输出目录
    output_dir = create_output_dir(args.output)

    # Step 3: 录制直播
    console.print(f"[bold blue]Step 3/5: 开始录制直播（{args.duration} 分钟）...[/bold blue]")

    # 录制
    recorder = LiveRecorder(output_dir)
    record_config = config.get("record", {})
    result = recorder.record(
        stream_url=stream_url,
        duration=args.duration,
        video_format=record_config.get("format", "mp4"),
        audio_format=record_config.get("audio_format", "wav"),
    )

    if not result.get("audio_path") and not result.get("video_path"):
        console.print("[red]录制失败，没有生成任何文件[/red]")
        sys.exit(1)

    if args.record_only:
        console.print("[green]录制完成（--record-only 模式，跳过总结）[/green]")
        console.print(f"输出目录: {output_dir}")
        return

    # Step 4: ASR 转写
    console.print("[bold blue]Step 4/5: ASR 语音转写...[/bold blue]")
    audio_path = result.get("audio_path", "")

    if not audio_path:
        console.print("[red]没有音频文件，无法进行 ASR 转写[/red]")
        sys.exit(1)

    asr_config = config.get("asr", {})
    provider = asr_config.get("provider", "qwen")

    if provider == "qwen":
        qwen_cfg = asr_config.get("qwen", {})
        asr = QwenASR(
            api_key=qwen_cfg.get("api_key", ""),
            model=qwen_cfg.get("model", "qwen3-asr-flash"),
        )
    elif provider == "xunfei":
        xunfei_cfg = asr_config.get("xunfei", {})
        asr = XunfeiASR(
            app_id=xunfei_cfg.get("app_id", ""),
            api_key=xunfei_cfg.get("api_key", ""),
            api_secret=xunfei_cfg.get("api_secret", ""),
        )
    else:
        console.print(f"[red]不支持的 ASR 提供商: {provider}[/red]")
        sys.exit(1)

    try:
        asr_text = asr.transcribe(audio_path)
    except Exception as e:
        console.print(f"[red]ASR 转写失败: {e}[/red]")
        sys.exit(1)

    # 保存 ASR 文本
    asr_path = os.path.join(output_dir, "asr_text.txt")
    with open(asr_path, "w", encoding="utf-8") as f:
        f.write(asr_text)
    console.print(f"  ASR 文本已保存: {asr_path}")

    # Step 5: LLM 总结
    console.print("[bold blue]Step 5/5: LLM 生成总结...[/bold blue]")
    llm_config = config.get("llm", {})

    summarizer = LiveSummarizer(
        base_url=llm_config.get("base_url", "https://api.deepseek.com/v1"),
        api_key=llm_config.get("api_key", ""),
        model=llm_config.get("model", "deepseek-chat"),
        max_tokens=llm_config.get("max_tokens", 8192),
        multimodal=llm_config.get("multimodal", False),
    )

    # 关键帧图片（多模态模式）
    keyframes = result.get("keyframes", [])

    try:
        summary = summarizer.summarize(
            asr_text=asr_text,
            anchor_name=live_info["anchor_name"],
            duration=args.duration,
            keyframes=keyframes,
        )
    except Exception as e:
        console.print(f"[red]LLM 总结失败: {e}[/red]")
        sys.exit(1)

    # 保存总结
    summary_path = save_summary(summary, output_dir)
    console.print(f"  总结已保存: {summary_path}")

    # 输出结果
    console.print()
    console.print(Panel(_safe(summary), title="直播总结", border_style="green"))
    console.print(f"\n[green]所有文件保存在: {output_dir}[/green]")


def cmd_summarize_only(args):
    """对已录制的文件做总结。"""
    config = load_config(args.config)
    output_dir = args.summarize

    # 查找 ASR 文本
    asr_path = os.path.join(output_dir, "asr_text.txt")
    if not Path(asr_path).exists():
        console.print(f"[red]找不到 ASR 文本: {asr_path}[/red]")
        sys.exit(1)

    with open(asr_path, "r", encoding="utf-8") as f:
        asr_text = f.read()

    # LLM 总结
    console.print("[bold blue]LLM 生成总结...[/bold blue]")
    llm_config = config.get("llm", {})

    summarizer = LiveSummarizer(
        base_url=llm_config.get("base_url", "https://api.deepseek.com/v1"),
        api_key=llm_config.get("api_key", ""),
        model=llm_config.get("model", "deepseek-chat"),
        max_tokens=llm_config.get("max_tokens", 8192),
        multimodal=llm_config.get("multimodal", False),
    )

    # 查找关键帧图片
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
    )

    summary_path = save_summary(summary, output_dir)
    console.print(f"  总结已保存: {summary_path}")

    console.print()
    console.print(Panel(_safe(summary), title="直播总结", border_style="green"))


def main():
    parser = argparse.ArgumentParser(
        description="抖音直播监控 + LLM 总结工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py --url "https://v.douyin.com/xxx" --duration 60
  python main.py --url "https://v.douyin.com/xxx" --duration 30 --record-only
  python main.py --summarize ./output/20240101_120000/
        """,
    )

    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--url", help="抖音直播间分享链接")
    parser.add_argument("--duration", type=int, default=60, help="录制时长（分钟），默认60")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--record-only", action="store_true", help="只录制不总结")
    parser.add_argument("--summarize", help="对已录制的文件做总结（指定输出目录路径）")

    args = parser.parse_args()

    # 检查 ffmpeg
    if not args.summarize and not LiveRecorder.check_ffmpeg():
        console.print("[red]未检测到 ffmpeg，请先安装 ffmpeg 并添加到 PATH[/red]")
        sys.exit(1)

    if args.summarize:
        cmd_summarize_only(args)
    elif args.url:
        cmd_record_and_summarize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
