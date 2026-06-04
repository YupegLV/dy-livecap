"""FFmpeg 录制直播流"""

import os
import subprocess
import time
from pathlib import Path


class LiveRecorder:
    """使用 FFmpeg 录制直播流，录制结束后提取音频。"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._process: subprocess.Popen | None = None

    def record(
        self,
        stream_url: str,
        duration: int = 60,
        video_format: str = "ts",
        audio_format: str = "wav",
    ) -> dict:
        """录制直播流。

        Args:
            stream_url: m3u8/flv 直播流地址
            duration: 录制时长（分钟）
            video_format: 视频保存格式（推荐 ts，中断不损坏）
            audio_format: 音频保存格式（供 ASR 使用）

        Returns:
            {"video_path": str, "audio_path": str, "keyframes": list,
             "early_stop": bool, "actual_duration": int}
        """
        duration_sec = duration * 60
        ts_path = self.output_dir / "live_record.ts"
        mp4_path = self.output_dir / "live_record.mp4"
        audio_path = self.output_dir / f"audio.{audio_format}"

        video_cmd = [
            "ffmpeg",
            "-y",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", stream_url,
            "-c", "copy",
            "-t", str(duration_sec),
            str(ts_path),
        ]

        print(f"[录制] 开始录制，时长 {duration} 分钟...")
        print(f"[录制] 视频保存至: {ts_path}")

        early_stop = False
        actual_duration_sec = 0

        # 用线程异步读取 ffmpeg stderr，避免管道阻塞
        import threading
        ffmpeg_log = []

        def _read_stderr(proc):
            try:
                for line in proc.stderr:
                    ffmpeg_log.append(line.decode(errors="replace").rstrip())
            except ValueError:
                pass

        try:
            self._process = subprocess.Popen(
                video_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )

            stderr_thread = threading.Thread(target=_read_stderr, args=(self._process,), daemon=True)
            stderr_thread.start()

            start = time.time()
            last_print = 0
            while self._process.poll() is None:
                elapsed = time.time() - start
                if elapsed >= duration_sec + 30:
                    # 超过设定时长30秒 ffmpeg 仍未退出，强制终止
                    print(f"\n[录制] ffmpeg 超时未退出，强制终止...")
                    self._graceful_stop()
                    break
                now = time.time()
                if now - last_print >= 30:
                    remaining = max(0, duration_sec - elapsed)
                    print(f"[录制] 已录制 {int(elapsed // 60)}分{int(elapsed % 60)}秒，剩余 {int(remaining // 60)}分{int(remaining % 60)}秒")
                    last_print = now
                time.sleep(1)

            # ffmpeg 已退出，检查退出码
            retcode = self._process.returncode
            actual_duration_sec = int(time.time() - start)

            if retcode != 0 and retcode is not None:
                # 打印 ffmpeg 错误信息（最后10行）
                tail = ffmpeg_log[-10:] if ffmpeg_log else []
                if tail:
                    print(f"[录制] ffmpeg 退出码={retcode}，错误信息:")
                    for line in tail:
                        print(f"  {line}")

            if actual_duration_sec < duration_sec - 30:
                print(f"[录制] 直播流提前结束，实际录制 {actual_duration_sec // 60}分{actual_duration_sec % 60}秒（计划 {duration} 分钟），可能主播已下播")
                early_stop = True
            else:
                print(f"[录制] 录制完成，共 {actual_duration_sec // 60}分{actual_duration_sec % 60}秒")

        except KeyboardInterrupt:
            print("\n[录制] 用户中断，正在停止录制...")
            self._graceful_stop()
        finally:
            self._process = None

        if actual_duration_sec == 0:
            actual_duration_sec = int(time.time() - start) if 'start' in dir() else 0

        # 后处理：ts → mp4 → 音频/关键帧
        result = {"video_path": "", "audio_path": "", "keyframes": [],
                  "early_stop": early_stop, "actual_duration": actual_duration_sec}

        if ts_path.exists() and ts_path.stat().st_size > 0:
            print(f"[录制] 视频录制完成: {ts_path} ({ts_path.stat().st_size / 1024 / 1024:.1f} MB)")

            # 转换为 mp4
            print("[录制] 正在转换为 mp4...")
            self._ts_to_mp4(str(ts_path), str(mp4_path))

            video_to_use = str(mp4_path) if mp4_path.exists() and mp4_path.stat().st_size > 0 else str(ts_path)
            result["video_path"] = video_to_use

            # 抽取关键帧
            result["keyframes"] = self.extract_keyframes(video_to_use, interval=30)

            # 从视频提取音频
            result["audio_path"] = self._extract_audio_from_video(
                video_to_use, str(audio_path)
            )
        else:
            print("[录制] 视频录制失败或文件为空")

        return result

    def _graceful_stop(self):
        """优雅地停止 ffmpeg：发送 'q' 命令让其正常关闭输出文件。"""
        if self._process and self._process.poll() is None:
            try:
                self._process.communicate(input=b"q", timeout=15)
            except subprocess.TimeoutExpired:
                print("[录制] 优雅退出超时，强制终止...")
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    self._process.kill()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    def _ts_to_mp4(self, ts_path: str, mp4_path: str):
        """将 ts 文件转换为 mp4（不转码，只改容器）。"""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", ts_path,
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            mp4_path,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=120)
            if Path(mp4_path).exists() and Path(mp4_path).stat().st_size > 0:
                print(f"[录制] 转换完成: {mp4_path}")
                Path(ts_path).unlink(missing_ok=True)
            else:
                print("[录制] mp4 转换失败，保留 ts 文件")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[录制] mp4 转换失败: {e}，保留 ts 文件")

    def _extract_audio_from_video(
        self, video_path: str, audio_path: str
    ) -> str:
        """从已录制的视频文件中提取音频。"""
        if not Path(video_path).exists():
            return ""

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
        ]

        print("[录制] 正在从视频中提取音频...")

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=300)
            if Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
                size_mb = Path(audio_path).stat().st_size / 1024 / 1024
                print(f"[录制] 音频提取完成: {audio_path} ({size_mb:.1f} MB)")
                return audio_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        print("[录制] 音频提取失败")
        return ""

    def extract_keyframes(self, video_path: str, interval: int = 30) -> list[str]:
        """从视频中按固定间隔抽取关键帧截图。"""
        keyframes_dir = self.output_dir / "keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            str(keyframes_dir / "frame_%04d.jpg"),
        ]

        print(f"[录制] 抽取关键帧（每 {interval} 秒一帧）...")

        try:
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=120
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("[录制] 关键帧抽取失败")
            return []

        frames = sorted(keyframes_dir.glob("frame_*.jpg"))
        print(f"[录制] 抽取了 {len(frames)} 张关键帧")

        return [str(f) for f in frames]

    @staticmethod
    def check_ffmpeg() -> bool:
        """检查系统是否安装了 ffmpeg。"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except FileNotFoundError:
            return False
