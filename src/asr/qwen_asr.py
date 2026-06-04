"""Qwen3-ASR-Flash / Paraformer 语音识别实现

通过阿里云 DashScope 异步文件转写 API 调用。

注意：Qwen3-ASR-Flash 目前需要通过 DashScope 专属端点调用，
如果 qwen3-asr-flash 不可用，会自动回退到 paraformer-v2。

大文件会自动按段拆分转写，避免超大 POST body 导致 SSL 断开。

文档：https://help.aliyun.com/zh/model-studio/developer-reference/qwen3-asr-flash
"""

import base64
import os
import tempfile
import time
from pathlib import Path

import requests
from pydub import AudioSegment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import ASRBase


# DashScope 异步转写 API
SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
QUERY_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

# 支持的模型列表（按优先级）
MODELS = ["qwen3-asr-flash", "paraformer-v2"]

# 单段最大时长（毫秒），5 分钟
SEGMENT_MAX_MS = 5 * 60 * 1000


def _build_session() -> requests.Session:
    """构建带 SSL 重试的 requests Session。"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _split_audio(audio_path: str, segment_ms: int = SEGMENT_MAX_MS) -> list[str]:
    """将音频文件按指定时长拆分，返回临时文件路径列表。"""
    ext = Path(audio_path).suffix.lower().lstrip(".")
    fmt = {"wav": "wav", "mp3": "mp3", "flac": "flac"}.get(ext, "wav")

    audio = AudioSegment.from_file(audio_path, format=fmt)
    total_ms = len(audio)

    if total_ms <= segment_ms:
        return [audio_path]

    # 最小有效段时长：1秒，低于此视为空段（如整除后的尾巴）
    min_segment_ms = 1000

    segments = []
    for i in range(0, total_ms, segment_ms):
        end = min(i + segment_ms, total_ms)
        if end - i < min_segment_ms:
            continue
        chunk = audio[i:end]
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        chunk.export(tmp.name, format=fmt)
        segments.append(tmp.name)
        tmp.close()

    print(f"[ASR] 音频已拆分为 {len(segments)} 段 (每段 ≤ {segment_ms // 1000}s)")
    return segments


class QwenASR(ASRBase):
    """阿里云 DashScope ASR 实现。

    支持 Qwen3-ASR-Flash 和 Paraformer-v2 模型。
    如果指定模型不可用，自动回退到 paraformer-v2。
    大文件自动拆段转写后合并。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-asr-flash",
        base_url: str = "",
    ):
        self.api_key = api_key
        self.model = model
        self._session = _build_session()

    def transcribe(self, audio_path: str) -> str:
        """将音频文件转写为文本。大文件自动拆段。"""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        file_size = Path(audio_path).stat().st_size
        file_size_mb = file_size / 1024 / 1024
        print(f"[ASR] 开始转写，文件大小: {file_size_mb:.1f} MB, 模型: {self.model}")

        # 拆分音频
        segment_paths = _split_audio(audio_path)
        is_split = len(segment_paths) > 1

        # 确定使用哪个模型
        models_to_try = [self.model]
        if self.model != "paraformer-v2":
            models_to_try.append("paraformer-v2")

        try:
            for model in models_to_try:
                try:
                    results = []
                    for idx, seg_path in enumerate(segment_paths):
                        if is_split:
                            print(f"[ASR] 转写第 {idx + 1}/{len(segment_paths)} 段...")

                        text = self._transcribe_segment(
                            seg_path, model, idx, len(segment_paths)
                        )
                        results.append(text)

                    # 合并结果
                    combined = "\n".join(results)
                    print(f"[ASR] 转写完成，文本长度: {len(combined)} 字")
                    return combined

                except Exception as e:
                    print(f"[ASR] 模型 {model} 失败: {e}")
                    if model != models_to_try[-1]:
                        print(
                            f"[ASR] 回退到 {models_to_try[models_to_try.index(model) + 1]}..."
                        )
                    else:
                        raise RuntimeError(f"ASR 转写失败: {e}")
        finally:
            # 清理临时拆分文件（在所有模型都尝试完之后）
            if is_split:
                for seg_path in segment_paths:
                    try:
                        os.unlink(seg_path)
                    except OSError:
                        pass

        raise RuntimeError("ASR 转写失败: 所有模型均失败")

    def _transcribe_segment(
        self, seg_path: str, model: str, idx: int, total: int,
        max_retries: int = 2,
    ) -> str:
        """转写单个音频段，失败自动重试。"""
        ext = Path(seg_path).suffix.lower().lstrip(".")
        audio_format = {"wav": "wav", "mp3": "mp3", "flac": "flac"}.get(ext, "wav")

        with open(seg_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                task_id = self._submit_task_with_retry(audio_b64, audio_format, model)
                print(f"[ASR] 段{idx+1}/{total} 已提交，task_id: {task_id}")

                result_url = self._wait_for_result(task_id)
                return self._fetch_result(result_url)

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = attempt * 5
                    print(f"[ASR] 段{idx+1} 第{attempt}次失败: {e}，{wait}s后重试...")
                    time.sleep(wait)

        raise RuntimeError(f"段{idx+1} 重试{max_retries}次仍失败: {last_error}")

    def _submit_task_with_retry(
        self, audio_b64: str, audio_format: str, model: str, max_retries: int = 3
    ) -> str:
        """提交转写任务，遇到 SSL/网络错误自动重试。"""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return self._submit_task(audio_b64, audio_format, model)
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = attempt * 3
                    print(f"[ASR] SSL/网络错误，{wait}s 后重试 ({attempt}/{max_retries}): {e}")
                    time.sleep(wait)
                else:
                    print(f"[ASR] SSL/网络错误，已重试 {max_retries} 次")
        raise last_error

    def _submit_task(self, audio_b64: str, audio_format: str, model: str) -> str:
        """提交异步转写任务。"""
        payload = {
            "model": model,
            "input": {
                "file_urls": [
                    f"data:audio/{audio_format};base64,{audio_b64}"
                ]
            },
            "parameters": {
                "channel_id": [0],
                "language_hints": ["zh", "en"],
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

        resp = self._session.post(
            SUBMIT_URL, headers=headers, json=payload, timeout=120
        )

        if resp.status_code == 400:
            error_data = resp.json()
            error_code = error_data.get("code", "")
            if "Model" in error_code or "model" in error_code.lower():
                raise ValueError(f"模型不可用: {error_data.get('message', '')}")
            elif "url" in error_data.get("message", "").lower():
                raise ValueError(f"URL 格式不支持: {error_data.get('message', '')}")

        resp.raise_for_status()

        data = resp.json()
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"提交任务失败: {data}")

        return task_id

    def _wait_for_result(self, task_id: str, max_wait: int = 600) -> str:
        """轮询等待转写任务完成。"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = QUERY_URL.format(task_id=task_id)
        start = time.time()

        while time.time() - start < max_wait:
            resp = self._session.post(url, headers=headers, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            status = data.get("output", {}).get("task_status", "")

            if status == "SUCCEEDED":
                results = data.get("output", {}).get("results", [])
                if isinstance(results, list) and len(results) > 0:
                    return results[0].get("transcription_url", "")
                elif isinstance(results, dict):
                    return results.get("transcription_url", "")
                return ""
            elif status in ("PENDING", "RUNNING"):
                elapsed = int(time.time() - start)
                print(f"[ASR] 转写中... ({elapsed}s)")
                time.sleep(10)
            elif status == "FAILED":
                raise RuntimeError(f"转写任务失败: {data}")
            else:
                time.sleep(10)

        raise TimeoutError("ASR 转写超时")

    def _fetch_result(self, result_url: str) -> str:
        """从结果 URL 获取转写文本。"""
        if not result_url:
            return ""

        resp = self._session.get(result_url, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        return self._parse_result(data)

    def _parse_result(self, data: dict) -> str:
        """解析 DashScope 返回的转写结果。"""
        lines = []
        transcripts = data.get("transcripts", [])

        for transcript in transcripts:
            sentences = transcript.get("sentences", [])
            for sent in sentences:
                begin_time = sent.get("begin_time", 0)
                mm, ss = divmod(int(begin_time // 1000), 60)
                text = sent.get("text", "").strip()
                if text:
                    lines.append(f"[{mm:02d}:{ss:02d}] {text}")

        return "\n".join(lines)
