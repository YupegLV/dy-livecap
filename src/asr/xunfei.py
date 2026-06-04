"""讯飞语音转写 ASR 实现

讯飞语音转写（LFASR）API 流程：
1. prepare - 预处理，获取 task_id
2. upload  - 分片上传音频文件
3. merge   - 合并音频分片
4. getResult - 获取转写结果

文档：https://www.xunfei.cn/doc/asr/lfasr/API.html
"""

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests

from .base import ASRBase


# 讯飞语音转写 API 地址
BASE_URL = "https://raasr.xfyun.cn/v2/api"


class XunfeiASR(ASRBase):
    """讯飞语音转写实现。"""

    def __init__(self, app_id: str, api_key: str, api_secret: str):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret

    def transcribe(self, audio_path: str) -> str:
        """使用讯飞语音转写 API 将音频转为文本。"""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        file_size = Path(audio_path).stat().st_size
        print(f"[ASR] 开始讯飞语音转写，文件大小: {file_size / 1024 / 1024:.1f} MB")

        # Step 1: prepare
        task_id = self._prepare(audio_path, file_size)
        print(f"[ASR] 预处理完成，task_id: {task_id}")

        # Step 2: upload
        self._upload(audio_path, task_id)
        print("[ASR] 音频上传完成")

        # Step 3: merge
        self._merge(task_id)
        print("[ASR] 音频合并完成，等待转写...")

        # Step 4: getResult (轮询)
        result = self._get_result(task_id)
        print(f"[ASR] 转写完成，文本长度: {len(result)} 字")

        return result

    def _prepare(self, audio_path: str, file_size: int) -> str:
        """预处理，获取 task_id。"""
        url = f"{BASE_URL}/prepare"
        filename = Path(audio_path).name

        body = {
            "app_id": self.app_id,
            "file_name": filename,
            "file_size": file_size,
            "language": "cn",  # 中文
            "has_participle": "true",  # 开启分词
        }

        headers = self._build_headers("POST", urlparse(url).path)
        resp = requests.post(url, headers=headers, data=urlencode(body))
        resp.raise_for_status()

        data = resp.json()
        if data.get("code") != "000000":
            raise RuntimeError(f"讯飞 prepare 失败: {data}")

        return data["data"]

    def _upload(self, audio_path: str, task_id: str):
        """分片上传音频文件。"""
        url = f"{BASE_URL}/upload"
        slice_size = 10 * 1024 * 1024  # 10MB 每片

        with open(audio_path, "rb") as f:
            slice_id = 0
            while True:
                chunk = f.read(slice_size)
                if not chunk:
                    break

                body = {
                    "app_id": self.app_id,
                    "task_id": task_id,
                    "slice_id": str(slice_id),
                }

                files = {
                    "file": (f"slice_{slice_id}", chunk, "application/octet-stream"),
                }

                headers = self._build_headers("POST", urlparse(url).path)
                # upload 使用 multipart/form-data，不设置 Content-Type
                resp = requests.post(url, headers=headers, data=body, files=files)
                resp.raise_for_status()

                data = resp.json()
                if data.get("code") != "000000":
                    raise RuntimeError(f"讯飞 upload 失败: {data}")

                slice_id += 1
                print(f"[ASR] 已上传分片 {slice_id}")

    def _merge(self, task_id: str):
        """合并音频分片。"""
        url = f"{BASE_URL}/merge"

        body = {
            "app_id": self.app_id,
            "task_id": task_id,
        }

        headers = self._build_headers("POST", urlparse(url).path)
        resp = requests.post(url, headers=headers, data=urlencode(body))
        resp.raise_for_status()

        data = resp.json()
        if data.get("code") != "000000":
            raise RuntimeError(f"讯飞 merge 失败: {data}")

    def _get_result(self, task_id: str, max_wait: int = 3600) -> str:
        """轮询获取转写结果。"""
        url = f"{BASE_URL}/getResult"

        body = {
            "app_id": self.app_id,
            "task_id": task_id,
        }

        headers = self._build_headers("POST", urlparse(url).path)
        start = time.time()

        while time.time() - start < max_wait:
            resp = requests.post(url, headers=headers, data=urlencode(body))
            resp.raise_for_status()

            data = resp.json()
            code = data.get("code")

            if code == "000000":
                # 转写完成，解析结果
                return self._parse_result(data.get("data", ""))
            elif code == "200000":
                # 正在转写中
                progress = data.get("progress", 0)
                print(f"[ASR] 转写进度: {progress}%")
                time.sleep(10)
            else:
                raise RuntimeError(f"讯飞 getResult 失败: {data}")

        raise TimeoutError("讯飞转写超时")

    def _parse_result(self, data: str) -> str:
        """解析讯飞返回的转写结果。"""
        if not data:
            return ""

        try:
            result = json.loads(data)
        except json.JSONDecodeError:
            return data

        # 讯飞返回格式: {"lattice": [{"xml": "...", "json_1best": "..."}]}
        lines = []
        lattices = result.get("lattice", [])
        for lattice in lattices:
            json_best = lattice.get("json_1best", "")
            if json_best:
                try:
                    best = json.loads(json_best)
                    st = best.get("st", {})
                    bg = st.get("bg", "0")
                    ed = st.get("ed", "0")
                    # 将时间转为 mm:ss
                    bg_sec = int(float(bg)) // 1000
                    mm, ss = divmod(bg_sec, 60)
                    rt = st.get("rt", [])
                    text_parts = []
                    for r in rt:
                        for ws in r.get("ws", []):
                            for cw in ws.get("cw", []):
                                text_parts.append(cw.get("w", ""))
                    text = "".join(text_parts)
                    if text:
                        lines.append(f"[{mm:02d}:{ss:02d}] {text}")
                except (json.JSONDecodeError, KeyError):
                    pass

        return "\n".join(lines)

    def _build_headers(self, method: str, path: str) -> dict:
        """构建鉴权请求头。"""
        # 生成 RFC1123 格式时间戳
        now = datetime.utcnow()
        date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

        # 拼接签名原文
        signature_origin = f"host: raasr.xfyun.cn\ndate: {date}\n{method} {path} HTTP/1.1"
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"),
                signature_origin.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        authorization = (
            f"api_key=\"{self.api_key}\", "
            f"algorithm=\"hmac-sha256\", "
            f"headers=\"host date request-line\", "
            f"signature=\"{signature}\""
        )
        authorization = base64.b64encode(authorization.encode("utf-8")).decode("utf-8")

        return {
            "Content-Type": "application/x-www-form-urlencoded",
            "Date": date,
            "Authorization": authorization,
        }
