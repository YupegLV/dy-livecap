"""抖音直播间弹幕抓取

通过 WebSocket + Protobuf 解析抖音直播间弹幕。
"""

import gzip
import json
import struct
import threading
import time
from pathlib import Path

import requests
import websocket


# 抖音弹幕 WebSocket URL 模板
WSS_URL = "wss://webcast5-ws-web-lf.douyin.com/webcast/im/push/v2/"

# 弹幕消息类型
MSG_CHAT = "WebcastChatMessage"          # 普通弹幕
MSG_GIFT = "WebcastGiftMessage"          # 礼物
MSG_LIKE = "WebcastLikeMessage"          # 点赞
MSG_MEMBER = "WebcastMemberMessage"      # 进入直播间
MSG_SOCIAL = "WebcastSocialMessage"      # 关注


class DanmakuCollector:
    """抖音直播间弹幕收集器。"""

    def __init__(self, room_id: str, output_dir: str, cookie: str = ""):
        self.room_id = room_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cookie = cookie
        self._ws = None
        self._running = False
        self._danmaku_list: list[dict] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0
        self._reconnect_attempts: int = 0

    def start(self):
        """启动弹幕收集（在后台线程中运行）。"""
        self._running = True
        self._reconnect_attempts = 0
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._connect, daemon=True)
        self._thread.start()
        print("[弹幕] 弹幕收集已启动")

    def stop(self) -> str:
        """停止弹幕收集并保存到文件。

        Returns:
            弹幕文件路径
        """
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)

        # 保存弹幕到文件
        output_path = self.output_dir / "danmaku.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for msg in self._danmaku_list:
                elapsed = msg.get("time_offset", 0)
                mm, ss = divmod(int(elapsed), 60)
                line = f"[{mm:02d}:{ss:02d}] {msg.get('type', '')} | {msg.get('nickname', '')}: {msg.get('content', '')}"
                f.write(line + "\n")

        print(f"[弹幕] 弹幕已保存至: {output_path} (共 {len(self._danmaku_list)} 条)")
        return str(output_path)

    @property
    def danmaku_text(self) -> str:
        """获取当前已收集的弹幕文本。"""
        with self._lock:
            lines = []
            for msg in self._danmaku_list:
                elapsed = msg.get("time_offset", 0)
                mm, ss = divmod(int(elapsed), 60)
                line = f"[{mm:02d}:{ss:02d}] {msg.get('nickname', '')}: {msg.get('content', '')}"
                lines.append(line)
            return "\n".join(lines)

    def _connect(self):
        """建立 WebSocket 连接。"""
        try:
            # 获取连接参数
            params = self._get_connect_params()
            if not params:
                print("[弹幕] 无法获取弹幕连接参数，跳过弹幕收集")
                return

            url = WSS_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
            self._ws = websocket.WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                header={"Cookie": self.cookie} if self.cookie else {},
            )
            self._ws.on_open = self._on_open
            self._ws.run_forever(ping_interval=10)

        except Exception as e:
            print(f"[弹幕] 连接异常: {e}")

    def _get_connect_params(self) -> dict:
        """从直播间页面获取 WebSocket 连接参数。"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"https://live.douyin.com/{self.room_id}",
        }
        if self.cookie:
            headers["Cookie"] = self.cookie

        try:
            resp = requests.get(
                f"https://live.douyin.com/{self.room_id}",
                headers=headers,
                timeout=10,
            )
            html = resp.text

            import re
            params = {}

            # 提取 ttwid
            ttwid_match = re.search(r'ttwid=([^;]+)', resp.headers.get("Set-Cookie", ""))
            if ttwid_match:
                params["ttwid"] = ttwid_match.group(1)

            # 从页面中提取 user_unique_id
            uid_match = re.search(r'"user_unique_id"\s*:\s*"(\d+)"', html)
            if uid_match:
                params["user_unique_id"] = uid_match.group(1)

            # room_id
            params["room_id"] = self.room_id
            params["compress"] = "gzip"

            return params if len(params) >= 2 else {}

        except Exception as e:
            print(f"[弹幕] 获取连接参数失败: {e}")
            return {}

    def _on_open(self, ws):
        print("[弹幕] WebSocket 连接成功")

    def _on_message(self, ws, message: bytes):
        """处理收到的消息。"""
        try:
            # 抖音弹幕数据是 gzip 压缩的 protobuf
            # 简单解析：先尝试 gzip 解压，然后按文本搜索关键字段
            try:
                data = gzip.decompress(message)
            except Exception:
                data = message

            # 尝试从二进制数据中提取文本消息
            # 这是一个简化的解析，实际应该用 protobuf 定义解析
            text = data.decode("utf-8", errors="ignore")

            # 提取弹幕内容（简化版，基于文本搜索）
            self._parse_simple(text)

        except Exception:
            pass

    def _parse_simple(self, text: str):
        """简化的弹幕解析（基于文本搜索）。

        注意：这是 fallback 方案，最佳实践是用 protobuf 定义解析。
        """
        import re

        elapsed = time.time() - self._start_time

        # 尝试提取 ChatMessage
        chat_matches = re.findall(
            r'"content"\s*:\s*"([^"]+)"', text
        )
        for content in chat_matches:
            if len(content) > 100:  # 过滤掉非弹幕的长文本
                continue
            msg = {
                "type": MSG_CHAT,
                "nickname": "",
                "content": content,
                "time_offset": elapsed,
            }
            with self._lock:
                self._danmaku_list.append(msg)

    def _on_error(self, ws, error):
        print(f"[弹幕] WebSocket 错误: {error}")

    def _on_close(self, ws, close_status, close_msg):
        if self._running:
            self._reconnect_attempts += 1
            if self._reconnect_attempts >= 3:
                print("[弹幕] 连续断开3次，停止重连")
                return
            print("[弹幕] WebSocket 断开，尝试重连...")
            time.sleep(3)
            if self._running:
                self._connect()
