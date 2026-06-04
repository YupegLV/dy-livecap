"""QQ Bot 核心

基于 qq-botpy SDK 的 QQ 机器人封装。
支持 C2C 私聊消息收发和图片上传。
"""

import asyncio
import threading
from typing import Callable

import botpy
from botpy import logging as botpy_logging
from botpy.message import C2CMessage
from botpy.types.message import MarkdownPayload

_log = botpy_logging.get_logger()


class LiveMonitorBot(botpy.Client):
    """直播监控 QQ 机器人。"""

    # 类级别属性，由 bot_main 在启动前设置
    _message_handler: Callable = None
    _api_ref = None  # botpy API 引用，用于主动发消息
    _event_loop = None  # botpy 运行的事件循环

    async def on_ready(self):
        _log.info(f"机器人 [{self.robot.name}] 已上线！")
        LiveMonitorBot._event_loop = asyncio.get_running_loop()

    async def on_c2c_message_create(self, message: C2CMessage):
        """收到 C2C 私聊消息。"""
        print(f"[QQ Bot] 收到 C2C 消息! id={message.id}")

        try:
            user_openid = message.author.user_openid
        except Exception as e:
            print(f"[QQ Bot] 获取 openid 失败: {e}, author={message.author}")
            return

        content = message.content or ""
        msg_id = message.id

        _log.info(f"收到私聊消息: openid={user_openid}, content={content[:50]}")

        # 保存 API 引用供主动发消息用
        LiveMonitorBot._api_ref = message._api

        # 调用外部消息处理器
        if LiveMonitorBot._message_handler:
            try:
                await LiveMonitorBot._message_handler(
                    content=content,
                    user_openid=user_openid,
                    msg_id=msg_id,
                    api=message._api,
                )
            except Exception as e:
                _log.error(f"消息处理失败: {e}")
                import traceback
                traceback.print_exc()


class QQBotManager:
    """QQ Bot 管理器，封装启动、消息收发。"""

    def __init__(self, appid: str, secret: str):
        self.appid = appid
        self.secret = secret
        self._bot: LiveMonitorBot | None = None
        self._thread: threading.Thread | None = None
        self._msg_seq: dict[str, int] = {}  # user_openid -> msg_seq

    def set_message_handler(self, handler: Callable):
        """设置消息处理回调。

        handler 签名: async def handler(content: str, user_openid: str,
                                         msg_id: str, api) -> None
        """
        LiveMonitorBot._message_handler = handler

    def start(self):
        """在后台线程中启动 QQ Bot。"""
        intents = botpy.Intents(public_messages=True)
        self._bot = LiveMonitorBot(intents=intents)

        def _run():
            asyncio.run(self._bot.run(appid=self.appid, secret=self.secret))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        print("[QQ Bot] 已启动，等待消息...")

    async def send_text(self, user_openid: str, text: str, msg_id: str = "",
                        use_markdown: bool = True):
        """发送消息，默认使用 Markdown 渲染，失败回退纯文本。"""
        api = LiveMonitorBot._api_ref
        if not api:
            print("[QQ Bot] API 未就绪，无法发送消息")
            return

        seq = self._next_seq(user_openid)

        if use_markdown:
            try:
                markdown = MarkdownPayload(content=text)
                await api.post_c2c_message(
                    openid=user_openid,
                    msg_type=2,
                    msg_id=msg_id,
                    msg_seq=seq,
                    markdown=markdown,
                )
                print(f"[QQ Bot] Markdown 消息发送成功 (seq={seq})")
                return
            except Exception as e:
                print(f"[QQ Bot] Markdown 发送失败: {e}，回退纯文本")

        # 纯文本回退
        try:
            await api.post_c2c_message(
                openid=user_openid,
                msg_type=0,
                msg_id=msg_id,
                msg_seq=seq,
                content=text,
            )
            print(f"[QQ Bot] 文本发送成功 (seq={seq})")
        except Exception as e:
            print(f"[QQ Bot] 发送文本失败: {e}")

    def send_text_sync(self, user_openid: str, text: str,
                        use_markdown: bool = True):
        """从非 asyncio 线程发送消息（线程安全）。"""
        loop = LiveMonitorBot._event_loop
        if not loop:
            print("[QQ Bot] 事件循环未就绪")
            return
        asyncio.run_coroutine_threadsafe(
            self.send_text(user_openid, text, use_markdown=use_markdown), loop
        )

    def send_result_sync(self, user_openid: str, summary_text: str,
                          image_paths: list[str] = []):
        """从非 asyncio 线程发送总结（线程安全，默认 Markdown）。"""
        loop = LiveMonitorBot._event_loop
        if not loop:
            print("[QQ Bot] 事件循环未就绪")
            return
        asyncio.run_coroutine_threadsafe(
            self.send_text(user_openid, summary_text, use_markdown=True), loop
        )

    def _next_seq(self, user_openid: str) -> int:
        """生成递增的 msg_seq。"""
        seq = self._msg_seq.get(user_openid, 0) + 1
        self._msg_seq[user_openid] = seq
        return seq
