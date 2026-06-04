"""LLM 意图解析 Agent

用 LLM 解析用户消息，提取直播录制参数。
"""

import json
import re

from openai import OpenAI


INTENT_SYSTEM_PROMPT = """你是一个直播监控助手的意图解析器。你的任务是分析用户消息，提取出执行参数。

用户可能的表达方式：
- 直接发抖音直播分享链接（包含 v.douyin.com 域名）
- "帮我录这个直播1小时 https://v.douyin.com/xxx"
- "看看这个直播在干嘛 https://v.douyin.com/xxx"
- "监控这个直播间半小时"
- "录一下这个"
- 发送包含抖音链接的分享文本
- "恢复任务ID 20260602_144803" / "重试 20260602_144803" / "恢复 20260602_144803"
- 对之前直播内容提问，如"刚才直播讲了什么技术"、"价格是多少"、"再详细说说第三段"
- 要求重新总结或换个角度总结，如"换个角度总结"、"只看弹幕部分"

请分析用户消息，返回 JSON 格式的结果：
{
  "action": "record" | "recover" | "query" | "chat" | "help" | "unknown",
  "url": "链接地址（仅 v.douyin.com 或 douyin.com 链接）",
  "duration": 分钟数（整数，如果用户没指定则根据上下文推断）,
  "task_id": "任务ID（格式为 YYYYMMDD_HHMMSS 的时间戳，仅 recover 时需要）",
  "user_request": "用户对总结内容的特殊要求（如无则为空字符串）"
}

规则：
1. 如果消息包含 v.douyin.com 或 douyin.com 链接，action 为 "record"
2. 如果用户说了时长（如"1小时"、"30分钟"、"10分钟"），按用户说的来
3. 如果用户没说时长：
   - "看看"/"看看在干嘛" → duration=10（快速查看）
   - 其他情况 → duration=60（默认1小时）
4. 如果用户想恢复/重试失败的任务，action 为 "recover"，从消息中提取 task_id（时间戳格式）
5. 如果用户在追问、讨论、提问关于之前直播的内容，或者要求重新/换个角度总结，action 为 "chat"
6. 如果没有链接但问关于用法的问题，action 为 "help"
7. 如果完全无法理解，action 为 "unknown"
8. url 字段只保留纯链接，去掉分享文本中的其他内容
9. user_request 提取用户对总结的特殊要求，例如：
   - "重点看讲了什么技术" → user_request: "重点总结技术相关内容"
   - "只要时间线和重点" → user_request: "只输出时间线和重点信息"
   - "关注价格和促销信息" → user_request: "关注价格、促销、优惠信息"
   - 没有特殊要求时 user_request 为空字符串

只输出 JSON，不要输出其他内容。"""

CHAT_SYSTEM_PROMPT = """你是一个专业的直播内容分析师助手。用户已经观看了一次直播，你之前为他生成了总结。现在用户针对这次直播内容提出了新的问题或要求，请根据直播的完整记录和之前的总结来回答。

要求：
1. 基于直播的实际内容回答，不要编造信息
2. 如果用户要求换个角度总结或聚焦某方面，重新组织信息
3. 回答要简洁清晰，直接回应用户的问题
4. 如果直播记录中没有相关信息，如实告知"""


class LLMAgent:
    """LLM 意图解析器。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def parse_intent(self, message: str) -> dict:
        """解析用户消息，返回 action + 参数。

        Returns:
            {
                "action": "record" | "recover" | "query" | "help" | "unknown",
                "url": str | None,
                "duration": int | None,  # 分钟
                "task_id": str | None,
            }
        """
        # 快速路径：不用 LLM，直接正则提取链接
        url = self._extract_url(message)
        if url:
            duration = self._extract_duration(message)
            if duration is None:
                # 根据消息语气推断
                if any(kw in message for kw in ["看看", "看看在干嘛", "瞄一眼", "看下"]):
                    duration = 10
                else:
                    duration = 60
            return {"action": "record", "url": url, "duration": duration}

        # 没有链接，判断是否在追问直播内容
        chat_keywords = [
            "讲了什么", "提到了", "说了什么", "详细说说", "再说一下", "展开讲讲",
            "换个角度", "重新总结", "只看", "重点关注", "价格", "优惠",
            "弹幕", "互动", "观众", "主播说了", "那段", "第几", "什么时间",
            "讲了多久", "有没有提到", "关键帧", "截图", "画面",
        ]
        if any(kw in message for kw in chat_keywords):
            return {"action": "chat", "url": None, "duration": None,
                    "user_request": message}

        # 没有链接，判断是否求助
        if any(kw in message for kw in ["帮助", "怎么用", "使用方法", "help", "帮助我"]):
            return {"action": "help", "url": None, "duration": None}

        # 有链接但正则没提取到，或者复杂表达，用 LLM 解析
        if any(kw in message.lower() for kw in [
            "douyin", "抖音", "直播", "录", "恢复", "重试", "retry"
        ]):
            return self._llm_parse(message)

        # 短消息且有已完成任务时，可能是追问，交给 LLM 判断
        if len(message) < 100:
            return self._llm_parse(message)

        return {"action": "unknown", "url": None, "duration": None}

    def chat_with_context(
        self,
        user_message: str,
        asr_text: str = "",
        previous_summary: str = "",
        chat_history: list[dict] | None = None,
        anchor_name: str = "",
        duration: int = 0,
    ) -> str:
        """基于直播上下文的多轮对话。

        Args:
            user_message: 用户的新问题
            asr_text: 直播 ASR 转写文本
            previous_summary: 之前的总结内容
            chat_history: 之前的对话历史 [{"role": "user"/"assistant", "content": "..."}]
            anchor_name: 主播名
            duration: 直播时长（分钟）

        Returns:
            LLM 的回答
        """
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

        # 构建上下文：ASR 文本截取（避免过长）
        context_parts = []
        if anchor_name:
            context_parts.append(f"主播：{anchor_name}")
        if duration:
            context_parts.append(f"直播时长：{duration} 分钟")
        if previous_summary:
            context_parts.append(f"## 之前的总结\n{previous_summary}")
        if asr_text:
            # ASR 文本可能很长，截取关键部分
            max_asr_len = 15000
            if len(asr_text) > max_asr_len:
                truncated = asr_text[:max_asr_len] + f"\n\n[...已截取，原文共 {len(asr_text)} 字符]"
                context_parts.append(f"## 直播语音转写（截取）\n{truncated}")
            else:
                context_parts.append(f"## 直播语音转写\n{asr_text}")

        context = "\n\n".join(context_parts)

        # 第一轮：注入上下文
        if not chat_history:
            messages.append({
                "role": "user",
                "content": f"以下是直播内容记录：\n\n{context}\n\n请记住这些内容，我接下来会提问。",
            })
            messages.append({
                "role": "assistant",
                "content": "好的，我已经了解了这次直播的内容，请随时提问。",
            })
        else:
            # 有历史对话，把上下文作为 system 的一部分注入
            messages[0]["content"] += f"\n\n{context}"
            # 添加历史对话（最近 10 轮）
            for msg in chat_history[-20:]:
                messages.append(msg)

        # 添加用户当前问题
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content
            return content.strip() if content else "抱歉，无法生成回答。"
        except Exception as e:
            print(f"[LLM Agent] 多轮对话失败: {e}")
            return f"生成回答失败: {e}"

    def _extract_url(self, message: str) -> str | None:
        """从消息中提取抖音链接。"""
        # 匹配 v.douyin.com/xxx 格式
        patterns = [
            r'https?://v\.douyin\.com/[A-Za-z0-9_/-]+',
            r'https?://www\.douyin\.com/[A-Za-z0-9_/-]+',
            r'https?://www\.iesdouyin\.com/[A-Za-z0-9_/-]+',
        ]
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                url = match.group(0)
                # 去掉末尾可能附带的标点
                url = url.rstrip("，。、！？,.!?")
                return url
        return None

    def _extract_duration(self, message: str) -> int | None:
        """从消息中提取时长（分钟）。"""
        # "X小时"
        match = re.search(r'(\d+)\s*个?小时', message)
        if match:
            return int(match.group(1)) * 60

        # "X半小时" = X*60+30
        match = re.search(r'(\d+)\s*个?半小时', message)
        if match:
            return int(match.group(1)) * 60 + 30

        # "半小时" = 30
        if "半小时" in message:
            return 30

        # "X分钟"
        match = re.search(r'(\d+)\s*分钟', message)
        if match:
            return int(match.group(1))

        # "Xmin"
        match = re.search(r'(\d+)\s*min', message, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def _llm_parse(self, message: str) -> dict:
        """用 LLM 解析复杂消息。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                max_tokens=1024,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content
            text = content.strip() if content else ""
            if not text:
                # 思考模式可能把内容放在 reasoning 字段
                reasoning = getattr(response.choices[0].message, "reasoning", None)
                if reasoning:
                    json_match = re.search(r'\{[^}]+\}', reasoning, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                        if result.get("action") not in ("record", "recover", "query", "help", "unknown"):
                            result["action"] = "unknown"
                        return result
                print(f"[LLM Agent] 返回为空")
                return {"action": "unknown", "url": None, "duration": None}
            # 提取 JSON
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 验证 action
                if result.get("action") not in ("record", "recover", "query", "help", "unknown"):
                    result["action"] = "unknown"
                return result
        except Exception as e:
            print(f"[LLM Agent] 解析失败: {e}")

        return {"action": "unknown", "url": None, "duration": None}


# 帮助信息
HELP_TEXT = """直播监控助手 使用说明：

发送抖音直播分享链接即可开始录制和总结。

示例：
- 直接发送直播分享链接 → 默认录制1小时
- "帮我录这个直播30分钟 [链接]"
- "看看这个直播在干嘛 [链接]" → 快速查看10分钟
- "录1小时 [链接]"
- "录这个直播，重点看讲了什么技术 [链接]" → 按需求总结

恢复失败任务：
- "恢复 任务ID" 或 "重试 任务ID" → 从 ASR 步骤重新执行
- 任务ID为时间戳格式，如 20260602_144803

录制完成后会自动发送总结。"""
