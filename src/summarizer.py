"""LLM 总结模块

使用 OpenAI 兼容 API 对直播内容进行总结。
支持超长文本的 Map-Reduce 分段总结。
支持多模态（关键帧图片 + 文本）总结。
"""

import base64
import os
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """你是一个专业的直播内容分析师。你的任务是分析直播的文字记录、弹幕和关键帧截图，生成结构化的直播总结。

请按以下格式输出总结：

## 直播内容总结

### 基本信息
- 主播：
- 直播时长：
- 主要话题：

### 内容时间线
按时间段列出直播的关键内容，格式：[时间段] 内容描述

### 关键信息点
列出直播中提到的关键信息、重要观点、值得注意的内容

### 互动亮点
（如有弹幕数据）列出观众互动的亮点、热门话题、主播回应的问题

### 总结
用2-3句话概括本次直播的核心内容
"""

SEGMENT_PROMPT = """请总结以下这一段直播内容的要点。用简洁的语言描述这一段直播讨论了什么、有哪些关键信息。

直播内容：
{content}

请输出要点总结："""

SEGMENT_PROMPT_WITH_IMAGES = """请结合以下关键帧截图和文字记录，总结这一段直播内容的要点。描述画面中展示了什么内容，以及主播在说什么。

文字记录：
{content}

请输出要点总结（包含画面内容和语音内容）："""

MERGE_PROMPT = """以下是按时间顺序排列的直播分段总结。请将它们整合为一份完整的直播内容总结。

{segments}

请按照直播内容总结的格式输出："""


def _encode_image(image_path: str) -> str:
    """将图片编码为 base64。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_media_type(image_path: str) -> str:
    """根据文件扩展名返回 MIME 类型。"""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


class LiveSummarizer:
    """直播内容 LLM 总结器。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "qwen3.5-397b-a17b",
        max_tokens: int = 8192,
        multimodal: bool = False,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.multimodal = multimodal

    def summarize(
        self,
        asr_text: str,
        danmaku_text: str = "",
        anchor_name: str = "",
        duration: int = 0,
        keyframes: list[str] | None = None,
        user_request: str = "",
    ) -> str:
        """对直播内容进行总结。

        Args:
            asr_text: ASR 转写文本
            danmaku_text: 弹幕文本（可选）
            anchor_name: 主播名
            duration: 直播时长（分钟）
            keyframes: 关键帧图片路径列表（多模态模式）
            user_request: 用户的自定义需求（可选）

        Returns:
            LLM 生成的总结文本
        """
        # 构建完整内容
        content_parts = []
        if asr_text:
            content_parts.append(f"## 语音转写内容\n{asr_text}")
        if danmaku_text:
            content_parts.append(f"## 弹幕内容\n{danmaku_text}")

        full_content = "\n\n".join(content_parts)

        # 多模态模式：有关键帧图片
        if self.multimodal and keyframes:
            return self._multimodal_summarize(
                full_content, keyframes, anchor_name, duration, user_request
            )

        # 纯文本模式
        estimated_tokens = len(full_content) / 1.5
        if estimated_tokens > 20000:
            print(f"[LLM] 文本较长（约 {int(estimated_tokens)} tokens），使用分段总结...")
            return self._map_reduce_summarize(full_content, anchor_name, duration, user_request)
        else:
            print(f"[LLM] 直接总结（约 {int(estimated_tokens)} tokens）...")
            return self._direct_summarize(full_content, anchor_name, duration, user_request)

    def _direct_summarize(
        self, content: str, anchor_name: str = "", duration: int = 0,
        user_request: str = ""
    ) -> str:
        """直接总结完整内容。"""
        request_hint = f"\n\n用户特别要求：{user_request}" if user_request else ""
        user_msg = f"请总结以下直播内容。\n\n主播：{anchor_name}\n直播时长：{duration} 分钟{request_hint}\n\n{content}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content

    def _multimodal_summarize(
        self,
        content: str,
        keyframes: list[str],
        anchor_name: str = "",
        duration: int = 0,
        user_request: str = "",
    ) -> str:
        """多模态总结：关键帧图片 + 文本。

        将关键帧图片按时间段分组，结合对应文本分段总结，最后合并。
        """
        print(f"[LLM] 多模态总结：{len(keyframes)} 张关键帧 + 文本")

        # 按关键帧数量分段（每组约5张图片+对应文本）
        group_size = 5
        keyframe_groups = [
            keyframes[i : i + group_size]
            for i in range(0, len(keyframes), group_size)
        ]

        # 同时将文本分段
        text_segments = self._split_text(content, len(keyframe_groups))

        # Map: 每组图片+文本 → 段小结
        segment_summaries = []
        for i, (kf_group, text_seg) in enumerate(
            zip(keyframe_groups, text_segments)
        ):
            print(
                f"[LLM] 多模态分段 {i + 1}/{len(keyframe_groups)}: "
                f"{len(kf_group)} 张关键帧"
            )
            summary = self._summarize_segment_with_images(text_seg, kf_group)
            segment_summaries.append(f"### 第 {i + 1} 段\n{summary}")

        # 如果文本段比图片组多，处理剩余文本
        if len(text_segments) > len(keyframe_groups):
            for i in range(len(keyframe_groups), len(text_segments)):
                print(f"[LLM] 文本分段 {i + 1}/{len(text_segments)}")
                prompt = SEGMENT_PROMPT.format(content=text_segments[i])
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.3,
                )
                segment_summaries.append(
                    f"### 第 {i + 1} 段\n{response.choices[0].message.content}"
                )

        # Reduce: 合并
        print("[LLM] 正在合并分段总结...")
        all_segments = "\n\n".join(segment_summaries)
        merge_msg = MERGE_PROMPT.format(segments=all_segments)
        request_hint = f"\n\n用户特别要求：{user_request}" if user_request else ""
        user_msg = (
            f"请整合以下分段总结为完整的直播内容总结。\n\n"
            f"主播：{anchor_name}\n直播时长：{duration} 分钟{request_hint}\n\n{merge_msg}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content

    def _summarize_segment_with_images(
        self, text: str, image_paths: list[str]
    ) -> str:
        """用多模态 LLM 总结一段图文内容。"""
        # 构建 OpenAI 多模态消息格式
        content_parts = []

        # 添加文本
        prompt = SEGMENT_PROMPT_WITH_IMAGES.format(content=text)
        content_parts.append({"type": "text", "text": prompt})

        # 添加图片
        for img_path in image_paths:
            if not Path(img_path).exists():
                continue
            try:
                b64 = _encode_image(img_path)
                media_type = _get_image_media_type(img_path)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}",
                        },
                    }
                )
            except Exception as e:
                print(f"[LLM] 图片加载失败 {img_path}: {e}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content_parts}],
            max_tokens=2048,
            temperature=0.3,
        )

        return response.choices[0].message.content

    def _map_reduce_summarize(
        self, content: str, anchor_name: str = "", duration: int = 0,
        user_request: str = ""
    ) -> str:
        """Map-Reduce 分段总结（纯文本）。"""
        lines = content.split("\n")
        segment_size = 5000
        segments = []

        current_segment = []
        current_len = 0
        for line in lines:
            current_segment.append(line)
            current_len += len(line)
            if current_len >= segment_size:
                segments.append("\n".join(current_segment))
                current_segment = []
                current_len = 0

        if current_segment:
            segments.append("\n".join(current_segment))

        print(f"[LLM] 分为 {len(segments)} 段进行总结...")

        # Map
        segment_summaries = []
        for i, seg in enumerate(segments):
            print(f"[LLM] 正在总结第 {i + 1}/{len(segments)} 段...")
            prompt = SEGMENT_PROMPT.format(content=seg)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )

            summary = response.choices[0].message.content
            segment_summaries.append(f"### 第 {i + 1} 段\n{summary}")

        # Reduce
        print("[LLM] 正在合并分段总结...")
        all_segments = "\n\n".join(segment_summaries)
        merge_msg = MERGE_PROMPT.format(segments=all_segments)
        request_hint = f"\n\n用户特别要求：{user_request}" if user_request else ""
        user_msg = (
            f"请整合以下分段总结为完整的直播内容总结。\n\n"
            f"主播：{anchor_name}\n直播时长：{duration} 分钟{request_hint}\n\n{merge_msg}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content

    @staticmethod
    def _split_text(text: str, num_segments: int) -> list[str]:
        """将文本均匀分成指定段数。"""
        if not text or num_segments <= 0:
            return [text] if text else []

        lines = text.split("\n")
        total_len = len(lines)
        seg_len = max(total_len // num_segments, 1)

        segments = []
        for i in range(0, total_len, seg_len):
            segments.append("\n".join(lines[i : i + seg_len]))

        return segments
