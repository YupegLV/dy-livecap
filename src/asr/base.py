"""ASR 抽象基类"""

from abc import ABC, abstractmethod


class ASRBase(ABC):
    """语音识别抽象接口。"""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """将音频文件转写为文本。

        Args:
            audio_path: 音频文件路径（WAV 16kHz 单声道）

        Returns:
            转写文本（带时间戳）
        """
        ...
