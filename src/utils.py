"""通用工具函数"""

import os
from datetime import datetime
from pathlib import Path

import yaml


def load_env(env_path: str = ".env") -> None:
    """加载 .env 文件到环境变量。不会覆盖已存在的环境变量。"""
    search_paths = [
        env_path,
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        # 不覆盖已存在的环境变量
                        if key and key not in os.environ:
                            os.environ[key] = value
            return


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件，并用环境变量覆盖敏感配置。"""
    # 先加载 .env
    load_env()

    # 依次查找配置文件
    search_paths = [
        config_path,
        "config.yaml",
        os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            # 用环境变量覆盖
            _apply_env_overrides(config)
            return config

    raise FileNotFoundError(
        "找不到 config.yaml 配置文件，请参考 config.yaml 模板创建"
    )


def _apply_env_overrides(config: dict) -> None:
    """用环境变量覆盖配置中的敏感字段。"""

    # ASR: DashScope
    asr_cfg = config.setdefault("asr", {})
    qwen_cfg = asr_cfg.setdefault("qwen", {})
    _env_to_cfg("DASHSCOPE_API_KEY", qwen_cfg, "api_key")
    _env_to_cfg("DASHSCOPE_BASE_URL", qwen_cfg, "base_url")
    _env_to_cfg("DASHSCOPE_ASR_MODEL", qwen_cfg, "model")

    # LLM
    llm_cfg = config.setdefault("llm", {})
    _env_to_cfg("LLM_BASE_URL", llm_cfg, "base_url")
    _env_to_cfg("LLM_API_KEY", llm_cfg, "api_key")
    _env_to_cfg("LLM_MODEL", llm_cfg, "model")

    # QQ Bot
    qq_cfg = config.get("qq") or {}
    config["qq"] = qq_cfg
    _env_to_cfg("QQ_BOT_APPID", qq_cfg, "appid")
    _env_to_cfg("QQ_BOT_SECRET", qq_cfg, "secret")


def _env_to_cfg(env_key: str, cfg: dict, cfg_key: str) -> None:
    """将环境变量写入配置字典（环境变量优先）。"""
    value = os.environ.get(env_key, "")
    if value:
        cfg[cfg_key] = value


def create_output_dir(base_dir: str = "output", name: str = "") -> str:
    """创建输出目录。如果指定 name 则用 name，否则用当前时间戳。"""
    dir_name = name if name else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, dir_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def save_summary(summary: str, output_dir: str, filename: str = "summary.md"):
    """保存总结到文件。"""
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    return path
