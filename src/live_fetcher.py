"""从抖音分享链接获取直播流地址"""

import re
import json
import requests
from urllib.parse import urlparse, unquote


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://live.douyin.com/",
    "Cookie": "",
}

MOBILE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
}


def resolve_share_url(url: str) -> str:
    """将短链接解析为长链接，提取 room_id。

    支持的输入格式：
    - 短链接: https://v.douyin.com/xxx/
    - 长链接: https://live.douyin.com/123456
    - 纯 room_id: 123456
    """
    # 纯数字视为 room_id
    if url.strip().isdigit():
        return url.strip()

    # 已经是长链接格式
    match = re.search(r"live\.douyin\.com/(\d+)", url)
    if match:
        return match.group(1)

    # 短链接需要重定向解析
    try:
        # 先用 HEAD 请求
        resp = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=10)
        final_url = resp.url

        # 从 live.douyin.com/xxx 提取
        match = re.search(r"live\.douyin\.com/(\d+)", final_url)
        if match:
            return match.group(1)

        # 从 webcast.amemv.com/douyin/webcast/reflow/xxx 提取
        match = re.search(r"webcast\.amemv\.com/douyin/webcast/reflow/(\d+)", final_url)
        if match:
            return match.group(1)

        # 如果 HEAD 没拿到，试试 GET
        resp = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=15)
        final_url = resp.url

        match = re.search(r"live\.douyin\.com/(\d+)", final_url)
        if match:
            return match.group(1)

        match = re.search(r"webcast\.amemv\.com/douyin/webcast/reflow/(\d+)", final_url)
        if match:
            return match.group(1)

        # 从 HTML 内容中提取 room_id
        html = resp.text
        match = re.search(r'"roomId"\s*:\s*"(\d+)"', html)
        if match:
            return match.group(1)

        match = re.search(r'"room_id"\s*:\s*"?(\d+)"?', html)
        if match:
            return match.group(1)

    except requests.RequestException:
        pass

    raise ValueError(f"无法从链接中提取 room_id: {url}")


def get_live_stream_url(room_id: str, cookie: str = "") -> dict:
    """获取直播流地址和直播间信息。

    优先使用移动端 API（无需 cookie），回退到 PC 端页面解析。

    返回:
        {
            "is_living": bool,
            "anchor_name": str,
            "room_title": str,
            "stream_url": str,  # m3u8 地址
            "flv_url": str,     # flv 地址
        }
    """
    # 优先：移动端 API（无需 cookie）
    result = _get_via_mobile_api(room_id)
    if result and result["is_living"] and result["stream_url"]:
        return result

    # 回退：PC 端页面解析
    result_pc = _get_via_pc_page(room_id, cookie)
    if result_pc and result_pc["is_living"] and result_pc["stream_url"]:
        return result_pc

    # 合并结果（可能一个有主播名，另一个有流地址）
    if result and result_pc:
        merged = result.copy()
        if not merged["anchor_name"] and result_pc["anchor_name"]:
            merged["anchor_name"] = result_pc["anchor_name"]
        if not merged["stream_url"] and result_pc["stream_url"]:
            merged["stream_url"] = result_pc["stream_url"]
        if not merged["flv_url"] and result_pc["flv_url"]:
            merged["flv_url"] = result_pc["flv_url"]
        merged["is_living"] = merged["is_living"] or result_pc["is_living"]
        return merged

    return result or result_pc or {
        "is_living": False,
        "anchor_name": "",
        "room_title": "",
        "stream_url": "",
        "flv_url": "",
    }


def _get_via_mobile_api(room_id: str) -> dict:
    """通过移动端 API 获取直播流（无需 cookie）。"""
    result = {
        "is_living": False,
        "anchor_name": "",
        "room_title": "",
        "stream_url": "",
        "flv_url": "",
    }

    try:
        api_url = (
            f"https://webcast.amemv.com/webcast/room/reflow/info/"
            f"?room_id={room_id}&live_id=1&app_id=1128"
        )
        resp = requests.get(api_url, headers=MOBILE_HEADERS, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        room = data.get("data", {}).get("room", {})
        if not room:
            return result

        # 状态: 2 = 正在直播
        status = room.get("status")
        if status == 2:
            result["is_living"] = True

        # 主播名
        owner = room.get("owner", {})
        result["anchor_name"] = owner.get("nickname", "")

        # 房间标题
        result["room_title"] = room.get("title", "")

        # 流地址
        stream_url = room.get("stream_url", {})
        if stream_url:
            # hls_pull_url 可能是 str 或 dict
            hls = stream_url.get("hls_pull_url", "")
            if isinstance(hls, dict):
                # 选最高清晰度
                result["stream_url"] = _pick_best_quality(hls)
            else:
                result["stream_url"] = hls

            # flv_pull_url 可能是 str 或 dict
            flv = stream_url.get("flv_pull_url", "")
            if isinstance(flv, dict):
                result["flv_url"] = _pick_best_quality(flv)
            else:
                result["flv_url"] = flv

            # 如果主地址为空，尝试从 live_core_sdk_data 提取
            if not result["stream_url"]:
                core = stream_url.get("live_core_sdk_data", {})
                pull_data = core.get("pull_data", {})
                options = pull_data.get("options", {})
                quality_list = options.get("quality", [])
                for q in quality_list:
                    url = q.get("pull_stream_url", "")
                    if url:
                        result["stream_url"] = url
                        break

    except Exception as e:
        print(f"[直播] 移动端 API 请求失败: {e}")

    return result


def _get_via_pc_page(room_id: str, cookie: str = "") -> dict:
    """通过 PC 端页面解析获取直播流（需要 cookie 才能获取完整数据）。"""
    result = {
        "is_living": False,
        "anchor_name": "",
        "room_title": "",
        "stream_url": "",
        "flv_url": "",
    }

    headers = HEADERS.copy()
    if cookie:
        headers["Cookie"] = cookie

    try:
        url = f"https://live.douyin.com/{room_id}"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        html = resp.text

        # 提取主播名
        anchor_match = re.search(r'"nickname"\s*:\s*"([^"]+)"', html)
        if anchor_match:
            result["anchor_name"] = anchor_match.group(1)

        # 提取房间标题
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', html)
        if title_match:
            result["room_title"] = title_match.group(1)

        # 从 RENDER_DATA 中提取
        render_match = re.search(r'RENDER_DATA[^>]*>(.*?)</script>', html)
        if render_match:
            try:
                render_data = json.loads(unquote(render_match.group(1)))
                for key, val in render_data.items():
                    if isinstance(val, dict):
                        room = val.get("room", {}) or val.get("roomInfo", {})
                        if isinstance(room, dict):
                            _extract_room_info(room, result)
            except (json.JSONDecodeError, KeyError):
                pass

        # 正则直接匹配流地址
        if not result["stream_url"]:
            m3u8_match = re.search(r'"hls_pull_url"\s*:\s*"(https?://[^"]+\.m3u8[^"]*)"', html)
            if m3u8_match:
                result["stream_url"] = m3u8_match.group(1).replace("\\u002F", "/")
                result["is_living"] = True

        if not result["flv_url"]:
            flv_match = re.search(r'"flv_pull_url"\s*:\s*"(https?://[^"]+\.flv[^"]*)"', html)
            if flv_match:
                result["flv_url"] = flv_match.group(1).replace("\\u002F", "/")
                result["is_living"] = True

        # 检查 status
        if not result["is_living"]:
            status_match = re.search(r'"status"\s*:\s*(\d+)', html)
            if status_match and status_match.group(1) == "2":
                result["is_living"] = True

    except Exception as e:
        print(f"[直播] PC 端页面请求失败: {e}")

    return result


def _extract_room_info(room: dict, result: dict):
    """从 room 数据中提取直播信息。"""
    status = room.get("status") or room.get("live_status")
    if status == 2 or status == "2":
        result["is_living"] = True

    stream_url = room.get("hls_pull_url") or room.get("stream_url", {})
    if isinstance(stream_url, str) and stream_url:
        result["stream_url"] = stream_url
    elif isinstance(stream_url, dict):
        result["stream_url"] = stream_url.get("hls_pull_url", "")
        result["flv_url"] = stream_url.get("flv_pull_url", "")

    owner = room.get("owner", {})
    if isinstance(owner, dict) and owner.get("nickname"):
        result["anchor_name"] = owner["nickname"]


# 清晰度优先级（从高到低）
_QUALITY_PRIORITY = ["FULL_HD1", "FULL_HD", "HD1", "HD", "SD1", "SD2", "SD"]


def _pick_best_quality(quality_dict: dict) -> str:
    """从清晰度字典中选择最高清晰度的 URL。"""
    for key in _QUALITY_PRIORITY:
        url = quality_dict.get(key, "")
        if isinstance(url, str) and url:
            return url
    # fallback: 取第一个值
    for v in quality_dict.values():
        if isinstance(v, str) and v:
            return v
    return ""
