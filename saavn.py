import base64
import html as html_lib
import json
from typing import Any

import httpx
from Crypto.Cipher import DES

BASE_URL = "https://www.jiosaavn.com/api.php"

_DES_KEY = b"38346591"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.jiosaavn.com/",
}
_COMMON_PARAMS = {
    "_format": "json",
    "_marker": "0",
    "api_version": "4",
    "ctx": "web6dot0",
}


class SaavnError(Exception):
    pass


def decrypt_media_url(encrypted_url: str) -> str:
    try:
        raw = base64.b64decode(encrypted_url)
        decrypted = DES.new(_DES_KEY, DES.MODE_ECB).decrypt(raw)
    except Exception as exc:
        raise SaavnError(f"Failed to decrypt media URL: {exc}") from exc
    pad = decrypted[-1] if decrypted else 0
    if 1 <= pad <= 8 and decrypted.endswith(bytes([pad]) * pad):
        decrypted = decrypted[:-pad]
    return decrypted.decode("utf-8", errors="ignore")


def _best_media_url(encrypted_url: str, has_320: bool) -> tuple[str, str]:
    base_url = decrypt_media_url(encrypted_url)
    if has_320:
        upgraded = base_url.replace("_96.mp4", "_320.mp4")
        if upgraded != base_url:
            return upgraded, "320kbps"
    return base_url, "96kbps"


def _clean(text: Any) -> str:
    return html_lib.unescape(str(text)).strip() if text else ""


def _clean_image(url: str) -> str:
    return url.replace("150x150", "500x500").replace("50x50", "500x500")


def _to_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _parse_json(text: str) -> Any:
    text = text.strip()
    if not text.startswith("{") and not text.startswith("["):
        brace = text.find("{")
        if brace > 0:
            text = text[brace:]
    return json.loads(text)


def parse_song(raw: dict[str, Any]) -> dict[str, Any]:
    more = raw.get("more_info") or {}
    artist_map = more.get("artistMap") or {}
    primary_artists = [
        a.get("name") for a in (artist_map.get("primary_artists") or []) if a.get("name")
    ]
    has_320 = str(more.get("320kbps", "")).lower() == "true"
    encrypted = more.get("encrypted_media_url") or ""
    media_url, media_quality = (None, None)
    if encrypted:
        media_url, media_quality = _best_media_url(encrypted, has_320)
    return {
        "id": raw.get("id"),
        "title": _clean(raw.get("title")),
        "subtitle": _clean(raw.get("subtitle")) or None,
        "album": _clean(more.get("album")) or None,
        "primary_artists": primary_artists or None,
        "duration_seconds": _to_int(more.get("duration")),
        "language": raw.get("language"),
        "year": _to_int(raw.get("year")),
        "play_count": _to_int(raw.get("play_count"), 0),
        "image": _clean_image(raw.get("image", "")) or None,
        "media_url": media_url,
        "media_quality": media_quality,
    }


class SaavnClient:
    def __init__(self, timeout: float = 15.0, retries: int = 1) -> None:
        self._client = httpx.Client(headers=_HEADERS, timeout=timeout)
        self._retries = max(0, retries)

    def _get_json(self, params: dict[str, Any]) -> Any:
        merged = {**_COMMON_PARAMS, **params}
        last_exc: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                resp = self._client.get(BASE_URL, params=merged)
                resp.raise_for_status()
                return _parse_json(resp.text)
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
        raise SaavnError(f"JioSaavn request failed: {last_exc}")

    def search_songs(self, query: str, page: int = 1, limit: int = 36) -> dict[str, Any]:
        data = self._get_json(
            {"__call": "search.getResults", "q": query, "p": page, "n": limit}
        )
        if not isinstance(data, dict) or "results" not in data:
            raise SaavnError("Unexpected search response from JioSaavn")
        songs = [
            parse_song(r)
            for r in data.get("results") or []
            if isinstance(r, dict) and r.get("type", "song") == "song"
        ]
        return {
            "total": _to_int(data.get("total"), 0) or 0,
            "start": _to_int(data.get("start"), 1) or 1,
            "songs": songs,
        }
