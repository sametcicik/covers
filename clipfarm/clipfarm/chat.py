"""Chat replay download (Twitch GQL, Kick history API).

Output: list of {"t": vod_seconds, "u": user, "m": text, "e": emote_count, "w": weight}
`w` > 1 when a busy chat is sampled instead of fully paged: sum(w) over a window estimates the
true message count, so rate signals stay correct while requests stay bounded.
"""
from __future__ import annotations

import datetime as dt
import re
import time

import httpx

from .platforms import TWITCH_CLIENT_ID, UA
from .util import log

TWITCH_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"
KICK_EMOTE = re.compile(r"\[emote:\d+:([^\]]+)\]")


def _twitch_page(client: httpx.Client, vid: str, offset: float | None = None, cursor: str | None = None) -> dict:
    var = {"videoID": vid}
    if cursor:
        var["cursor"] = cursor
    else:
        var["contentOffsetSeconds"] = int(offset or 0)
    body = [{"operationName": "VideoCommentsByOffsetOrCursor", "variables": var,
             "extensions": {"persistedQuery": {"version": 1, "sha256Hash": TWITCH_HASH}}}]
    for attempt in range(4):
        try:
            r = client.post("https://gql.twitch.tv/gql", json=body)
            r.raise_for_status()
            data = r.json()[0].get("data") or {}
            return ((data.get("video") or {}).get("comments")) or {}
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return {}


def _twitch_msgs(page: dict) -> list[dict]:
    out = []
    for e in page.get("edges") or []:
        n = e.get("node") or {}
        frags = ((n.get("message") or {}).get("fragments")) or []
        text = "".join(f.get("text") or "" for f in frags)
        emotes = sum(1 for f in frags if f.get("emote"))
        out.append({"t": float(n.get("contentOffsetSeconds") or 0), "u": ((n.get("commenter") or {}).get("login") or ""),
                    "m": text, "e": emotes, "w": 1.0})
    return out


def twitch_chat(video_id: str, start: float, end: float, max_requests: int = 1500, progress=None) -> list[dict]:
    vid = video_id.lstrip("v")
    msgs: list[dict] = []
    step = 8.0
    with httpx.Client(headers={"Client-ID": TWITCH_CLIENT_ID, "User-Agent": UA}, timeout=30) as c:
        t = start
        req = 0
        while t < end and req < max_requests:
            page = _twitch_page(c, vid, offset=t)
            req += 1
            got = _twitch_msgs(page)
            got = [m for m in got if m["t"] >= t - 1]
            if not got:
                t += step
                continue
            span = got[-1]["t"] - got[0]["t"]
            if span < step and (page.get("pageInfo") or {}).get("hasNextPage"):
                # busy chat: this page covers less than `step` seconds -> sample, weight by density
                w = step / max(span, 0.5)
                for m in got:
                    m["w"] = round(w, 3)
                msgs.extend(got)
                t += step
            else:
                msgs.extend(got)
                t = max(t + 1.0, got[-1]["t"] + 0.01)
            if progress:
                progress(min(1.0, (t - start) / max(1.0, end - start)))
    log.info("twitch chat: %d msgs (%d requests)", len(msgs), req)
    return [m for m in msgs if start <= m["t"] <= end]


def kick_chat(channel_id: str | int, vod_start_ts: float, start: float, end: float, max_requests: int = 3000,
              progress=None) -> list[dict]:
    """Kick history endpoint returns messages after start_time and a cursor (µs) for the next message."""
    msgs: list[dict] = []
    seen = set()
    cur = vod_start_ts + start
    stop = vod_start_ts + end
    req = 0
    with httpx.Client(headers={"User-Agent": UA, "Accept": "application/json"}, timeout=30) as c:
        while cur < stop and req < max_requests:
            iso = dt.datetime.fromtimestamp(cur, dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            try:
                r = c.get(f"https://kick.com/api/v2/channels/{channel_id}/messages", params={"start_time": iso})
                r.raise_for_status()
                d = r.json().get("data") or {}
            except Exception:  # noqa: BLE001
                req += 1
                cur += 5
                time.sleep(1)
                continue
            req += 1
            for m in d.get("messages") or []:
                if m.get("id") in seen:
                    continue
                seen.add(m.get("id"))
                ts = dt.datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")).timestamp()
                content = m.get("content") or ""
                emotes = len(KICK_EMOTE.findall(content))
                text = KICK_EMOTE.sub(lambda mm: mm.group(1), content)
                msgs.append({"t": ts - vod_start_ts, "u": ((m.get("sender") or {}).get("username") or ""),
                             "m": text, "e": emotes, "w": 1.0})
            try:
                nxt = int(d.get("cursor")) / 1e6
            except (TypeError, ValueError):
                nxt = cur + 5
            cur = nxt if nxt > cur else cur + 2
            if progress:
                progress(min(1.0, (cur - vod_start_ts - start) / max(1.0, end - start)))
    msgs.sort(key=lambda m: m["t"])
    log.info("kick chat: %d msgs (%d requests)", len(msgs), req)
    return [m for m in msgs if start <= m["t"] <= end]


def fetch_chat(meta: dict, start: float, end: float, progress=None) -> list[dict]:
    try:
        if meta["platform"] == "twitch" and meta["kind"] == "vod":
            return twitch_chat(meta["id"], start, end, progress=progress)
        if meta["platform"] == "kick" and meta["kind"] == "vod" and meta.get("timestamp") and meta.get("channel_id"):
            return kick_chat(meta["channel_id"], float(meta["timestamp"]), start, end, progress=progress)
    except Exception as e:  # noqa: BLE001
        log.warning("chat replay failed: %s", e)
    return []
