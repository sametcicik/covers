"""Link parsing and platform metadata for Kick and Twitch."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import httpx

TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"  # public web client id used by twitch.tv itself
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


@dataclass
class Source:
    url: str
    platform: str  # kick | twitch
    kind: str  # vod | clip | live | channel
    channel: str  # slug/login
    video_id: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


_PATTERNS = [
    ("twitch", "vod", r"twitch\.tv/(?:[\w]+/)?videos?/(?P<id>\d+)"),
    ("twitch", "clip", r"clips\.twitch\.tv/(?P<id>[\w-]+)"),
    ("twitch", "clip", r"twitch\.tv/(?P<ch>\w+)/clip/(?P<id>[\w-]+)"),
    ("twitch", "channel", r"twitch\.tv/(?P<ch>\w+)/?(?:\?.*)?$"),
    ("kick", "vod", r"kick\.com/(?P<ch>[\w-]+)/videos/(?P<id>[0-9a-f-]{36})"),
    ("kick", "vod", r"kick\.com/video/(?P<id>[0-9a-f-]{36})"),
    ("kick", "clip", r"kick\.com/(?P<ch>[\w-]+)/clips/(?P<id>clip_\w+)"),
    ("kick", "clip", r"kick\.com/(?P<ch>[\w-]+)/?\?(?:.*&)?clip=(?P<id>clip_\w+)"),
    ("kick", "channel", r"kick\.com/(?P<ch>[\w-]+)/?(?:\?.*)?$"),
]


def parse_link(url: str) -> Source:
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    for platform, kind, pat in _PATTERNS:
        m = re.search(pat, url)
        if m:
            g = m.groupdict()
            return Source(url=url, platform=platform, kind=kind, channel=(g.get("ch") or "").lower(),
                          video_id=g.get("id"))
    raise ValueError(f"Desteklenmeyen link: {url} (Kick veya Twitch VOD/klip/kanal linki gerekli)")


# ---------------------------------------------------------------- Twitch

def twitch_gql(query: str, variables: dict | None = None) -> dict:
    r = httpx.post("https://gql.twitch.tv/gql", json={"query": query, "variables": variables or {}},
                   headers={"Client-ID": TWITCH_CLIENT_ID, "User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json().get("data") or {}


def twitch_channel(login: str) -> dict:
    d = twitch_gql("""query($l:String!){user(login:$l){id login displayName description profileImageURL(width:300)
      bannerImageURL offlineImageURL primaryColorHex stream{id createdAt}
      panels{... on DefaultPanel{imageURL title}}
      videos(first:1,type:ARCHIVE,sort:TIME){edges{node{id status lengthSeconds createdAt}}}}}""", {"l": login})
    u = d.get("user") or {}
    if not u:
        return {}
    vids = [e["node"] for e in (u.get("videos") or {}).get("edges", [])]
    return {
        "platform": "twitch",
        "slug": u["login"],
        "display_name": u.get("displayName") or u["login"],
        "avatar": u.get("profileImageURL"),
        "banner": u.get("bannerImageURL"),
        "offline": u.get("offlineImageURL"),
        "panels": [p["imageURL"] for p in (u.get("panels") or []) if p and p.get("imageURL")],
        "brand_color": ("#" + u["primaryColorHex"]) if u.get("primaryColorHex") else None,
        "is_live": bool(u.get("stream")),
        "latest_archive": vids[0] if vids else None,
    }


def twitch_video_owner(video_id: str) -> str | None:
    d = twitch_gql("query($id:ID!){video(id:$id){owner{login}}}", {"id": video_id})
    return (((d.get("video") or {}).get("owner")) or {}).get("login")


# ---------------------------------------------------------------- Kick

def kick_channel(slug: str) -> dict:
    r = httpx.get(f"https://kick.com/api/v2/channels/{slug}", headers={"User-Agent": UA, "Accept": "application/json"},
                  timeout=30, follow_redirects=True)
    if r.status_code != 200:
        return {}
    j = r.json()
    off = j.get("offline_banner_image") or {}
    return {
        "platform": "kick",
        "slug": j.get("slug") or slug,
        "id": j.get("id"),
        "display_name": (j.get("user") or {}).get("username") or slug,
        "avatar": (j.get("user") or {}).get("profile_pic"),
        "banner": (j.get("banner_image") or {}).get("url"),
        "offline": off.get("src") if isinstance(off, dict) else None,
        "panels": [],
        "brand_color": None,
        "is_live": bool(j.get("livestream")),
        "livestream": j.get("livestream"),
    }


def channel_info(platform: str, slug: str) -> dict:
    try:
        return twitch_channel(slug) if platform == "twitch" else kick_channel(slug)
    except Exception:
        return {}
