# Caption style research: @nocodealex → ClipFarm caption engine

## 1. Sources and method

| Source | Result |
|---|---|
| `instagram.com/nocodealex/reels` (yt-dlp, anonymous) | **Blocked**: Instagram redirects anonymous profile requests to `accounts/login`. ClipFarm does not bypass login walls. |
| `youtube.com/@nocodealex/shorts` | Listing works, downloads need a signed-in session ("confirm you're not a bot") from this datacenter IP. |
| `tiktok.com/@nocodealex` | **Worked**: 8 most recent public videos downloaded (posted 2023-01-20 … 2023-01-24, 576x1024, 25-30 fps). |

The analysed videos are the account's public cross-posts. They are older than the current Reels the brief refers to.
To analyse the latest Reels, export your own Instagram cookies and run:

```bash
COOKIES_FILE=~/instagram_cookies.txt .venv/bin/python scripts/caption_research.py https://www.instagram.com/nocodealex/reels/ --n 10
```

The script downloads the latest Reels and writes contact sheets and 30 fps caption strips to `docs/caption_ref/research/` for review.

Method: frames at 2 fps (contact sheets of the first 8 s), 30 fps strips around page changes, full-resolution crops
of caption regions, and counts of words per page.

Reference crops: `docs/caption_ref/ref_emphasis_green.jpg`, `ref_emphasis_red.jpg`, `ref_emphasis_blue.jpg`, `ref_stroke_detail.jpg`.

## 2. Observations (8 videos)

- **Font:** a geometric grotesk (closest free match is Montserrat Bold/ExtraBold), ALL CAPS, tight tracking (about −0.01 em).
- **Words per page:** 2–6, mostly 3–4. Pages are 2 balanced lines and do not cross sentences: a page ends at `,` / `.` / `…`
  and the next sentence starts a new page.
  - Quotes keep their punctuation (`"YOU DON'T GET REALLY RICH…`).
- **Page rate:** about 1 page per 0.9–1.2 s, which is 3–4 words per second of speech.
- **Emphasis:** exactly one keyword per page at most, about 1 page in 3. The emphasis colour follows the word's meaning:
  - green for gain or positive words ("RICH", "BENEFITS");
  - red for loss, negation or pain ("DON'T", "HEADACHES");
  - blue for key nouns ("EXIT", "AGENCY").

  Same font and size as the neutral words; only the colour changes.
- **Animation:** none in these 2023 posts. Pages hard-cut on the spoken word and there is no pop-in, which the 30 fps strip confirms.
- **Position:** centred, at about 74–79 % of the frame height (chest, under the chin). It never covers the face.
- **Colour and readability:** white text, a thin black outline (about 4–5 px at 1080 width) and a soft dark drop shadow.
  No background boxes, emojis or stickers.

## 3. What we keep and what we push further (per the ClipFarm brief)

The 2023 posts give the base system: short sentence-bounded pages, ALL CAPS geometric sans, one semantic accent word,
a dark outline with a soft shadow, and placement clear of the face.

The brief asks for the current, more editorial feel, and the engine adds that on top:

| Aspect | @nocodealex 2023 | ClipFarm "dynamic" |
|---|---|---|
| Neutral font | Montserrat-like caps | Archivo Black (mixed case) or Montserrat Black caps ("bold") |
| Emphasis | colour only | contrasting font from a curated set (serif italic, marker, condensed, mono, display) + 1.42x size + accent colour + slight rotation |
| Colour logic | semantic green/red/blue | streamer palette (accent / accent2 alternating), so the brand stays consistent |
| Animation | hard cut | spring pop-in per page; words light up as spoken; emphasis pops with overshoot |
| Position | 75-79 % height | split layout: just below the facecam divider (~51 %); camera-only: under the face, inside 15-72 % (clear of the platform UI) |

## 4. Implementation spec (`studio/src/remotion/Captions.tsx`, `clipfarm/captions.py`)

- **Canvas:** 1080x1920. **Safe width:** 840 px, centred (x 120–960), clear of the right-hand action column.
  **Vertical safe band:** 15 %–72 % of the height.
- **Sizes:** neutral 92 px × font `size_scale` normalisation; emphasis × 1.42; the first page of every clip × 1.3 (big opening page).
  - Real glyph advances (fontTools) fit each line into the 840 px width; minimum 52 px.
- **Stroke:** `max(6px, 0.12 × size)` with `paint-order: stroke fill`, so the outline sits outside the glyph.
  **Shadow:** `0 0.06em 0.2em rgba(0,0,0,.5)`.
- **Pages:** 1–4 words.
  - A new page starts at sentence ends, gaps over 450 ms, more than 20 characters, more than 1.7 s, or the cold-open cut.
  - Pages shorter than 260 ms merge with the next page.
  - An emphasis word ends the page early, so it never sits in a crowded line.
- **Rhythm:** the neutral font is fixed per clip. Emphasis fonts cycle through a seeded shuffle; the seed is a hash of job and clip.
  - The accent colour alternates accent / accent2.
  - Two consecutive pages never share the same (neutral, emphasis-font, colour) combination.
  - Emphasis is spaced at most once per 2 pages, unless the LLM marks the word as must-emphasize (score ≥ 5).
- **Animation (Remotion `spring`):**
  - Page enter: `damping 15, stiffness 260, mass 0.55`. Scale 0.84 → 1, translateY 26 → 0 px, opacity 0 → 1 over about 5 frames.
  - Word activation: inactive words at opacity 0.5. The active word fades to 1 in 3 frames with a scale bump of up to 1.05.
  - Emphasis: `damping 10, stiffness 240, mass 0.6`. Scale 0.55 → 1 with natural overshoot; rotation ±2–4.5°, seeded.
  - Page exit: 2-frame fade.
- **Turkish:**
  - Locale-aware upper-casing (i→İ, ı→I) in Python, never CSS `text-transform`.
  - `lang="tr"` on spans and `font-synthesis: none`.
  - Every font is checked for ğ Ğ ş Ş ı İ ö Ö ü Ü ç Ç (see `assets/fonts/fonts.json`, renders in `docs/font_test/`).
  - QA re-checks every rendered glyph against the font cmap.
- **Styles available per clip:**
  - `dynamic` (default): as above.
  - `bold`: caps, max 3 words per page.
  - `clean`: a single font family; emphasis is size and colour only, closest to the 2023 look.
  - `classic`: paging straight from `@remotion/captions` `createTikTokStyleCaptions`, with the active word highlighted.
