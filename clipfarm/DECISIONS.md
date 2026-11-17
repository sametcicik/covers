# DECISIONS

One line per decision made without asking (newest at the bottom of each section).

## Environment and tooling
- Network was limited to package registries; the environment was switched to unrestricted access, because Twitch, Kick and Instagram CDNs use rotating hostnames (`video-weaver.*`, `scontent-*`) that an allowlist cannot cover.
- The test box has 4 CPU cores, 15.7 GB RAM and no NVIDIA GPU, so the auto-selected backend is `small` int8 on CPU. With a GPU (≥5 GB VRAM) it switches to `large-v3`; NVENC is used when ffmpeg can open it.
- Python 3.13 venv (`.venv`) + Node 22. yt-dlp is installed from the pre-release channel (nightly), as requested; all other packages are stable releases.
- Remotion 4.0.544 was verified on npm (published the day before the build) and is used for both preview (`@remotion/player`) and export (`@remotion/renderer`) from the same composition file.
- Remotion licence: `REMOTION_LICENSE_KEY=free-license` by default, valid for individuals and companies with ≤3 people. Larger companies need a paid key in `.env`.
- Official skills were installed into `.claude/skills/`: `remotion-best-practices`, `remotion-captions`, `remotion-render`, `remotion-multimedia` (remotion-dev/skills) and `frontend-design` (anthropics/skills).
- TwitchDownloaderCLI is installed by `install.sh` from GitHub Releases. This session cannot download release assets (GitHub API 403), so that fallback is untested here.

## Download
- Stage 1 downloads one muxed 360p rendition and extracts the audio from it. Twitch/Kick only ship muxed HLS video, so a separate audio-only track plus 360p video would cost more bandwidth than the 360p alone. `PREVIEW_HEIGHT=0` downloads audio only.
- Format discovery and resolution always go through yt-dlp. Sections are fetched by a native HLS segment fetcher, because yt-dlp `--download-sections` hands HLS to ffmpeg, which ignores `--limit-rate`.
  - The fetcher downloads only the segments that overlap the range, with a shared token-bucket rate limit (`DOWNLOAD_RATE_LIMIT`), parallel fetches and retries.
  - Fallback order: yt-dlp `--download-sections` → streamlink (`--hls-start-offset`) → TwitchDownloaderCLI.
- Stage 2 pads every selected range by `SECTION_PADDING`=5 s on each side.
  - It re-encodes to CFR at the snapped source fps with a short GOP (fps/2) and no B-frames, so Remotion's frame extraction is exact and fast.
  - MAX_HEIGHT is 1080, so 1080p60 is preferred.
- Live input: a Twitch channel resolves to the archive VOD being recorded; a Kick channel resolves to the newest VOD entry, which already exists while live. Either way the analysis covers the stream up to now.
- Cookies: `COOKIES_FILE` is passed to yt-dlp for sub-only VODs.

## Chat
- Twitch uses GQL `VideoCommentsByOffsetOrCursor`. When one page covers less than 8 s (a busy chat), it samples every 8 s and weights each message by density, which bounds request count while keeping the rate signal correct.
- Kick uses `api/v2/channels/{id}/messages?start_time=`, paged with its cursor. Kick history is sparse (≈4 msgs/min for a big channel), so chat signals are switched off below 3 msgs/min. Audio and video then carry the score.
- Chat is shifted 6 s earlier to account for reaction latency.

## Transcript
- faster-whisper (CTranslate2) is used rather than WhisperX: word timestamps are native, it installs with fewer dependencies on Python 3.13, and Silero VAD is built in.
- On GPU or short windows (≤15 min) everything is transcribed. On CPU only the heat-peak windows are transcribed (≈64 s each, `ceil(clips×2.2)+2` windows): `small` runs at ≈0.9× realtime on 4 cores, so a full hour would take about an hour.
- The streamer dictionary is applied three ways: as Whisper `hotwords` + `initial_prompt`, then as exact or suffix-aware (Turkish agglutination) or fuzzy (≥0.86) replacement after decoding.
- Language is auto-detected per job on a 30 s probe and stored in the profile as `language_detected`. A fixed `language` in the profile overrides detection.

## Moments
- Heat curve, 1 Hz: audio (loudness z, energy jump, "shout" = loud + bright) 42 %, chat (rate z, emote/laugh/hype z) 40 %, video (scene cut, game motion, facecam motion) 18 %. Without usable chat the weights are audio 75 % and video 25 %.
- Robust z-scores (rolling median/MAD) have a floor of 0.35·σ, so sparse series cannot explode.
- With an LLM gateway, the LLM reads candidate windows as numbered sentences annotated with signals. It returns start/end sentence, peak, score, category, why, emphasis words, titles, description and hashtags. Final score = 62 % LLM + 38 % signals.
- Without an LLM, a deterministic heuristic searches every sentence span of 18–(max−6) s and scores it on:
  - heat peak, top-5 heat;
  - the lexicon (TR+EN shock, laugh, profanity, numbers);
  - setup→payoff position;
  - speech rate, a 32 s target and starter conjunctions.
- Clip bounds always fall on sentence starts and ends. They are padded into the surrounding silence, and the end may ring out laughter or reaction for up to 2.5 s, stopping before the next sentence.

## Edit and hook
- The peak is the strongest 1.5–4 s window: 85th-percentile loudness plus heat, snapped to word bounds and preferring a phrase start.
- A cold open is used when the peak starts more than 3 s into the clip. It plays the peak (−0.1/+0.15 s), then a 5-frame flash with whoosh + impact, then the clip from its start; the peak replays in place.
- Dead air: word gaps over 0.4 s that are also quiet (gap loudness < speech − 7 dB) are cut down to 0.2 s.
  - Gaps within 3 s before the peak, and after a question, keep up to 0.7 s for comic timing.
  - Loud gaps (laughter, game audio) are never cut.
  - Jump cuts alternate the base scale between 1.00 and 1.06 to hide them.
- Motion:
  - opening punch-in of +12 % within 0.12 s;
  - transition punch +8 %;
  - peak punch +13 % with a 9 px decaying shake;
  - emphasis punch-ins +7 %, at least 3 s apart.
- The max duration of 59 s includes the cold open; the cut is made at a sentence start before the peak.

## Layout
- Faces are detected with OpenCV YuNet (ONNX, 230 KB), falling back to a Haar cascade.
  - The facecam box is found from edge persistence: Canny edges averaged over 48 frames give the overlay's frame lines.
  - If no edges are found, the face box is expanded about 3.4×3.
- Modes: split (game + facecam), cam (face over 17 % of frame height, or two large faces), game (no face).
- Split mode: the facecam panel is 38.5 % of the height, or 35 % when the source facecam is under 300 px tall, which limits upscaling softness.
  - An 8 px divider in the palette's primary colour separates the panels.
  - The game crop follows motion saliency with a strong centre prior, a dead zone and a speed limit, and shifts to avoid duplicating the facecam when that is possible.
- Cam mode tracks the face. With several people it follows the active speaker (mouth-region motion with 1.5 s hysteresis).
- There is no blurred background anywhere: every pixel of the 1080x1920 frame is real footage.

## Captions
- See CAPTION_STYLE.md.
- Instagram needs a login for @nocodealex, so the analysis used his public TikTok cross-posts. `scripts/caption_research.py` re-runs it on the newest Reels with the user's own cookies.
- Font set (all OFL, Turkish-verified): Archivo Black, Montserrat Black, Anton, Instrument Serif Italic, Caveat Brush, JetBrains Mono ExtraBold, Unbounded Black.
  - Rejected: Permanent Marker, Rock Salt and Gochi Hand (no ğ/ş/İ), Sedgwick Ave (dotted I), Syne (Ü dots), Bebas Neue (no lowercase).
- Python builds the pages so they are deterministic, seeded and testable; Remotion only renders them.
- Sizes are fitted with real glyph advances (fontTools). Turkish upper-casing is done in Python.
- The `classic` style uses `@remotion/captions` `createTikTokStyleCaptions` directly.
- Profanity censoring is caption-only (s*ktir); audio is not bleeped. It is controlled by `campaign.censor_profanity`.

## Audio
- Speech chain: 70 Hz high-pass → afftdn (nr 8, gentle) → compressor (−21 dB, 2.8:1, 6/140 ms).
- SFX are gain-matched to 10–14 dB under the measured speech LUFS by category, then ducked with `sidechaincompress` keyed by speech.
- Two-pass `loudnorm` (I −14, TP −1.3, linear), then a −1.2 dBFS limiter. QA verifies −14 ±1 LUFS and TP ≤ −0.9 dBTP.
- Density cap: at most 3 effects per 10 s by default (setting 0–3 maps to 0/2/3/4). Transition sounds have priority.
- Per category, variations rotate through a seeded shuffle and the same file never plays twice in a row.
- SFX library: 58 Mixkit sounds (Mixkit SFX Free License, commercial use allowed) plus 1 synthesized CC0 riser.
  - Pixabay (bot protection, 403) and Freesound (API key needed) were not used.
  - Any category short of 5 variations is topped up with synthesized sounds.
- Background music is off by default (`music.enabled=false`), and no music library is shipped.

## Render and output
- Remotion renders the video only (`muted`), as H.264 yuv420p at 14 Mbps (`VIDEO_BITRATE`) with x264 preset medium. ffmpeg muxes AAC 192 kHz/48 k audio with `-c:v copy` and faststart.
- The output fps is the source fps snapped to an integer (59.94→60), capped by `MAX_FPS`.
- Each clip produces mp4, cover.jpg (taken at the peak), info.json (3 titles, description, hashtags, score, reason, QA, settings, source times), copied to `output/<channel>/<job>/`.

## QA
- Checks:
  - duration under 60 s and 1080x1920;
  - loudness and true peak;
  - black frames (blackdetect);
  - glyph coverage of every caption character in its font, plus mojibake;
  - sentence-boundary start and end;
  - caption sync: the final audio is re-transcribed and word onsets compared, with a median under 80 ms required;
  - captions over faces (YuNet on 10 frames against the caption rectangles);
  - safe area;
  - optional LLM vision review of 9 frames.
- Fixes before up to 2 re-renders: shorten, move captions, offset captions by the measured sync error, fall back to a single font, remix without denoise. If it still fails, the clip is rejected and the next candidate is used, so only passing clips reach the gallery.

## UI
- FastAPI + SSE serves a single-screen React (Vite) app. The Player renders the same `ClipComposition` with live props, so a setting change previews instantly and "Yeniden render et" exports exactly that.
- The streamer's palette primary/accent are injected as CSS variables, so the console re-skins per streamer. Type is Bricolage Grotesque (OFL, npm `@fontsource-variable`).
