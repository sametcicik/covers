import {createTikTokStyleCaptions} from '@remotion/captions';
import React, {useMemo} from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import type {CaptionPage, ClipProps, FontDef} from './types';

// Animation constants (see CAPTION_STYLE.md "Implementation spec")
const PAGE_SPRING = {damping: 15, stiffness: 260, mass: 0.55};
const EMPH_SPRING = {damping: 10, stiffness: 240, mass: 0.6};
const WORD_FADE_FRAMES = 3;
const INACTIVE_OPACITY = 0.5;

type Props = {captions: ClipProps['captions']; palette: ClipProps['palette']};

const fontMap = (fonts: FontDef[]) => Object.fromEntries(fonts.map((f) => [f.id, f]));

const Word: React.FC<{
	text: string;
	font: FontDef | undefined;
	size: number;
	emph: boolean;
	startFrame: number;
	color: string;
	textColor: string;
	stroke: string;
	rotate: number;
	lang: string;
}> = ({text, font, size, emph, startFrame, color, textColor, stroke, rotate, lang}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const local = frame - startFrame;
	const active = local >= 0;
	const fade = interpolate(local, [0, WORD_FADE_FRAMES], [INACTIVE_OPACITY, 1], {
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
	});
	const bump = active ? spring({frame: local, fps, config: {damping: 12, stiffness: 300, mass: 0.4}}) : 0;
	let scale = 1 + 0.05 * Math.sin(Math.min(1, bump) * Math.PI);
	let rot = 0;
	if (emph) {
		const s = active ? spring({frame: local, fps, config: EMPH_SPRING}) : 0;
		scale = active ? interpolate(s, [0, 1], [0.55, 1]) : 0.92;
		rot = rotate * Math.min(1, s);
	}
	const strokeW = Math.max(6, size * 0.12);
	return (
		<span
			lang={lang}
			style={{
				display: 'inline-block',
				fontFamily: font ? `'${font.family}'` : 'sans-serif',
				fontWeight: font?.weight ?? 900,
				fontStyle: font?.style ?? 'normal',
				fontSize: size,
				lineHeight: 1.04,
				letterSpacing: `${font?.letterSpacingEm ?? 0}em`,
				color: emph ? (active ? color : textColor) : textColor,
				opacity: emph ? (active ? 1 : INACTIVE_OPACITY) : fade,
				WebkitTextStroke: `${strokeW}px ${stroke}`,
				paintOrder: 'stroke fill',
				textShadow: `0 ${size * 0.06}px ${size * 0.2}px rgba(0,0,0,0.5)`,
				transform: `scale(${scale}) rotate(${rot}deg)`,
				transformOrigin: '50% 60%',
				margin: `0 ${size * 0.13}px`,
				whiteSpace: 'nowrap',
				fontSynthesis: 'none',
			}}
		>
			{text}
		</span>
	);
};

const Page: React.FC<{page: CaptionPage; fonts: Record<string, FontDef>; palette: Props['palette']; y: number; lang: string}> = ({
	page,
	fonts,
	palette,
	y,
	lang,
}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const start = Math.round((page.startMs / 1000) * fps);
	const end = Math.round((page.endMs / 1000) * fps);
	const s = spring({frame: frame - start, fps, config: PAGE_SPRING});
	const outFade = interpolate(frame, [end - 2, end], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
	const scale = interpolate(s, [0, 1], [0.84, 1]);
	const ty = interpolate(s, [0, 1], [26, 0]);
	return (
		<div
			style={{
				position: 'absolute',
				left: 0,
				right: 0,
				top: y,
				transform: `translateY(-50%) translateY(${ty}px) scale(${scale})`,
				opacity: Math.min(1, s * 1.8) * outFade,
				display: 'flex',
				flexDirection: 'column',
				alignItems: 'center',
				gap: 4,
			}}
		>
			{page.lines.map((line, li) => (
				<div key={li} style={{display: 'flex', alignItems: 'baseline', justifyContent: 'center'}}>
					{line.words.map((w, wi) => (
						<Word
							key={wi}
							text={w.text}
							font={fonts[w.font]}
							size={w.size}
							emph={w.emph}
							startFrame={Math.round((w.startMs / 1000) * fps)}
							color={page.color}
							textColor={palette.text}
							stroke={palette.stroke}
							rotate={page.rotate}
							lang={lang}
						/>
					))}
				</div>
			))}
		</div>
	);
};

/** "classic": paging straight from @remotion/captions, active word highlighted. */
const Classic: React.FC<Props & {fonts: Record<string, FontDef>; neutral: string}> = ({captions, palette, fonts, neutral}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const {pages} = useMemo(
		() => createTikTokStyleCaptions({captions: captions.tokens, combineTokensWithinMilliseconds: 900}),
		[captions.tokens],
	);
	const ms = (frame / fps) * 1000;
	const page = pages.find((p) => ms >= p.startMs && ms < p.startMs + p.durationMs);
	if (!page) {
		return null;
	}
	const f = fonts[neutral];
	const local = Math.round(((ms - page.startMs) / 1000) * fps);
	const s = spring({frame: local, fps, config: PAGE_SPRING});
	return (
		<div
			style={{
				position: 'absolute',
				left: 120,
				right: 120,
				top: captions.y,
				transform: `translateY(-50%) scale(${interpolate(s, [0, 1], [0.86, 1])})`,
				textAlign: 'center',
				fontFamily: `'${f?.family}'`,
				fontSize: 82,
				lineHeight: 1.08,
				WebkitTextStroke: `10px ${palette.stroke}`,
				paintOrder: 'stroke fill',
				textShadow: '0 5px 16px rgba(0,0,0,0.5)',
				color: palette.text,
			}}
			lang={captions.lang ?? 'tr'}
		>
			{page.tokens.map((t, i) => (
				<span key={i} style={{color: ms >= t.fromMs && ms < t.toMs ? palette.accent : palette.text, whiteSpace: 'pre'}}>
					{f?.upper ? t.text.toLocaleUpperCase(captions.lang ?? 'tr') : t.text}
				</span>
			))}
		</div>
	);
};

export const Captions: React.FC<Props> = ({captions, palette}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const fonts = useMemo(() => fontMap(captions.fonts), [captions.fonts]);
	if (!captions.enabled) {
		return null;
	}
	if (captions.style === 'classic') {
		return (
			<AbsoluteFill>
				<Classic captions={captions} palette={palette} fonts={fonts} neutral={captions.neutralFont} />
			</AbsoluteFill>
		);
	}
	const ms = (frame / fps) * 1000;
	const page = captions.pages.find((p) => ms >= p.startMs - 1 && ms < p.endMs);
	return <AbsoluteFill>{page ? <Page page={page} fonts={fonts} palette={palette} y={captions.y} lang={captions.lang ?? 'tr'} /> : null}</AbsoluteFill>;
};
