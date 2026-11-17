import React from 'react';
import {
	AbsoluteFill,
	Audio,
	Easing,
	interpolate,
	OffthreadVideo,
	Sequence,
	useCurrentFrame,
	useVideoConfig,
} from 'remotion';
import {Captions} from './Captions';
import {useFontsReady} from './fonts';
import type {ClipProps, Rect, Segment, TrackKey} from './types';

const easeOut = Easing.bezier(0.16, 1, 0.3, 1);
const easeInOut = Easing.bezier(0.65, 0, 0.35, 1);

/** Sum of active zoom events (punch-ins) at a given output frame. */
const zoomAt = (frame: number, props: ClipProps): number => {
	if (!props.effects.enabled) {
		return 1;
	}
	const base = props.effects.base.find((b) => frame >= b.from && frame < b.to)?.scale ?? 1;
	let extra = 0;
	for (const z of props.effects.zooms) {
		const a = z.f;
		const b = a + z.attack;
		const c = b + z.hold;
		const d = c + z.release;
		if (frame < a || frame >= d) {
			continue;
		}
		if (frame < b) {
			extra += z.amount * easeOut((frame - a) / Math.max(1, z.attack));
		} else if (frame < c) {
			extra += z.amount;
		} else {
			extra += z.amount * (1 - easeInOut((frame - c) / Math.max(1, z.release)));
		}
	}
	return base + extra;
};

const shakeAt = (frame: number, props: ClipProps, seed: number): {x: number; y: number; r: number} => {
	if (!props.effects.enabled) {
		return {x: 0, y: 0, r: 0};
	}
	let x = 0;
	let y = 0;
	let r = 0;
	for (const s of props.effects.shakes) {
		const t = frame - s.f;
		if (t < 0 || t >= s.dur) {
			continue;
		}
		const decay = Math.pow(1 - t / s.dur, 2);
		x += Math.sin(t * 1.9 + seed) * s.amp * decay;
		y += Math.cos(t * 2.3 + seed * 0.7) * s.amp * 0.7 * decay;
		r += Math.sin(t * 1.3 + seed) * 0.6 * decay;
	}
	return {x, y, r};
};

const trackAt = (keys: TrackKey[], t: number): {x: number; y: number} => {
	if (!keys.length) {
		return {x: 0, y: 0};
	}
	if (t <= keys[0].t) {
		return keys[0];
	}
	const last = keys[keys.length - 1];
	if (t >= last.t) {
		return last;
	}
	let i = 0;
	while (i < keys.length - 1 && keys[i + 1].t < t) {
		i++;
	}
	const a = keys[i];
	const b = keys[i + 1];
	const k = easeInOut((t - a.t) / Math.max(1e-6, b.t - a.t));
	return {x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k};
};

/** A panel showing `crop` (normalized source rect) scaled to cover the panel. */
const Panel: React.FC<{
	props: ClipProps;
	seg: Segment;
	crop: Rect;
	top: number;
	height: number;
	zoom: number;
	focusY?: number;
}> = ({props, seg, crop, top, height, zoom, focusY = 0.45}) => {
	const W = props.width;
	const sw = props.video.width;
	const sh = props.video.height;
	const scale = W / (crop.w * sw);
	const vw = sw * scale;
	const vh = sh * scale;
	const left = -crop.x * sw * scale;
	const topOff = -crop.y * sh * scale + (height - crop.h * sh * scale) / 2;
	return (
		<div style={{position: 'absolute', left: 0, top, width: W, height, overflow: 'hidden'}}>
			<div
				style={{
					position: 'absolute',
					inset: 0,
					transform: `scale(${zoom})`,
					transformOrigin: `50% ${focusY * 100}%`,
				}}
			>
				<OffthreadVideo
					src={props.video.src}
					trimBefore={Math.round(seg.srcStart * props.fps)}
					muted={props.muteVideo}
					pauseWhenBuffering
					style={{position: 'absolute', left, top: topOff, width: vw, height: vh, maxWidth: 'none'}}
				/>
			</div>
		</div>
	);
};

const SegmentView: React.FC<{props: ClipProps; seg: Segment}> = ({props, seg}) => {
	const local = useCurrentFrame();
	const frame = seg.outStart + local;
	const srcT = seg.srcStart + local / props.fps;
	const zoom = zoomAt(frame, props);
	const {layout} = props;
	const pos = trackAt(layout.track.keys, srcT);
	if (layout.mode === 'split' && layout.facecam) {
		return (
			<AbsoluteFill>
				<Panel props={props} seg={seg} crop={layout.facecam} top={0} height={layout.topH} zoom={1 + (zoom - 1) * 1.15} focusY={0.4} />
				<div
					style={{
						position: 'absolute',
						left: 0,
						right: 0,
						top: layout.topH,
						height: layout.divider,
						background: props.palette.primary,
					}}
				/>
				<Panel
					props={props}
					seg={seg}
					crop={{x: pos.x, y: pos.y, w: layout.track.w, h: layout.track.h}}
					top={layout.topH + layout.divider}
					height={layout.bottomH}
					zoom={1 + (zoom - 1) * 0.8}
					focusY={0.5}
				/>
			</AbsoluteFill>
		);
	}
	return (
		<AbsoluteFill>
			<Panel
				props={props}
				seg={seg}
				crop={{x: pos.x, y: pos.y, w: layout.track.w, h: layout.track.h}}
				top={0}
				height={props.height}
				zoom={zoom}
				focusY={0.38}
			/>
		</AbsoluteFill>
	);
};

const Flash: React.FC<{props: ClipProps}> = ({props}) => {
	const frame = useCurrentFrame();
	let o = 0;
	for (const f of props.effects.flashes) {
		const t = frame - f.f;
		if (t >= 0 && t < f.dur) {
			o = Math.max(o, interpolate(t, [0, 1, f.dur], [0.0, 0.92, 0], {extrapolateRight: 'clamp'}));
		}
	}
	if (o <= 0) {
		return null;
	}
	return <AbsoluteFill style={{background: props.palette.text, opacity: o, mixBlendMode: 'screen'}} />;
};

const PlatformIcon: React.FC<{platform: 'kick' | 'twitch'; color: string}> = ({platform, color}) =>
	platform === 'twitch' ? (
		<svg width="30" height="30" viewBox="0 0 24 24" aria-hidden>
			<path fill={color} d="M4 3l-1.5 4v13h5v3h3l3-3h4l5-5V3H4zm15 11l-3 3h-4.5L8.5 20v-3H5V5h14v9zM16 7.5h-2v5h2v-5zm-5 0H9v5h2v-5z" />
		</svg>
	) : (
		<svg width="30" height="30" viewBox="0 0 24 24" aria-hidden>
			<rect x="1" y="1" width="22" height="22" rx="5" fill={color} />
			<path fill="#0b0b0f" d="M6.5 5h3.2v4.3h1.6V7.2h1.6V5h3.2v4.3h-1.6v1.6h-1.6v2.2h1.6v1.6h1.6V19h-3.2v-2.2h-1.6v-2.1H9.7V19H6.5z" />
		</svg>
	);

const Label: React.FC<{props: ClipProps}> = ({props}) => {
	const frame = useCurrentFrame();
	if (!props.label.enabled) {
		return null;
	}
	const o = interpolate(frame, [0, 8], [0, 1], {extrapolateRight: 'clamp'});
	const neutral = props.captions.fonts.find((f) => f.id === props.captions.neutralFont);
	return (
		<div
			style={{
				position: 'absolute',
				top: 196,
				left: 0,
				right: 0,
				display: 'flex',
				justifyContent: 'center',
				opacity: o * 0.95,
			}}
		>
			<div
				style={{
					display: 'flex',
					alignItems: 'center',
					gap: 12,
					padding: '10px 22px 10px 14px',
					borderRadius: 999,
					background: 'rgba(8,8,12,0.42)',
					border: `1.5px solid ${props.palette.primary}55`,
				}}
			>
				<PlatformIcon platform={props.label.platform} color={props.label.platform === 'kick' ? '#53FC18' : '#A970FF'} />
				<span
					style={{
						fontFamily: neutral ? `'${neutral.family}'` : 'sans-serif',
						fontSize: 30,
						color: '#F4F4F6',
						letterSpacing: '0.01em',
						fontSynthesis: 'none',
					}}
				>
					{props.label.name}
				</span>
			</div>
		</div>
	);
};

export const ClipComposition: React.FC<ClipProps> = (props) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	const ready = useFontsReady(props.fontBaseUrl, props.captions.fonts);
	const sh = shakeAt(frame, props, props.seed % 7);
	return (
		<AbsoluteFill style={{backgroundColor: '#050507'}}>
			<AbsoluteFill style={{transform: `translate(${sh.x}px, ${sh.y}px) rotate(${sh.r}deg) scale(${sh.x || sh.y ? 1.02 : 1})`}}>
				{props.segments.map((seg, i) => (
					<Sequence key={i} from={seg.outStart} durationInFrames={seg.outEnd - seg.outStart} name={`${seg.kind} ${i}`}>
						<SegmentView props={props} seg={seg} />
					</Sequence>
				))}
			</AbsoluteFill>
			<Flash props={props} />
			{ready ? <Captions captions={props.captions} palette={props.palette} /> : null}
			<Label props={props} />
			{props.playSfx
				? props.sfx.map((s, i) => (
						<Sequence key={`sfx${i}`} from={Math.round(s.start * fps)} durationInFrames={Math.round(4 * fps)}>
							<Audio src={`${props.sfxBaseUrl.replace(/\/$/, '')}/${s.file}`} volume={Math.pow(10, s.gainDb / 20)} />
						</Sequence>
					))
				: null}
		</AbsoluteFill>
	);
};
