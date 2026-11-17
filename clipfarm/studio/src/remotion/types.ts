import type {Caption} from '@remotion/captions';

export type Palette = {
	primary: string;
	accent: string;
	accent2: string;
	text: string;
	stroke: string;
};

export type FontDef = {
	id: string;
	family: string;
	file: string;
	weight: number;
	style: string;
	sizeScale: number;
	letterSpacingEm: number;
	upper: boolean;
};

export type CaptionWord = {
	text: string;
	startMs: number;
	endMs: number;
	emph: boolean;
	font: string;
	size: number;
};

export type CaptionPage = {
	startMs: number;
	endMs: number;
	lines: {words: CaptionWord[]; width: number}[];
	color: string;
	rotate: number;
	hasEmph: boolean;
};

export type Segment = {
	kind: 'cold' | 'main';
	srcStart: number; // seconds in the hq source file
	srcEnd: number;
	outStart: number; // frames
	outEnd: number;
};

export type Rect = {x: number; y: number; w: number; h: number};

export type TrackKey = {t: number; x: number; y: number};

export type Layout = {
	mode: 'split' | 'cam' | 'game';
	topH: number;
	divider: number;
	bottomH: number;
	facecam: Rect | null;
	track: {w: number; h: number; keys: TrackKey[]};
};

export type ZoomEvent = {f: number; attack: number; hold: number; release: number; amount: number; kind: string};
export type BaseScale = {from: number; to: number; scale: number};
export type Shake = {f: number; dur: number; amp: number};
export type Flash = {f: number; dur: number};
export type SfxEvent = {cat: string; file: string; start: number; gainDb: number};

export type ClipProps = {
	fps: number;
	width: number;
	height: number;
	durationInFrames: number;
	video: {src: string; width: number; height: number; fps: number};
	muteVideo: boolean;
	segments: Segment[];
	layout: Layout;
	effects: {zooms: ZoomEvent[]; base: BaseScale[]; shakes: Shake[]; flashes: Flash[]; enabled: boolean};
	captions: {
		enabled: boolean;
		style: string;
		y: number; // centre of the caption block in px
		pages: CaptionPage[];
		tokens: Caption[];
		fonts: FontDef[];
		neutralFont: string;
	};
	fontBaseUrl: string;
	sfxBaseUrl: string;
	sfx: SfxEvent[];
	playSfx: boolean;
	palette: Palette;
	label: {enabled: boolean; platform: 'kick' | 'twitch'; name: string};
	seed: number;
};
