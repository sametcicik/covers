import React from 'react';
import {Composition} from 'remotion';
import {ClipComposition} from './ClipComposition';
import type {ClipProps} from './types';

export const defaultClipProps: ClipProps = {
	fps: 30,
	width: 1080,
	height: 1920,
	durationInFrames: 90,
	video: {src: '', width: 1920, height: 1080, fps: 30},
	muteVideo: true,
	segments: [],
	layout: {mode: 'game', topH: 720, divider: 8, bottomH: 1192, facecam: null, track: {w: 0.316, h: 1, keys: []}},
	effects: {zooms: [], base: [], shakes: [], flashes: [], enabled: true},
	captions: {enabled: true, style: 'dynamic', y: 960, pages: [], tokens: [], fonts: [], neutralFont: ''},
	fontBaseUrl: '',
	sfxBaseUrl: '',
	sfx: [],
	playSfx: false,
	palette: {primary: '#7C5CFF', accent: '#FFD84D', accent2: '#4DE1FF', text: '#FFFFFF', stroke: '#0A0A0F'},
	label: {enabled: true, platform: 'twitch', name: 'streamer'},
	seed: 1,
};

export const RemotionRoot: React.FC = () => (
	<Composition
		id="Clip"
		component={ClipComposition}
		width={1080}
		height={1920}
		fps={30}
		durationInFrames={90}
		defaultProps={defaultClipProps}
		calculateMetadata={({props}) => ({
			durationInFrames: Math.max(1, props.durationInFrames),
			fps: props.fps,
			width: props.width,
			height: props.height,
		})}
	/>
);
