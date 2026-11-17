import {useEffect, useState} from 'react';
import {continueRender, delayRender} from 'remotion';
import type {FontDef} from './types';

const loaded = new Map<string, Promise<void>>();

export const loadFont = (baseUrl: string, f: FontDef): Promise<void> => {
	const key = `${f.family}|${baseUrl}|${f.file}`;
	const existing = loaded.get(key);
	if (existing) {
		return existing;
	}
	const face = new FontFace(f.family, `url(${baseUrl.replace(/\/$/, '')}/${f.file})`, {
		weight: String(f.weight),
		style: f.style,
		display: 'block',
	});
	const p = face
		.load()
		.then((ff) => {
			document.fonts.add(ff);
		})
		.catch((err) => {
			console.error('font failed', f.family, err);
		});
	loaded.set(key, p);
	return p;
};

/** Blocks rendering until every caption font is ready, so no frame is drawn with a fallback font. */
export const useFontsReady = (baseUrl: string, fonts: FontDef[]): boolean => {
	const key = fonts.map((f) => f.id).join(',');
	const [handle] = useState(() => delayRender(`fonts: ${key}`, {timeoutInMilliseconds: 60000}));
	const [ready, setReady] = useState(false);
	useEffect(() => {
		let cancelled = false;
		Promise.all(fonts.map((f) => loadFont(baseUrl, f))).finally(() => {
			if (!cancelled) {
				setReady(true);
				continueRender(handle);
			}
		});
		return () => {
			cancelled = true;
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [baseUrl, key]);
	return ready;
};
