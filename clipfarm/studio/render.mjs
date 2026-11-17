// ClipFarm render worker: node render.mjs <props.json> <out.mp4> [--frames=0,30,60 --stills-dir=dir]
// Bundles the Remotion project once (cached in .bundle/) and renders the "Clip" composition.
import {bundle} from '@remotion/bundler';
import {renderMedia, renderStill, selectComposition} from '@remotion/renderer';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [, , propsPath, outPath, ...rest] = process.argv;
const opts = Object.fromEntries(rest.map((a) => a.replace(/^--/, '').split('=')));
const props = JSON.parse(fs.readFileSync(propsPath, 'utf8'));

const srcDir = path.join(here, 'src', 'remotion');
const bundleDir = path.join(here, '.bundle');
const stampFile = path.join(bundleDir, '.stamp');
const stamp = fs
	.readdirSync(srcDir)
	.map((f) => `${f}:${fs.statSync(path.join(srcDir, f)).mtimeMs}`)
	.join('|');

let serveUrl = bundleDir;
if (!fs.existsSync(stampFile) || fs.readFileSync(stampFile, 'utf8') !== stamp) {
	process.stderr.write('bundling...\n');
	serveUrl = await bundle({entryPoint: path.join(srcDir, 'index.ts'), outDir: bundleDir});
	fs.writeFileSync(stampFile, stamp);
}

const composition = await selectComposition({serveUrl, id: 'Clip', inputProps: props});
const concurrency = Number(opts.concurrency || process.env.REMOTION_CONCURRENCY || 0) || Math.max(1, os.cpus().length);
const common = {
	composition,
	serveUrl,
	inputProps: props,
	chromiumOptions: {gl: 'swangle'},
	licenseKey: process.env.REMOTION_LICENSE_KEY || 'free-license',
	isProduction: process.env.CLIPFARM_PRODUCTION === '1',
	logLevel: 'error',
	timeoutInMilliseconds: 120000,
	offthreadVideoCacheSizeInBytes: 1024 * 1024 * 1024,
};

if (opts.frames) {
	const dir = opts['stills-dir'];
	fs.mkdirSync(dir, {recursive: true});
	for (const f of opts.frames.split(',').map(Number)) {
		await renderStill({...common, frame: f, output: path.join(dir, `f_${String(f).padStart(5, '0')}.jpg`), imageFormat: 'jpeg', jpegQuality: 88});
	}
	process.stdout.write(JSON.stringify({stills: dir}) + '\n');
} else {
	let last = -1;
	await renderMedia({
		...common,
		codec: 'h264',
		outputLocation: outPath,
		muted: true,
		videoBitrate: process.env.VIDEO_BITRATE || '14M',
		x264Preset: process.env.X264_PRESET || 'medium',
		pixelFormat: 'yuv420p',
		concurrency,
		onProgress: ({progress}) => {
			const p = Math.floor(progress * 100);
			if (p !== last) {
				last = p;
				process.stdout.write(`PROGRESS ${progress.toFixed(3)}\n`);
			}
		},
	});
	process.stdout.write(JSON.stringify({out: outPath}) + '\n');
}
