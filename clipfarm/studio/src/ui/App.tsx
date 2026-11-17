import {Player} from '@remotion/player';
import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {ClipComposition} from '../remotion/ClipComposition';
import type {ClipProps, Palette} from '../remotion/types';

type Resolved = {
	platform: 'kick' | 'twitch';
	kind: string;
	channel: string;
	display_name: string;
	avatar?: string | null;
	palette?: (Palette & {source?: string}) | null;
	known: boolean;
	title?: string | null;
};

type Clip = {
	id: string;
	status: 'processing' | 'ready' | 'rejected' | 'error';
	score: number;
	why: string;
	category?: string;
	duration?: number;
	titles?: string[];
	cover?: string | null;
	video?: string | null;
	mtime?: number;
	error?: string;
	qa?: {passed: boolean; issues: string[]; rounds: number};
	info?: {description?: string; hashtags?: string[]; hook?: boolean; settings?: Record<string, unknown>; titles?: string[]};
};

type Job = {
	id: string;
	status: string;
	title: string;
	channel: string;
	platform: 'kick' | 'twitch';
	clips: Clip[];
	streamer?: {name?: string; palette?: Palette; avatar?: string | null; platform?: string};
	error?: string;
	window?: [number, number];
};

type StepState = {frac: number; msg: string; state: 'idle' | 'active' | 'done'};

const STEPS: [string, string][] = [
	['resolve', 'Yayın çözümleniyor'],
	['profile', 'Yayıncı profili ve palet'],
	['download_preview', 'Ses + 360p önizleme'],
	['chat', 'Chat replay'],
	['signals', 'Ses, chat ve görüntü sinyalleri'],
	['transcribe', 'Transkript'],
	['moments', 'En iyi anlar'],
	['download_hq', 'Seçilen aralıklar yüksek kalitede'],
	['edit', 'Kurgu, hook, kadraj'],
	['render', 'Render'],
	['qa', 'Denetim'],
];

const STYLES: [string, string][] = [
	['dynamic', 'Dinamik'],
	['bold', 'Kalın'],
	['clean', 'Sade'],
	['classic', 'Klasik'],
];

const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
	const r = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...init});
	if (!r.ok) {
		let msg = r.statusText;
		try {
			msg = (await r.json()).detail ?? msg;
		} catch {
			/* ignore */
		}
		throw new Error(msg);
	}
	return r.json() as Promise<T>;
};

const fmtDur = (s?: number) => (s ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}` : '');

const PlatformMark: React.FC<{p: 'kick' | 'twitch'}> = ({p}) => (
	<span className={`pmark pmark-${p}`} aria-label={p}>
		{p === 'kick' ? 'K' : 'T'}
	</span>
);

export const App: React.FC = () => {
	const [url, setUrl] = useState('');
	const [count, setCount] = useState(10);
	const [startAt, setStartAt] = useState('');
	const [endAt, setEndAt] = useState('');
	const [resolved, setResolved] = useState<Resolved | null>(null);
	const [resolveErr, setResolveErr] = useState<string | null>(null);
	const [jobId, setJobId] = useState<string | null>(null);
	const [job, setJob] = useState<Job | null>(null);
	const [steps, setSteps] = useState<Record<string, StepState>>({});
	const [busy, setBusy] = useState(false);
	const [selected, setSelected] = useState<string | null>(null);
	const [jobs, setJobs] = useState<{id: string; channel: string; title: string; ready: number; status: string}[]>([]);

	// streamer accent -> whole UI
	const palette = job?.streamer?.palette ?? resolved?.palette ?? null;
	useEffect(() => {
		const root = document.documentElement;
		root.style.setProperty('--accent', palette?.primary ?? '#8B7CF6');
		root.style.setProperty('--accent-2', palette?.accent ?? '#5EEAD4');
	}, [palette]);

	useEffect(() => {
		api<typeof jobs>('/api/jobs').then(setJobs).catch(() => undefined);
	}, [jobId]);

	// resolve as you paste
	useEffect(() => {
		setResolveErr(null);
		if (!/kick\.com|twitch\.tv/.test(url)) {
			setResolved(null);
			return;
		}
		const t = setTimeout(() => {
			api<Resolved>(`/api/resolve?url=${encodeURIComponent(url)}`)
				.then(setResolved)
				.catch((e) => setResolveErr(String(e.message ?? e)));
		}, 350);
		return () => clearTimeout(t);
	}, [url]);

	const refreshJob = useCallback(async (id: string) => {
		try {
			setJob(await api<Job>(`/api/jobs/${id}`));
		} catch {
			/* not created yet */
		}
	}, []);

	// live progress over SSE
	useEffect(() => {
		if (!jobId) {
			return;
		}
		refreshJob(jobId);
		const es = new EventSource(`/api/jobs/${jobId}/events`);
		es.onmessage = (m) => {
			const ev = JSON.parse(m.data);
			if (ev.type === 'progress') {
				setSteps((prev) => {
					const next = {...prev};
					const idx = STEPS.findIndex(([k]) => k === ev.step);
					STEPS.forEach(([k], i) => {
						if (i < idx && next[k]?.state !== 'done' && !['download_hq', 'edit', 'render', 'qa'].includes(k)) {
							next[k] = {frac: 1, msg: next[k]?.msg ?? '', state: 'done'};
						}
					});
					next[ev.step] = {frac: ev.frac, msg: ev.msg || prev[ev.step]?.msg || '', state: ev.frac >= 1 && !['download_hq', 'edit', 'render', 'qa'].includes(ev.step) ? 'done' : 'active'};
					return next;
				});
				if (['qa', 'moments'].includes(ev.step) && ev.frac >= 1) {
					refreshJob(jobId);
				}
			} else if (ev.type === 'done' || ev.type === 'clip') {
				setBusy(false);
				refreshJob(jobId);
			} else if (ev.type === 'error') {
				setBusy(false);
				refreshJob(jobId);
			}
		};
		const poll = setInterval(() => refreshJob(jobId), 8000);
		return () => {
			es.close();
			clearInterval(poll);
		};
	}, [jobId, refreshJob]);

	const start = async () => {
		setBusy(true);
		setSteps({});
		setSelected(null);
		try {
			const toSec = (s: string) => (s ? s.split(':').reduce((a, b) => a * 60 + Number(b), 0) : undefined);
			const r = await api<{job_id: string}>('/api/jobs', {
				method: 'POST',
				body: JSON.stringify({url, clips: count, start: toSec(startAt), end: toSec(endAt)}),
			});
			setJobId(r.job_id);
		} catch (e) {
			setBusy(false);
			setResolveErr(String((e as Error).message));
		}
	};

	const ready = useMemo(
		() => (job?.clips ?? []).filter((c) => c.status === 'ready').sort((a, b) => b.score - a.score),
		[job],
	);
	const pending = (job?.clips ?? []).filter((c) => c.status === 'processing');
	const rejected = (job?.clips ?? []).filter((c) => c.status === 'rejected' || c.status === 'error');
	const selClip = ready.find((c) => c.id === selected) ?? null;

	return (
		<div className="app">
			<header className="bar">
				<div className="brand">
					<span className="brand-mark" />
					ClipFarm
				</div>
				<form
					className="linkform"
					onSubmit={(e) => {
						e.preventDefault();
						if (url && !busy) {
							start();
						}
					}}
				>
					<input
						className="link"
						placeholder="Kick veya Twitch linki yapıştır (VOD, devam eden yayın, klip)"
						value={url}
						onChange={(e) => setUrl(e.target.value)}
						aria-label="Yayın linki"
						spellCheck={false}
					/>
					<label className="mini">
						<span>Klip</span>
						<input type="number" min={1} max={30} value={count} onChange={(e) => setCount(Number(e.target.value))} />
					</label>
					<label className="mini">
						<span>Başlangıç</span>
						<input placeholder="0:00:00" value={startAt} onChange={(e) => setStartAt(e.target.value)} />
					</label>
					<label className="mini">
						<span>Bitiş</span>
						<input placeholder="sona kadar" value={endAt} onChange={(e) => setEndAt(e.target.value)} />
					</label>
					<button className="go" disabled={!url || busy} type="submit">
						{busy ? 'Çalışıyor' : 'Klipleri üret'}
					</button>
				</form>
			</header>

			{(resolved || resolveErr) && (
				<section className="streamer">
					{resolveErr ? (
						<p className="err">{resolveErr}</p>
					) : resolved ? (
						<>
							{resolved.avatar ? <img className="avatar" src={resolved.avatar} alt="" /> : <div className="avatar" />}
							<div className="who">
								<div className="name">
									<PlatformMark p={resolved.platform} /> {resolved.display_name}
								</div>
								<div className="sub">
									{resolved.known ? 'Kayıtlı profil' : 'Yeni yayıncı: profil ve palet oluşturuldu'} · {resolved.kind}
									{resolved.title ? ` · ${resolved.title}` : ''}
								</div>
							</div>
							{resolved.palette && (
								<div className="swatches" title="Yayıncı paleti">
									{(['primary', 'accent', 'accent2', 'text', 'stroke'] as const).map((k) => (
										<span key={k} className="sw" style={{background: resolved.palette![k]}} title={`${k} ${resolved.palette![k]}`} />
									))}
								</div>
							)}
						</>
					) : null}
				</section>
			)}

			<main className="grid">
				<aside className="steps" aria-label="İlerleme">
					{STEPS.map(([k, label]) => {
						const s = steps[k];
						return (
							<div key={k} className={`step ${s?.state ?? 'idle'}`}>
								<div className="step-row">
									<span className="dot" />
									<span className="label">{label}</span>
								</div>
								{s?.state === 'active' && (
									<div className="meter">
										<div style={{width: `${Math.round(s.frac * 100)}%`}} />
									</div>
								)}
								{s?.msg && <div className="msg">{s.msg}</div>}
							</div>
						);
					})}
					{job?.error && <p className="err">{job.error}</p>}
					{jobs.length > 0 && (
						<div className="history">
							<div className="history-title">Önceki işler</div>
							{jobs.slice(0, 8).map((j) => (
								<button key={j.id} className={`hist ${j.id === jobId ? 'on' : ''}`} onClick={() => setJobId(j.id)}>
									<span>{j.channel}</span>
									<span className="hist-n">{j.ready} klip</span>
								</button>
							))}
						</div>
					)}
				</aside>

				<section className="gallery" aria-label="Klipler">
					{job && (
						<div className="gal-head">
							<div>
								<h1>{job.title || job.channel}</h1>
								<p className="sub">
									{ready.length} hazır klip{pending.length ? `, ${pending.length} işleniyor` : ''}
									{rejected.length ? `, ${rejected.length} denetimden geçmedi` : ''}
								</p>
							</div>
							{ready.length > 0 && (
								<a className="ghost" href={`/api/jobs/${job.id}/download.zip`}>
									Hepsini indir (.zip)
								</a>
							)}
						</div>
					)}
					{!job && (
						<div className="empty">
							<p>Bir yayın linki yapıştır. ClipFarm sadece sesi ve 360p önizlemeyi indirir, en iyi anları bulur, sadece o aralıkları yüksek kalitede çekip dikey klip olarak render eder.</p>
						</div>
					)}
					<div className="cards">
						{ready.map((c, i) => (
							<button key={c.id} className={`card ${selected === c.id ? 'sel' : ''}`} onClick={() => setSelected(c.id)}>
								<div className="thumb">
									{c.cover ? <img src={`${c.cover}?v=${c.mtime ?? 0}`} alt="" loading="lazy" /> : null}
									<span className="score">{Math.round(c.score)}</span>
									<span className="rank">{i + 1}</span>
								</div>
								<div className="card-meta">
									<div className="title">{c.titles?.[0] ?? c.id}</div>
									<div className="sub">
										{fmtDur(c.duration)} {c.info?.hook ? '· hook' : ''} · {c.category}
									</div>
								</div>
							</button>
						))}
						{pending.map((c) => (
							<div key={c.id} className="card ghostcard">
								<div className="thumb shimmer" />
								<div className="card-meta">
									<div className="title">İşleniyor</div>
									<div className="sub">puan {Math.round(c.score)}</div>
								</div>
							</div>
						))}
					</div>
				</section>

				{selClip && job && <Detail key={selClip.id} job={job} clip={selClip} onRerender={() => setBusy(true)} />}
			</main>
		</div>
	);
};

const Detail: React.FC<{job: Job; clip: Clip; onRerender: () => void}> = ({job, clip, onRerender}) => {
	const initial = (clip.info?.settings ?? {}) as Record<string, unknown>;
	const [st, setSt] = useState<Record<string, unknown>>({
		caption_style: initial.caption_style ?? 'dynamic',
		hook: initial.hook ?? true,
		sfx_density: initial.sfx_density ?? 2,
		label: initial.label ?? true,
		palette: initial.palette ?? null,
	});
	const [props, setProps] = useState<ClipProps | null>(null);
	const [err, setErr] = useState<string | null>(null);
	const [queued, setQueued] = useState(false);
	const reqId = useRef(0);
	const basePalette = job.streamer?.palette;

	useEffect(() => {
		const id = ++reqId.current;
		const t = setTimeout(() => {
			api<{props: ClipProps}>(`/api/jobs/${job.id}/clips/${clip.id}/props`, {method: 'POST', body: JSON.stringify({settings: st})})
				.then((r) => {
					if (id === reqId.current) {
						setProps(r.props);
						setErr(null);
					}
				})
				.catch((e) => setErr(String(e.message ?? e)));
		}, 250);
		return () => clearTimeout(t);
	}, [job.id, clip.id, st]);

	const pal = (st.palette as Palette | null) ?? basePalette;
	const setPal = (k: keyof Palette, v: string) => setSt((s) => ({...s, palette: {...(pal ?? {}), [k]: v}}));

	return (
		<aside className="detail" aria-label="Klip detayı">
			<div className="player">
				{props ? (
					<Player
						component={ClipComposition}
						inputProps={props}
						durationInFrames={props.durationInFrames}
						fps={props.fps}
						compositionWidth={1080}
						compositionHeight={1920}
						style={{width: '100%', aspectRatio: '9 / 16'}}
						controls
						loop
						acknowledgeRemotionLicense
					/>
				) : (
					<div className="player-wait shimmer" />
				)}
			</div>
			<div className="detail-body">
				<h2>{clip.titles?.[0]}</h2>
				<p className="why">{clip.why}</p>
				{clip.titles && clip.titles.length > 1 && (
					<ul className="alt-titles">
						{clip.titles.slice(1).map((t) => (
							<li key={t}>{t}</li>
						))}
					</ul>
				)}
				{clip.info?.hashtags && <p className="tags">{clip.info.hashtags.join(' ')}</p>}

				<div className="controls">
					<div className="ctl">
						<span className="ctl-label">Altyazı stili</span>
						<div className="seg">
							{STYLES.map(([k, l]) => (
								<button key={k} className={st.caption_style === k ? 'on' : ''} onClick={() => setSt((s) => ({...s, caption_style: k}))}>
									{l}
								</button>
							))}
						</div>
					</div>
					<div className="ctl">
						<span className="ctl-label">SFX yoğunluğu</span>
						<div className="seg">
							{[0, 1, 2, 3].map((n) => (
								<button key={n} className={st.sfx_density === n ? 'on' : ''} onClick={() => setSt((s) => ({...s, sfx_density: n}))}>
									{['Kapalı', 'Az', 'Normal', 'Yoğun'][n]}
								</button>
							))}
						</div>
					</div>
					<label className="toggle">
						<input type="checkbox" checked={Boolean(st.hook)} onChange={(e) => setSt((s) => ({...s, hook: e.target.checked}))} />
						<span>Hook (zirveyi başa al)</span>
					</label>
					<label className="toggle">
						<input type="checkbox" checked={Boolean(st.label)} onChange={(e) => setSt((s) => ({...s, label: e.target.checked}))} />
						<span>Yayıncı etiketi</span>
					</label>
					{pal && (
						<div className="ctl">
							<span className="ctl-label">Palet</span>
							<div className="palette-edit">
								{(['primary', 'accent', 'accent2', 'text', 'stroke'] as const).map((k) => (
									<label key={k} className="pe" title={k}>
										<input type="color" value={pal[k]} onChange={(e) => setPal(k, e.target.value)} aria-label={k} />
									</label>
								))}
								{st.palette ? (
									<button className="linkbtn" onClick={() => setSt((s) => ({...s, palette: null}))}>
										Sıfırla
									</button>
								) : null}
							</div>
						</div>
					)}
				</div>
				{err && <p className="err">{err}</p>}
				<div className="actions">
					<button
						className="go"
						disabled={queued}
						onClick={async () => {
							setQueued(true);
							onRerender();
							await api(`/api/jobs/${job.id}/clips/${clip.id}/rerender`, {method: 'POST', body: JSON.stringify({settings: st})});
							setTimeout(() => setQueued(false), 4000);
						}}
					>
						{queued ? 'Sıraya alındı' : 'Yeniden render et'}
					</button>
					<a className="ghost" href={`/api/jobs/${job.id}/clips/${clip.id}/download`}>
						İndir (.mp4)
					</a>
				</div>
				{clip.qa && (
					<p className="qa">
						Denetim: {clip.qa.passed ? 'geçti' : 'geçmedi'} · {clip.qa.rounds} tur
					</p>
				)}
			</div>
		</aside>
	);
};
