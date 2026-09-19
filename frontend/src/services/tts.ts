/**
 * Dedicated Text-to-Speech (TTS) Service Module
 * Handles Sarvam AI Bulbul V3 audio streaming and manual switching between
 * Browser Speech and Sarvam Studio Voice.
 */

export type TTSProvider = 'browser' | 'sarvam' | 'rumik';

const TTS_STORAGE_KEY = 'voice_agent_tts_provider';
const DEFAULT_BACKEND_URL =
	import.meta.env.VITE_BACKEND_URL ||
	(typeof window !== 'undefined' && window.location.origin
		? window.location.origin
		: 'http://localhost:8000');

let currentAudio: HTMLAudioElement | null = null;
let persistentAudioElement: HTMLAudioElement | null = null;
let currentObjectUrl: string | null = null;
let currentAbortController: AbortController | null = null;
let isAudioPlayingState = false;
let isLoudspeakerMuted = false;

/**
 * Convert base64 audio string to a Blob for immediate browser audio playback.
 * Strips whitespace, data URI headers, and sniffs magic bytes for accurate MIME type.
 */
export function b64ToBlob(b64Data: string, fallbackContentType = 'audio/wav'): Blob {
	let cleanB64 = (b64Data || '').trim();
	if (cleanB64.includes(',')) {
		cleanB64 = cleanB64.split(',')[1];
	}
	cleanB64 = cleanB64.replace(/\s+/g, '');

	const binary = atob(cleanB64);
	const bytes = new Uint8Array(binary.length);
	for (let i = 0; i < binary.length; i++) {
		bytes[i] = binary.charCodeAt(i);
	}

	let detectedType = fallbackContentType;
	if (bytes.length >= 4) {
		// RIFF (WAV)
		if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
			detectedType = 'audio/wav';
		} else if (
			// ID3 or MPEG frame header (MP3)
			(bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) ||
			(bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0)
		) {
			detectedType = 'audio/mpeg';
		}
	}

	return new Blob([bytes], { type: detectedType });
}

/**
 * Detect if client is running on a mobile device or Safari browser.
 */
export function isMobileOrSafari(): boolean {
	if (typeof window === 'undefined' || typeof navigator === 'undefined') return false;
	const ua = navigator.userAgent || '';
	const isIOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
	const isMobile = /Mobi|Android/i.test(ua);
	const isSafari = /^((?!chrome|android).)*safari/i.test(ua);
	return isIOS || isMobile || isSafari;
}

/**
 * Web Audio routing hook (kept for API compatibility without intrusive CORS-failing nodes).
 */
export function initTtsAudioRouting(_audioElement?: HTMLAudioElement): void {
	// Standard HTML5 Audio output to system device is preferred for reliability
}

/**
 * Control mute state of audio loudspeaker output
 */
export function setSpeakerMuted(muted: boolean): void {
	isLoudspeakerMuted = muted;
	if (persistentAudioElement) {
		persistentAudioElement.muted = muted;
	}
	if (currentAudio) {
		currentAudio.muted = muted;
	}
}

/**
 * Pre-unlock an HTML5 Audio element during user interaction gesture for iOS Safari compatibility.
 */
export function unlockAudioForMobile(): void {
	if (typeof window === 'undefined') return;
	try {
		if (!persistentAudioElement) {
			persistentAudioElement = new Audio();
			persistentAudioElement.preload = 'auto';
			persistentAudioElement.setAttribute('playsinline', 'true');
			persistentAudioElement.setAttribute('webkit-playsinline', 'true');
		}
		// Play silent wav buffer to permanently unlock audio playback on iOS Safari
		persistentAudioElement.src = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA';
		const p = persistentAudioElement.play();
		if (p && typeof p.then === 'function') {
			p.catch(() => {});
		}
	} catch (err) {
		console.warn('Audio unlock warning:', err);
	}
}

/**
 * Fetch the active TTS provider configured on the backend.
 * Always defaults to 'sarvam' studio voice.
 */
export async function fetchConfiguredTTSProvider(backendUrl?: string): Promise<TTSProvider> {
	try {
		const backend = backendUrl || DEFAULT_BACKEND_URL;
		const res = await fetch(`${backend}/audio/config`, {
			headers: { 'ngrok-skip-browser-warning': 'true' },
		});
		if (res.ok) {
			const data = await res.json();
			if (data.tts_provider === 'sarvam' || data.tts_provider === 'rumik') {
				return data.tts_provider;
			}
		}
	} catch (e) {
		console.warn('Could not fetch TTS provider from backend, defaulting to sarvam:', e);
	}
	return 'sarvam';
}

const VALID_RUMIK_EMOTION_TAGS = new Set([
	'laugh', 'laugh_harder', 'sigh', 'chuckle', 'gasp', 'angry',
	'excited', 'whisper', 'cry', 'scream', 'sing', 'snort',
	'exhale', 'gulp', 'giggle', 'sarcastic', 'curious',
]);

/**
 * Clean spoken text of internal markdown, tags, and formatting before speech synthesis.
 * Preserves Rumik Silk Mulberry inline emotion tags (<laugh>, <chuckle>, <curious>, etc.).
 */
export function cleanSpeechText(text: string): string {
	if (!text) return '';
	return text
		.replace(/<think>[\s\S]*?<\/think>/gi, '')
		.replace(/<think>[\s\S]*$/gi, '')
		.replace(/<([a-zA-Z0-9_]+)>/g, (_, tag) => {
			if (VALID_RUMIK_EMOTION_TAGS.has(tag.toLowerCase())) {
				return `<${tag.toLowerCase()}>`;
			}
			return '';
		})
		.replace(/<[^>]+>/g, '')
		.replace(/[*_#`~[\]]/g, '')
		.replace(/\|/g, ', ')
		.replace(/\bkitne\s+(?:baje|bje)\b/gi, 'kis time')
		.replace(/(\d{1,2}(?::\d{2})?)\s*(AM|PM)\s*(?:baje|bje)+\b/gi, '$1 $2')
		.replace(/(\d{1,2}(?::\d{2})?)\s*(?:baje|bje)+\b/gi, '$1')
		.replace(/\b(?:baje|bje)\b/gi, '')
		.replace(/\b(AM|PM)(?:\s+(?:AM|PM))+\b/gi, '$1')
		.replace(/\b(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*(?:AM|PM)+\b/gi, '$1')
		.replace(/\b(\d{1,2}(?::\d{2})?)\s*AM\s*(aur|ya|,|and|or)\s*(\d{1,2}(?::\d{2})?)\s*AM\b/gi, '$1 $2 $3 AM')
		.replace(/\b(\d{1,2}(?::\d{2})?)\s*PM\s*(aur|ya|,|and|or)\s*(\d{1,2}(?::\d{2})?)\s*PM\b/gi, '$1 $2 $3 PM')
		.replace(/\bsubah\s+(\d{1,2}(?::\d{2})?)\s*AM\b/gi, 'subah $1')
		.replace(/\bdopahar\s+(\d{1,2}(?::\d{2})?)\s*PM\b/gi, 'dopahar $1')
		.replace(/\bshaam\s+(\d{1,2}(?::\d{2})?)\s*PM\b/gi, 'shaam $1')
		.replace(/[ ]{2,}/g, ' ')
		.trim();
}

/**
 * Check if Sarvam audio is currently playing.
 */
export function isSarvamPlaying(): boolean {
	return isAudioPlayingState;
}

/**
 * Stop any active Sarvam audio playback and cancel ongoing HTTP requests.
 */
export function stopSarvamAudio(): void {
	if (currentAbortController) {
		try {
			currentAbortController.abort();
		} catch {}
		currentAbortController = null;
	}

	const audio = currentAudio || persistentAudioElement;
	if (audio) {
		// Crucial: Detach all event listeners BEFORE pausing to prevent spurious 'error' or 'interrupted' callbacks
		audio.onplay = null;
		audio.onended = null;
		audio.onerror = null;
		audio.onpause = null;
		try {
			audio.pause();
			audio.currentTime = 0;
		} catch {}
		try {
			audio.src = '';
		} catch {}
	}
	currentAudio = null;

	if (currentObjectUrl) {
		try {
			URL.revokeObjectURL(currentObjectUrl);
		} catch {}
		currentObjectUrl = null;
	}

	isAudioPlayingState = false;
}

export interface PlaySarvamOptions {
	backendUrl?: string;
	provider?: TTSProvider;
	speaker?: string;
	language?: string;
	pace?: number;
	model?: string;
	description?: string;
	audioB64?: string;
	onStart?: () => void;
	onEnd?: () => void;
	onError?: (err: any) => void;
}

export type PlayRemoteTTSOptions = PlaySarvamOptions;

/**
 * Request audio from the backend Sarvam TTS endpoint (or play direct audio_b64) and stream playback.
 * Automatically handles aborting previous playback, object URL cleanup,
 * and calling completion callbacks without CORS issues.
 */
export async function playSarvamAudio(
	text: string,
	options: PlaySarvamOptions = {}
): Promise<void> {
	const cleanText = cleanSpeechText(text);
	if (!cleanText) {
		if (options.onEnd) options.onEnd();
		return;
	}

	// Stop any previously playing audio cleanly before starting new utterance
	stopSarvamAudio();

	const controller = new AbortController();
	currentAbortController = controller;

	const backend = options.backendUrl || DEFAULT_BACKEND_URL;

	try {
		let blob: Blob;

		if (options.audioB64) {
			// Real Latency Optimization: Direct in-memory decode in <2ms, skipping HTTP fetch entirely!
			blob = b64ToBlob(options.audioB64);
		} else {
			const activeProv = options.provider || 'sarvam';
			const defaultSpeaker = activeProv === 'rumik' ? 'aisha' : 'simran';
			const res = await fetch(`${backend}/audio/tts`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'ngrok-skip-browser-warning': 'true',
				},
				body: JSON.stringify({
					text: cleanText,
					provider: activeProv,
					speaker: options.speaker || defaultSpeaker,
					language: options.language || 'hi-IN',
					pace: options.pace ?? 1.0,
					model: options.model,
					description: options.description,
				}),
				signal: controller.signal,
			});

			if (!res.ok) {
				const errDetail = await res.text();
				throw new Error(`${activeProv} TTS endpoint error (${res.status}): ${errDetail}`);
			}

			blob = await res.blob();
		}

		if (controller.signal.aborted) return;

		const objectUrl = URL.createObjectURL(blob);
		currentObjectUrl = objectUrl;

		// Use persistent audio element if pre-unlocked, or create new audio element
		if (!persistentAudioElement) {
			persistentAudioElement = new Audio();
			persistentAudioElement.preload = 'auto';
			persistentAudioElement.setAttribute('playsinline', 'true');
			persistentAudioElement.setAttribute('webkit-playsinline', 'true');
		}
		const audio = persistentAudioElement;
		currentAudio = audio;
		audio.muted = isLoudspeakerMuted;

		// Speech Speed: brisk 1.15 pace for Rumik Silk Aisha, 1.0 for Sarvam Simran
		const activeProv = options.provider || 'sarvam';
		const targetPace = options.pace ?? (activeProv === 'rumik' ? 1.15 : 1.0);
		try {
			audio.playbackRate = targetPace;
		} catch {}

		let finished = false;
		const cleanup = () => {
			if (finished) return;
			finished = true;
			isAudioPlayingState = false;
			audio.onplay = null;
			audio.onended = null;
			audio.onerror = null;
			if (currentObjectUrl === objectUrl) {
				try {
					URL.revokeObjectURL(objectUrl);
				} catch {}
				currentObjectUrl = null;
			}
			if (options.onEnd) {
				options.onEnd();
			}
		};

		const wordCount = cleanText.split(/\s+/).length;

		audio.onplay = () => {
			isAudioPlayingState = true;
			if (options.onStart) {
				options.onStart();
			}
			// Safety watchdog starts only AFTER audio begins playing and respects audio.duration
			const expectedMs =
				audio.duration && !isNaN(audio.duration) && isFinite(audio.duration)
					? audio.duration * 1000 + 3000
					: Math.max(8000, wordCount * 800 + 5000);

			setTimeout(() => {
				if (!finished && currentAudio === audio) {
					console.warn('Sarvam audio watchdog triggered cleanup');
					cleanup();
				}
			}, expectedMs);
		};

		audio.onended = () => {
			cleanup();
		};

		audio.onerror = (e) => {
			console.warn('Sarvam Audio playback notice:', e);
			cleanup();
			if (options.onError) {
				options.onError(e);
			}
		};

		audio.src = objectUrl;
		try {
			const playPromise = audio.play();
			if (playPromise && typeof playPromise.then === 'function') {
				await playPromise;
			}
		} catch (playErr: any) {
			if (playErr?.name === 'AbortError') {
				// Benign interruption when another utterance starts or user navigates
				return;
			}
			console.warn('Audio play notice:', playErr);
			cleanup();
			if (options.onError) {
				options.onError(playErr);
			}
		}
	} catch (err: any) {
		if (err?.name === 'AbortError') {
			// Expected cancellation when user clicks or interrupts
			return;
		}
		console.warn('Failed to synthesize/play Sarvam audio:', err);
		stopSarvamAudio();
		if (options.onError) {
			options.onError(err);
		} else if (options.onEnd) {
			options.onEnd();
		}
	}
}

export const playRemoteTTSAudio = playSarvamAudio;
export const stopRemoteTTSAudio = stopSarvamAudio;
