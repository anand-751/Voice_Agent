/**
 * Real-Time Linear16 16kHz PCM Audio Streamer for Sarvam STT WebSocket.
 * 
 * Captures microphone audio using Web Audio API, applies hardware echo cancellation,
 * converts Float32 audio to 16-bit linear PCM at 16,000Hz, and streams binary chunks
 * directly over the WebSocket connection with minimal latency.
 */

let audioContext: AudioContext | null = null;
let mediaStream: MediaStream | null = null;
let scriptProcessor: ScriptProcessorNode | null = null;
let sourceNode: MediaStreamAudioSourceNode | null = null;
let muteGainNode: GainNode | null = null;
let activeWs: WebSocket | null = null;
let isStreamingActive = false;
let agentSpeakingGetter: (() => boolean) | null = null;

/**
 * Register a getter that returns true if the agent is actively speaking audio.
 * When true, audio streaming is paused to eliminate loudspeaker bleed and mid-sentence cutoffs.
 */
export function setAgentSpeakingGetter(fn: () => boolean): void {
	agentSpeakingGetter = fn;
}

/**
 * Resample Float32 audio buffer from source sample rate to 16,000 Hz.
 */
function downsampleTo16k(input: Float32Array, sourceSampleRate: number): Float32Array {
	if (sourceSampleRate === 16000) {
		return input;
	}
	const ratio = sourceSampleRate / 16000;
	const newLength = Math.round(input.length / ratio);
	const result = new Float32Array(newLength);
	let offsetResult = 0;
	let offsetInput = 0;

	while (offsetResult < result.length) {
		const nextOffsetInput = Math.round((offsetResult + 1) * ratio);
		let accum = 0;
		let count = 0;
		for (let i = offsetInput; i < nextOffsetInput && i < input.length; i++) {
			accum += input[i];
			count++;
		}
		result[offsetResult] = count > 0 ? accum / count : 0;
		offsetResult++;
		offsetInput = nextOffsetInput;
	}
	return result;
}

/**
 * Convert normalized Float32 samples (-1.0 to 1.0) into 16-bit signed PCM (Int16Array).
 */
function floatTo16BitPCM(input: Float32Array): Int16Array {
	const output = new Int16Array(input.length);
	for (let i = 0; i < input.length; i++) {
		const s = Math.max(-1, Math.min(1, input[i]));
		output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
	}
	return output;
}

export interface AudioStreamerOptions {
	onAudioLevel?: (level: number) => void;
	onError?: (err: any) => void;
}

/**
 * Start streaming microphone audio over the given WebSocket connection.
 */
export async function startAudioStreaming(
	ws: WebSocket,
	options?: AudioStreamerOptions,
	existingStream?: MediaStream
): Promise<MediaStream> {
	stopAudioStreaming();

	if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
		throw new Error('Microphone streaming not supported on this browser.');
	}

	activeWs = ws;

	let stream: MediaStream;
	if (existingStream && existingStream.getAudioTracks().some((t) => t.readyState === 'live')) {
		stream = existingStream;
	} else {
		stream = await navigator.mediaDevices.getUserMedia({
			audio: {
				echoCancellation: true,
				noiseSuppression: true,
				autoGainControl: true,
				channelCount: 1,
			},
		});
	}

	mediaStream = stream;

	const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
	try {
		// Prefer 16000Hz directly
		audioContext = new AudioContextClass({ sampleRate: 16000 });
	} catch {
		// Fallback to default sample rate
		audioContext = new AudioContextClass();
	}

	if (audioContext.state === 'suspended') {
		await audioContext.resume();
	}

	sourceNode = audioContext.createMediaStreamSource(stream);

	// 85Hz High-Pass Filter: filters out 50Hz/60Hz electrical hum, fan rumble, and low-frequency handling noise
	const highPassFilter = audioContext.createBiquadFilter();
	highPassFilter.type = 'highpass';
	highPassFilter.frequency.value = 85;

	// Buffer size 2048 gives ~46ms at 44.1k or ~128ms at 16k
	const bufferSize = 2048;
	scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);

	scriptProcessor.onaudioprocess = (e) => {
		if (!isStreamingActive || !activeWs || activeWs.readyState !== WebSocket.OPEN) {
			return;
		}

		const inputData = e.inputBuffer.getChannelData(0);

		// Calculate audio energy level for visualizer
		let sum = 0;
		for (let i = 0; i < inputData.length; i++) {
			sum += inputData[i] * inputData[i];
		}
		const rms = Math.sqrt(sum / inputData.length);

		// Half-Duplex Speaking Gate:
		// When Niaa is speaking, completely pause streaming mic chunks to Sarvam STT
		// and reset mic audio level to 0 to prevent visualizer bouncing.
		if (agentSpeakingGetter && agentSpeakingGetter()) {
			if (options?.onAudioLevel) {
				options.onAudioLevel(0);
			}
			return;
		}

		if (options?.onAudioLevel) {
			options.onAudioLevel(Math.min(1, rms * 5));
		}

		// Resample to 16kHz if needed
		const currentSampleRate = audioContext?.sampleRate || 16000;
		const resampled = downsampleTo16k(inputData, currentSampleRate);
		const pcm16 = floatTo16BitPCM(resampled);

		try {
			// Stream raw binary ArrayBuffer over WebSocket
			activeWs.send(pcm16.buffer);
		} catch (err) {
			console.warn('Error sending PCM audio chunk over WebSocket:', err);
		}
	};

	// Use a 0-gain mute node to keep ScriptProcessor active in Web Audio
	// without routing raw microphone audio straight back out into the speakers (which causes loopback feedback)
	muteGainNode = audioContext.createGain();
	muteGainNode.gain.value = 0;

	sourceNode.connect(highPassFilter);
	highPassFilter.connect(scriptProcessor);
	scriptProcessor.connect(muteGainNode);
	muteGainNode.connect(audioContext.destination);

	isStreamingActive = true;
	return stream;
}

/**
 * Stop active microphone audio streaming and release Web Audio resources.
 */
export function stopAudioStreaming(): void {
	isStreamingActive = false;
	activeWs = null;

	if (scriptProcessor) {
		try {
			scriptProcessor.disconnect();
		} catch { }
		scriptProcessor = null;
	}

	if (muteGainNode) {
		try {
			muteGainNode.disconnect();
		} catch { }
		muteGainNode = null;
	}

	if (sourceNode) {
		try {
			sourceNode.disconnect();
		} catch { }
		sourceNode = null;
	}

	if (audioContext) {
		try {
			audioContext.close();
		} catch { }
		audioContext = null;
	}

	if (mediaStream) {
		try {
			mediaStream.getTracks().forEach((track) => track.stop());
		} catch { }
		mediaStream = null;
	}
}

/**
 * Check if the audio streamer is currently capturing and streaming.
 */
export function isAudioStreaming(): boolean {
	return isStreamingActive;
}
