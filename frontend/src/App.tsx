import { useEffect, useRef, useState } from 'react';
import {
	Mic,
	MicOff,
	Phone,
	PhoneOff,
	Send,
	Loader2,
	ShieldCheck,
	AlertCircle,
	Clock,
	Users,
	Sparkles,
	Volume2,
	Calendar,
	ExternalLink,
} from 'lucide-react';
import {
	fetchConfiguredTTSProvider,
	playSarvamAudio,
	stopSarvamAudio,
	unlockAudioForMobile,
	cleanSpeechText,
	isMobileOrSafari,
	setSpeakerMuted,
	type TTSProvider,
} from './services/tts';
import {
	startAudioStreaming,
	stopAudioStreaming,
	isAudioStreaming,
	setAgentSpeakingGetter,
} from './services/audioStreamer';
import CallScreen from './components/CallScreen';

type Page = 'FORM' | 'INTRO' | 'CALL' | 'QUEUED' | 'PAYMENT_SUCCESS' | 'PAYMENT_CANCEL';

interface Message {
	role: 'user' | 'assistant';
	content: string;
}

interface Profile {
	name: string;
	phone: string;
	email: string;
}

interface BookingSummary {
	id?: string;
	date: string;
	time: string;
	name?: string;
	doctor?: string;
	calendar_url?: string;
}

const BACKEND_URL =
	import.meta.env.VITE_BACKEND_URL ||
	(typeof window !== 'undefined' && window.location.origin
		? window.location.origin
		: 'http://localhost:8000');
const DEFAULT_CALL_DURATION = 180; // 180 seconds (3 minutes) limit

function App() {
	const [page, setPage] = useState<Page>(() => {
		if (typeof window !== 'undefined') {
			const params = new URLSearchParams(window.location.search);
			if (params.get('preview') === 'call' || params.get('call') === 'true') {
				return 'CALL';
			}
		}
		return 'FORM';
	});
	const [profile, setProfile] = useState<Profile>(() => {
		try {
			const stored = localStorage.getItem('userProfile');
			if (stored) return JSON.parse(stored);
		} catch {}
		return { name: '', phone: '', email: '' };
	});
	const [messages, setMessages] = useState<Message[]>(() => {
		if (typeof window !== 'undefined') {
			const params = new URLSearchParams(window.location.search);
			if (params.get('preview') === 'call' || params.get('call') === 'true') {
				return [
					{ role: 'assistant', content: 'Namaste! Bright Dental Clinic mein aapka swagat hai. Main Niaa bol rahi hoon, main aapki kaise madad kar sakti hoon?' },
					{ role: 'user', content: 'How can I help you today?' },
				];
			}
		}
		return [];
	});
	const [input, setInput] = useState('');
	const [status, setStatus] = useState('idle');
	const [error, setError] = useState('');
	const [micHint, setMicHint] = useState('');
	const [isListening, setIsListening] = useState(false);
	const [isTranscribing, setIsTranscribing] = useState(false);
	const [checkoutUrl, setCheckoutUrl] = useState<string | null>(null);
	const [pendingBookingId, setPendingBookingId] = useState<string | null>(null);
	const [confirmedBooking, setConfirmedBooking] = useState<BookingSummary | null>(null);
	const [lastConfirmedBooking, setLastConfirmedBooking] = useState<BookingSummary | null>(null);

	// 180s Timer & Concurrency Queue State
	const [timeLeft, setTimeLeft] = useState<number>(DEFAULT_CALL_DURATION);
	const [queuePosition, setQueuePosition] = useState<number>(1);
	const [queueMessage, setQueueMessage] = useState<string>('Agents are busy kindly wait');
	const [continuousMode, setContinuousMode] = useState<boolean>(true);
	const [ttsProvider, setTtsProvider] = useState<TTSProvider>('sarvam');
	const ttsProviderRef = useRef<TTSProvider>('sarvam');
	const [sttProvider, setSttProvider] = useState<'sarvam' | 'browser'>('sarvam');
	const sttProviderRef = useRef<'sarvam' | 'browser'>('sarvam');
	const [isNiaaSpeaking, setIsNiaaSpeaking] = useState<boolean>(false);
	const [isNiaaThinking, setIsNiaaThinking] = useState<boolean>(false);
	const [callElapsedSeconds, setCallElapsedSeconds] = useState<number>(() =>
		typeof window !== 'undefined' &&
		(new URLSearchParams(window.location.search).get('preview') === 'call' ||
			new URLSearchParams(window.location.search).get('call') === 'true')
			? 28
			: 0
	);
	const [audioLevel, setAudioLevel] = useState<number>(0);
	const [isMuted, setIsMuted] = useState<boolean>(false);
	const isMutedRef = useRef<boolean>(false);
	const [isSpeakerMuted, setIsSpeakerMuted] = useState<boolean>(false);

	const toggleMute = () => {
		const nextMuted = !isMuted;
		setIsMuted(nextMuted);
		isMutedRef.current = nextMuted;
		if (mediaStreamRef.current) {
			mediaStreamRef.current.getAudioTracks().forEach((track) => {
				track.enabled = !nextMuted;
			});
		}
	};

	const toggleSpeaker = () => {
		const nextSpeakerMuted = !isSpeakerMuted;
		setIsSpeakerMuted(nextSpeakerMuted);
		setSpeakerMuted(nextSpeakerMuted);
	};

	// Audio volume reactivity when Niaa is speaking
	useEffect(() => {
		if (!isNiaaSpeaking) {
			if (!isListening || isMuted) {
				setAudioLevel(0);
			}
			return;
		}
		let animId: number;
		let tick = 0;
		const updateLevel = () => {
			tick++;
			const base = 0.35 + 0.3 * Math.sin(tick * 0.25);
			const jitter = Math.random() * 0.35;
			setAudioLevel(Math.min(1, Math.max(0.1, base + jitter)));
			animId = requestAnimationFrame(updateLevel);
		};
		animId = requestAnimationFrame(updateLevel);
		return () => {
			cancelAnimationFrame(animId);
			setAudioLevel(0);
		};
	}, [isNiaaSpeaking, isListening, isMuted]);

	// Synchronize active voice & STT engines strictly with backend config
	useEffect(() => {
		setAgentSpeakingGetter(() => isSpeakingRef.current);
		fetchConfiguredTTSProvider(BACKEND_URL).then((prov) => {
			setTtsProvider(prov);
			ttsProviderRef.current = prov;
		});
		fetch(`${BACKEND_URL}/audio/config`, {
			headers: { 'ngrok-skip-browser-warning': 'true' },
		})
			.then((res) => res.json())
			.then((data) => {
				if (data.stt_provider) {
					setSttProvider(data.stt_provider);
					sttProviderRef.current = data.stt_provider;
				}
			})
			.catch(() => {});
	}, []);

	const socketRef = useRef<WebSocket | null>(null);

	const cycleTtsProvider = () => {
		const next: TTSProvider = ttsProvider === 'sarvam' ? 'rumik' : 'sarvam';
		setTtsProvider(next);
		ttsProviderRef.current = next;
		if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
			try {
				socketRef.current.send(JSON.stringify({ type: 'set_tts_provider', provider: next }));
			} catch {}
		}
	};
	const timerRef = useRef<any>(null);
	const confirmationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const recognitionRef = useRef<any>(null);
	const mediaStreamRef = useRef<MediaStream | null>(null);
	const mediaRecorderRef = useRef<MediaRecorder | null>(null);
	const audioChunksRef = useRef<Blob[]>([]);
	const paymentPendingRef = useRef(false);
	const continuousModeRef = useRef<boolean>(true);
	const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const currentTranscriptRef = useRef<string>('');
	const isSpeakingRef = useRef<boolean>(false);
	const callTimeExpiredPendingRef = useRef<boolean>(false);
	const isTurnInFlightRef = useRef<boolean>(false);
	const pageRef = useRef<Page>('FORM');
	const voicesRef = useRef<SpeechSynthesisVoice[]>([]);
	const activeUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
	const heartbeatRef = useRef<any>(null);

	useEffect(() => {
		const loadVoices = () => {
			if ('speechSynthesis' in window) {
				const v = window.speechSynthesis.getVoices();
				if (v && v.length > 0) {
					voicesRef.current = v;
				}
			}
		};
		loadVoices();
		if ('speechSynthesis' in window) {
			window.speechSynthesis.onvoiceschanged = loadVoices;
		}
	}, []);

	useEffect(() => {
		continuousModeRef.current = continuousMode;
	}, [continuousMode]);

	useEffect(() => {
		pageRef.current = page;
	}, [page]);

	useEffect(() => {
		const stored = localStorage.getItem('userProfile');
		if (stored) {
			try {
				setProfile(JSON.parse(stored));
				setPage('INTRO');
			} catch {
				localStorage.removeItem('userProfile');
			}
		}
		return () => {
			stopListening();
			clearInterval(timerRef.current);
			if (heartbeatRef.current) {
				clearInterval(heartbeatRef.current);
				heartbeatRef.current = null;
			}
			socketRef.current?.close();
		};
	}, []);

	useEffect(() => {
		const path = window.location.pathname;
		if (path === '/payment-success') setPage('PAYMENT_SUCCESS');
		if (path === '/payment-cancel') setPage('PAYMENT_CANCEL');
	}, []);

	const formatTime = (secs: number) => {
		const m = Math.floor(secs / 60);
		const s = secs % 60;
		return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
	};

	const startCallTimer = (duration: number = DEFAULT_CALL_DURATION) => {
		setTimeLeft(duration);
		setCallElapsedSeconds(0);
		callTimeExpiredPendingRef.current = false;
		isTurnInFlightRef.current = false;
		if (timerRef.current) clearInterval(timerRef.current);

		const startTime = Date.now();
		timerRef.current = setInterval(() => {
			if (paymentPendingRef.current) return;
			const elapsed = Math.floor((Date.now() - startTime) / 1000);
			const remaining = Math.max(0, duration - elapsed);
			setCallElapsedSeconds(elapsed);
			setTimeLeft(remaining);

			if (remaining <= 0) {
				clearInterval(timerRef.current);
				timerRef.current = null;
				// If agent is speaking, or a response is being generated, let speech complete gracefully
				if (isSpeakingRef.current || isTurnInFlightRef.current || status === 'processing') {
					console.log('Call timer reached 0 while speech or response turn is in-flight. Gracefully awaiting speech completion before ending call.');
					callTimeExpiredPendingRef.current = true;
					stopListening();
					setMicHint('Call time completed • Finishing receptionist answer...');
				} else {
					handleCallTimeExpired();
				}
			}
		}, 1000);
	};

	const handleCallTimeExpired = () => {
		stopListening();
		callTimeExpiredPendingRef.current = false;
		isTurnInFlightRef.current = false;
		if (timerRef.current) {
			clearInterval(timerRef.current);
			timerRef.current = null;
		}
		const timeoutMsg = 'Your 3-minute call session has ended. Thank you for contacting Bright Dental Clinic!';
		setMessages((current) => [...current, { role: 'assistant', content: timeoutMsg }]);
		setStatus('ended');

		// Automatically end call ONLY after speech completely finishes
		let ended = false;
		const autoEnd = () => {
			if (!ended) {
				ended = true;
				endCall();
			}
		};

		speak(timeoutMsg, () => {
			setTimeout(autoEnd, 1200);
		});
		// Generous safety fallback in case speech API hangs: 25 seconds
		setTimeout(autoEnd, 25000);
	};

	const getFemaleVoice = (): SpeechSynthesisVoice | undefined => {
		const voices =
			voicesRef.current.length > 0
				? voicesRef.current
				: 'speechSynthesis' in window
					? window.speechSynthesis.getVoices()
					: [];

		if (!voices || voices.length === 0) return undefined;

		// Prioritize locally available voices to avoid un-downloaded network voices on macOS
		const localVoices = voices.filter((v) => v.localService !== false);
		const candidatePool = localVoices.length > 0 ? localVoices : voices;

		const preferredFemaleNames = [
			'Samantha',
			'Victoria',
			'Karen',
			'Tessa',
			'Google US English',
			'Google UK English Female',
			'Moira',
			'Veena',
			'Fiona',
			'Ava',
			'Allison',
			'Microsoft Zira',
			'Microsoft Jenny',
		];

		// 1. Explicit female voice name match
		for (const name of preferredFemaleNames) {
			const found = candidatePool.find((v) => v.name.toLowerCase().includes(name.toLowerCase()));
			if (found) return found;
		}

		// 2. Voices explicitly containing female or woman in name
		const taggedFemale = candidatePool.find((v) => /female|woman/i.test(v.name));
		if (taggedFemale) return taggedFemale;

		// 3. Any English voice that is NOT an explicitly known male voice
		const knownMaleKeywords = [
			'alex', 'fred', 'daniel', 'oliver', 'rishi', 'george', 'david',
			'mark', 'tom', 'ralph', 'bruce', 'junior', 'albert', 'male'
		];
		const nonMaleEn = candidatePool.find(
			(v) => /^en(-|_)/i.test(v.lang) && !knownMaleKeywords.some((m) => v.name.toLowerCase().includes(m))
		);
		if (nonMaleEn) return nonMaleEn;

		return candidatePool.find((v) => /^en(-|_)/i.test(v.lang)) || voices.find((v) => v.default) || voices[0];
	};

	const executeSpeak = (cleanText: string, onEndCallback?: () => void, onStartCallback?: () => void) => {
		try {
			if (window.speechSynthesis.paused) {
				window.speechSynthesis.resume();
			}

			const utterance = new SpeechSynthesisUtterance(cleanText);
			activeUtteranceRef.current = utterance;
			(window as any).__activeUtterance = utterance;

			const voice = getFemaleVoice();
			if (voice) {
				utterance.voice = voice;
				utterance.lang = voice.lang;
			} else {
				utterance.lang = 'en-US';
			}

			utterance.rate = 1.0;
			utterance.pitch = 1.0;
			utterance.volume = 1.0;

			let finished = false;
			const handleDone = () => {
				if (finished) return;
				finished = true;
				isSpeakingRef.current = false;
				setIsNiaaSpeaking(false);
				activeUtteranceRef.current = null;
				(window as any).__activeUtterance = null;
				if (onEndCallback) {
					onEndCallback();
					return;
				}
				// In continuous hands-free mode, re-arm microphone immediately as soon as speech completes
				if (
					continuousModeRef.current &&
					pageRef.current === 'CALL' &&
					socketRef.current &&
					socketRef.current.readyState === WebSocket.OPEN
				) {
					setIsListening(true);
					startListening();
				}
			};

			utterance.onstart = () => {
				isSpeakingRef.current = true;
				setIsNiaaSpeaking(true);
				setIsNiaaThinking(false);
				if (mediaStreamRef.current) {
					mediaStreamRef.current.getAudioTracks().forEach((track) => {
						track.enabled = false;
					});
				}
				setIsListening(false);
				setAudioLevel(0);
				if (onStartCallback) onStartCallback();
				stopListening();
			};

			utterance.onend = () => {
				handleDone();
			};

			utterance.onerror = (e) => {
				if (e.error !== 'canceled' && e.error !== 'interrupted') {
					console.warn('Speech synthesis utterance error:', e);
				}
				handleDone();
			};

			// Safety watchdog: if the browser drops onend or fails to start, never leave mic blocked
			const wordCount = cleanText.split(/\s+/).length;
			const maxExpectedMs = Math.max(3000, wordCount * 500 + 2000);
			setTimeout(() => {
				if (!finished) {
					handleDone();
				}
			}, maxExpectedMs);

			window.speechSynthesis.speak(utterance);
		} catch (err) {
			console.warn('speechSynthesis.speak error:', err);
			setIsNiaaSpeaking(false);
			if (onEndCallback) onEndCallback();
		}
	};

	const speak = (
		text: string,
		onEndCallback?: () => void,
		audioB64?: string,
		onStartCallback?: () => void,
		providerOverride?: TTSProvider
	) => {
		const cleanText = cleanSpeechText(text);

		if (!cleanText) {
			if (onEndCallback) onEndCallback();
			return;
		}

		const rawProv = providerOverride || ttsProviderRef.current || 'sarvam';
		const activeProv = rawProv === 'browser' && !audioB64 ? 'browser' : (rawProv === 'rumik' ? 'rumik' : 'sarvam');

		// 1. Remote Indic Studio Voice Mode (Sarvam Bulbul v3 or Rumik Silk)
		// If audioB64 is available OR provider is sarvam/rumik, ALWAYS use remote studio audio!
		if (audioB64 || activeProv === 'sarvam' || activeProv === 'rumik') {
			if ('speechSynthesis' in window && window.speechSynthesis.speaking) {
				try {
					window.speechSynthesis.cancel();
				} catch { }
			}

			const provToUse = activeProv === 'browser' ? 'sarvam' : activeProv;
			playSarvamAudio(cleanText, {
				backendUrl: BACKEND_URL,
				provider: provToUse,
				speaker: provToUse === 'rumik' ? 'aisha' : 'simran',
				language: 'hi-IN',
				pace: provToUse === 'rumik' ? 1.15 : 1.0,
				audioB64: audioB64,
				onStart: () => {
					isSpeakingRef.current = true;
					setIsNiaaSpeaking(true);
					setIsNiaaThinking(false);
					// Turn off user microphone while AI is speaking
					if (mediaStreamRef.current) {
						mediaStreamRef.current.getAudioTracks().forEach((track) => {
							track.enabled = false;
						});
					}
					setIsListening(false);
					setAudioLevel(0);
					if (onStartCallback) onStartCallback();
					if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
						try {
							socketRef.current.send(JSON.stringify({ type: 'agent_speaking', speaking: true }));
						} catch {}
					}
					if (sttProviderRef.current !== 'sarvam') {
						stopListening();
					}
				},
				onEnd: () => {
					isSpeakingRef.current = false;
					setIsNiaaSpeaking(false);
					if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
						try {
							socketRef.current.send(JSON.stringify({ type: 'agent_speaking', speaking: false }));
						} catch {}
					}
					// Turn user microphone back on after AI finishes speaking
					if (!isMutedRef.current && mediaStreamRef.current) {
						mediaStreamRef.current.getAudioTracks().forEach((track) => {
							track.enabled = true;
						});
					}
					if (!isMutedRef.current) {
						setIsListening(true);
					}
					if (onEndCallback) {
						onEndCallback();
						return;
					}
					// In continuous hands-free mode, re-arm microphone immediately as soon as speech completes
					if (
						continuousModeRef.current &&
						pageRef.current === 'CALL' &&
						socketRef.current &&
						socketRef.current.readyState === WebSocket.OPEN
					) {
						startListening();
					}
				},
				onError: (err) => {
					console.warn(`${provToUse} TTS audio playback notice:`, err);
					isSpeakingRef.current = false;
					setIsNiaaSpeaking(false);
					if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
						try {
							socketRef.current.send(JSON.stringify({ type: 'agent_speaking', speaking: false }));
						} catch {}
					}
					if (!isMutedRef.current && mediaStreamRef.current) {
						mediaStreamRef.current.getAudioTracks().forEach((track) => {
							track.enabled = true;
						});
					}
					if (!isMutedRef.current) {
						setIsListening(true);
					}
					// CRITICAL: NEVER trigger robotic English browser speech during studio calls!
					// Keep turn state clean and proceed with normal conversation flow.
					if (onEndCallback) {
						onEndCallback();
					}
				},
			});
			return;
		}

		// 2. Browser Voice Mode (Free / Local Dev)
		stopSarvamAudio();

		if (!('speechSynthesis' in window)) {
			if (onEndCallback) setTimeout(onEndCallback, 1000);
			return;
		}

		// If the synthesizer is actively speaking, cancel and let the queue settle briefly
		if (window.speechSynthesis.speaking) {
			try {
				window.speechSynthesis.cancel();
			} catch { }
			setTimeout(() => {
				executeSpeak(cleanText, onEndCallback, onStartCallback);
			}, 50);
			return;
		}

		// When not speaking, speak immediately without calling cancel()
		executeSpeak(cleanText, onEndCallback, onStartCallback);
	};

	const sendMessage = (value = input) => {
		const text = value.trim();
		const socket = socketRef.current;
		if (!text || !socket || socket.readyState !== WebSocket.OPEN) return;

		let activeProfile = profile;
		if (!activeProfile?.name) {
			try {
				const stored = localStorage.getItem('userProfile');
				if (stored) activeProfile = JSON.parse(stored);
			} catch {}
		}

		socket.send(JSON.stringify({ user_input: text, user_profile: activeProfile, language: 'en', tts_provider: ttsProviderRef.current }));
		setMessages((current) => [...current, { role: 'user', content: text }]);
		setInput('');
		setStatus('processing');
		isTurnInFlightRef.current = true;
	};

	const openPaymentAndConfirm = () => {
		const targetUrl = checkoutUrl || (pendingBookingId ? `${BACKEND_URL}/payment-success?mock=1&booking=${pendingBookingId}` : null);
		if (!targetUrl) return;
		window.open(targetUrl, '_blank', 'noopener,noreferrer');
		paymentPendingRef.current = true;
		const bookingId = pendingBookingId;
		if (!bookingId) {
			setError('The booking reference was lost. Please select the slot again.');
			return;
		}
		setStatus('processing');
		// Finalize payment in backend and automatically trigger 'paid' verification
		fetch(`${BACKEND_URL}/api/dev/simulate-payment/${bookingId}`, {
			method: 'POST',
			headers: { 'ngrok-skip-browser-warning': 'true' },
		})
			.then((response) => {
				if (!response.ok) throw new Error('payment verification failed');
				return response.json();
			})
			.then(() => {
				// Automatically trigger the paid message once user clicks open payment
				sendMessage('paid');
			})
			.catch(() => {
				setStatus('error');
				setError('Payment verification failed. Please try clicking the button again.');
			});
	};

	const stopListening = () => {
		stopAudioStreaming();
		if (silenceTimerRef.current) {
			clearTimeout(silenceTimerRef.current);
			silenceTimerRef.current = null;
		}
		currentTranscriptRef.current = '';
		if (recognitionRef.current) {
			try {
				recognitionRef.current.abort();
			} catch { }
			recognitionRef.current = null;
		}
		if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
			try {
				mediaRecorderRef.current.stop();
			} catch { }
			mediaRecorderRef.current = null;
		}
		if (mediaStreamRef.current) {
			mediaStreamRef.current.getTracks().forEach((track) => track.stop());
			mediaStreamRef.current = null;
		}
		setIsListening(false);
	};

	const startMediaRecorder = (stream: MediaStream) => {
		try {
			audioChunksRef.current = [];
			const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
				? 'audio/webm;codecs=opus'
				: MediaRecorder.isTypeSupported('audio/webm')
					? 'audio/webm'
					: MediaRecorder.isTypeSupported('audio/mp4')
						? 'audio/mp4'
						: '';

			const options = mimeType ? { mimeType } : undefined;
			const recorder = new MediaRecorder(stream, options);

			recorder.ondataavailable = (event) => {
				if (event.data && event.data.size > 0) {
					audioChunksRef.current.push(event.data);
				}
			};

			recorder.onstop = async () => {
				setIsListening(false);
				setMicHint('');
				stream.getTracks().forEach((t) => t.stop());
				mediaStreamRef.current = null;

				const blobType = recorder.mimeType || 'audio/webm';
				const audioBlob = new Blob(audioChunksRef.current, { type: blobType });
				if (audioBlob.size < 500) return;

				setIsTranscribing(true);
				try {
					const res = await fetch(`${BACKEND_URL}/audio/transcribe`, {
						method: 'POST',
						headers: {
							'Content-Type': blobType,
							'ngrok-skip-browser-warning': 'true',
						},
						body: audioBlob,
					});
					if (res.ok) {
						const data = await res.json();
						if (data.text && data.text.trim()) {
							sendMessage(data.text.trim());
						} else {
							setMicHint('No speech detected in audio.');
						}
					} else {
						setError('Audio transcription failed. You can type instead.');
					}
				} catch {
					setError('Failed to connect to audio transcription service.');
				} finally {
					setIsTranscribing(false);
				}
			};

			mediaRecorderRef.current = recorder;
			recorder.start();
			setIsListening(true);
			setError('');
			setMicHint('Recording audio... Click mic to send');
		} catch (err) {
			console.error('MediaRecorder error:', err);
			setError('Could not start audio recorder. You can type your message.');
			setIsListening(false);
			stream.getTracks().forEach((t) => t.stop());
		}
	};

	const startListening = async () => {
		if (isSpeakingRef.current && sttProviderRef.current !== 'sarvam') return;
		setError('');
		setMicHint('');

		// Check for insecure HTTP context (Safari & mobile browsers strictly block microphone on plain HTTP)
		if (typeof window !== 'undefined' && window.isSecureContext === false) {
			setIsListening(false);
			setMicHint('');
			setError('Microphone requires HTTPS on Safari / mobile. Please open via the secure HTTPS link (https://...ngrok-free.dev) instead of http://.');
			return;
		}

		if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
			setIsListening(false);
			setMicHint('');
			const isIOS = typeof navigator !== 'undefined' && (/iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1));
			if (isIOS) {
				setError('Microphone access is unavailable. If you opened this link inside an app (e.g. WhatsApp or Telegram), tap the Safari compass icon in the corner to open in Safari.');
			} else {
				setError('Microphone access is not supported by your browser. You can type instead.');
			}
			return;
		}

		// 1. Sarvam Realtime STT Audio Streaming (Server-side VAD & Barge-in)
		if (
			sttProviderRef.current === 'sarvam' &&
			socketRef.current &&
			socketRef.current.readyState === WebSocket.OPEN
		) {
			if (isAudioStreaming()) {
				setIsListening(true);
				setMicHint('Sarvam Realtime Voice Active • Speak naturally');
				return;
			}
			try {
				await startAudioStreaming(socketRef.current, {
					onError: (err) => {
						console.warn('Sarvam streaming error, falling back to Web Speech:', err);
						sttProviderRef.current = 'browser';
						startListening();
					},
					onAudioLevel: (lvl) => {
						if (!isMutedRef.current && !isSpeakingRef.current) {
							setAudioLevel(lvl);
						}
					},
				}, mediaStreamRef.current || undefined);
				setIsListening(true);
				setMicHint('Sarvam Realtime Voice Active • Speak naturally');
				return;
			} catch (err: any) {
				console.warn('Microphone streaming access error:', err);
				if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
					setIsListening(false);
					setMicHint('');
					setError('Microphone permission blocked. In Safari, tap "aA" (or settings icon) in the address bar -> Website Settings -> Microphone -> Allow.');
					return;
				}
				sttProviderRef.current = 'browser';
			}
		}

		let stream: MediaStream;
		try {
			if (mediaStreamRef.current && mediaStreamRef.current.getAudioTracks().some((t) => t.readyState === 'live')) {
				stream = mediaStreamRef.current;
			} else {
				stream = await navigator.mediaDevices.getUserMedia({
					audio: {
						echoCancellation: true,
						noiseSuppression: true,
						autoGainControl: true,
					},
				});
				mediaStreamRef.current = stream;
			}
		} catch (err: any) {
			console.warn('getUserMedia error:', err);
			setIsListening(false);
			setMicHint('');
			if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
				setError('Microphone permission blocked. In Safari, tap "aA" in the address bar -> Website Settings -> Microphone -> Allow.');
			} else {
				setError('Could not access microphone. Please check your audio input settings.');
			}
			return;
		}

		// Try Web Speech API first
		const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
		if (SpeechRecognition) {
			try {
				const recognition = new SpeechRecognition();
				recognition.lang = 'en-IN';
				recognition.continuous = true;
				recognition.interimResults = true;

				recognition.onstart = () => {
					setIsListening(true);
					setError('');
					setMicHint('Listening... Speak freely');
				};

				recognition.onresult = (event: any) => {
					let fullTranscript = '';
					for (let i = 0; i < event.results.length; i++) {
						fullTranscript += event.results[i][0].transcript;
					}
					fullTranscript = fullTranscript.trim();
					if (!fullTranscript) return;

					currentTranscriptRef.current = fullTranscript;
					setMicHint(`"${fullTranscript}"`);

					// Reset 0.3-second silence buffer timer
					if (silenceTimerRef.current) {
						clearTimeout(silenceTimerRef.current);
					}

					silenceTimerRef.current = setTimeout(() => {
						const finalQuery = currentTranscriptRef.current.trim();
						if (finalQuery) {
							sendMessage(finalQuery);
							currentTranscriptRef.current = '';
							setMicHint('');
							stopListening();
						}
					}, 300); // 0.3-second silence buffer before auto-dispatching query
				};

				recognition.onerror = (event: any) => {
					console.warn('SpeechRecognition error:', event.error);
					if (event.error === 'no-speech') {
						if (!currentTranscriptRef.current.trim()) {
							setMicHint('No speech detected. Speak or click mic.');
							stopListening();
						}
						return;
					}
					if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
						setError('Microphone permission denied in browser settings.');
						stopListening();
						return;
					}
					if (event.error === 'aborted') {
						stopListening();
						return;
					}
					// Fallback to MediaRecorder + Groq Whisper
					console.info('SpeechRecognition error, falling back to MediaRecorder + Groq Whisper');
					startMediaRecorder(stream);
				};

				recognition.onend = () => {
					setIsListening(false);
					if (silenceTimerRef.current) {
						clearTimeout(silenceTimerRef.current);
						silenceTimerRef.current = null;
					}
					const pending = currentTranscriptRef.current.trim();
					if (pending) {
						currentTranscriptRef.current = '';
						setMicHint('');
						sendMessage(pending);
						stopListening();
					}
				};

				recognitionRef.current = recognition;
				recognition.start();
				return;
			} catch (speechErr) {
				console.warn('SpeechRecognition failed, falling back to MediaRecorder:', speechErr);
			}
		}

		// Fallback to MediaRecorder
		startMediaRecorder(stream);
	};

	const startCall = () => {
		setError('');

		// Check for insecure HTTP context (Safari & mobile strictly block microphone on plain HTTP)
		if (typeof window !== 'undefined' && window.isSecureContext === false) {
			setError('Microphone requires HTTPS on Safari / mobile. Please open via the secure HTTPS link (https://...ngrok-free.dev) instead of plain http://.');
			return;
		}

		setStatus('connecting');

		// Pre-warm microphone permission during user gesture tap for iOS Safari
		if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
			navigator.mediaDevices.getUserMedia({
				audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }
			}).then((stream) => {
				mediaStreamRef.current = stream;
			}).catch((err) => {
				console.warn('Pre-warm microphone gesture notice:', err);
			});
		}

		// Unlock mobile audio playback on initial user gesture tap:
		try {
			const silentAudio = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
			silentAudio.play().catch(() => {});
		} catch {}

		// Resume speech synthesis on user click:
		if ('speechSynthesis' in window) {
			try {
				window.speechSynthesis.resume();
			} catch (e) {
				console.warn('speech resume error:', e);
			}
		}

		// Unlock persistent audio for mobile Safari
		unlockAudioForMobile();

		let activeProfile = profile;
		if (!activeProfile?.name) {
			try {
				const stored = localStorage.getItem('userProfile');
				if (stored) activeProfile = JSON.parse(stored);
			} catch {}
		}

		const params = new URLSearchParams({
			tts_provider: ttsProviderRef.current,
		});
		if (activeProfile?.name) params.set('name', activeProfile.name);
		if (activeProfile?.phone) params.set('phone', activeProfile.phone);
		if (activeProfile?.email) params.set('email', activeProfile.email);

		const socketUrl = `${BACKEND_URL.replace(/^http/, 'ws')}/conversation/stream?${params.toString()}`;
		const socket = new WebSocket(socketUrl);
		socketRef.current = socket;

		socket.onopen = () => {
			if (activeProfile?.name) {
				try {
					socket.send(JSON.stringify({
						type: 'init_profile',
						user_profile: activeProfile,
						tts_provider: ttsProviderRef.current,
					}));
				} catch {}
			}
		};

		// 8-second keepalive heartbeat to prevent cellular NAT & ngrok timeouts
		if (heartbeatRef.current) clearInterval(heartbeatRef.current);
		heartbeatRef.current = setInterval(() => {
			if (socket.readyState === WebSocket.OPEN) {
				try {
					socket.send(JSON.stringify({ type: 'ping' }));
				} catch {}
			}
		}, 8000);

		socket.onmessage = (event) => {
			if (typeof event.data !== 'string') return;
			let response: any;
			try {
				response = JSON.parse(event.data);
			} catch (err) {
				console.warn('Non-JSON WebSocket message received:', err);
				return;
			}

			// Keepalive heartbeat pong
			if (response.type === 'pong') {
				return;
			}

			// Conversational filler audio from backend to mask network/LLM latency on alternate replies
			if (response.type === 'filler_audio') {
				setIsNiaaThinking(true);
				if (response.audio_b64) {
					const fillerProv = (response.tts_provider as TTSProvider) || ttsProviderRef.current;
					playSarvamAudio(response.text || 'Ek minute...', {
						backendUrl: BACKEND_URL,
						provider: fillerProv,
						speaker: fillerProv === 'rumik' ? 'aisha' : 'simran',
						language: 'hi-IN',
						pace: fillerProv === 'rumik' ? 1.15 : 1.0,
						audioB64: response.audio_b64,
						onStart: () => {
							isSpeakingRef.current = true;
							setIsNiaaSpeaking(true);
							if (mediaStreamRef.current) {
								mediaStreamRef.current.getAudioTracks().forEach((track) => {
									track.enabled = false;
								});
							}
							setIsListening(false);
							setAudioLevel(0);
							if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
								try {
									socketRef.current.send(JSON.stringify({ type: 'agent_speaking', speaking: true }));
								} catch {}
							}
							if (sttProviderRef.current !== 'sarvam') {
								stopListening();
							}
						},
						onEnd: () => {
							isSpeakingRef.current = false;
							setIsNiaaSpeaking(false);
							if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
								try {
									socketRef.current.send(JSON.stringify({ type: 'agent_speaking', speaking: false }));
								} catch {}
							}
							if (!isMutedRef.current && mediaStreamRef.current) {
								mediaStreamRef.current.getAudioTracks().forEach((track) => {
									track.enabled = true;
								});
							}
							if (!isMutedRef.current) {
								setIsListening(true);
							}
						},
					});
				}
				return;
			}

			// Backend turn processing started: Niaa is thinking
			if (response.type === 'agent_thinking') {
				setIsNiaaThinking(true);
				return;
			}

			// Realtime Barge-in: user interrupted while agent was speaking
			if (response.type === 'interrupt') {
				stopSarvamAudio();
				if ('speechSynthesis' in window) {
					try {
						window.speechSynthesis.cancel();
					} catch { }
				}
				isSpeakingRef.current = false;
				setIsNiaaSpeaking(false);
				setIsNiaaThinking(false);
				setStatus('connected');
				setMicHint('Interrupted • Listening to you...');
				return;
			}

			// VAD Speech Start (activates listening UI state; audio interruption is managed cleanly via 'interrupt' signal)
			if (response.type === 'speech_start') {
				setIsListening(true);
				return;
			}

			// Realtime interim partial transcript from Sarvam STT
			if (response.type === 'interim_transcript') {
				if (response.text) {
					setMicHint(`"${response.text}"`);
				}
				return;
			}

			// Turn superseded / merged after user pause or barge-in
			if (response.type === 'turn_superseded') {
				if (response.combined_text) {
					setMessages((current) => {
						const updated = [...current];
						// Pop aborted assistant message if present
						if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
							updated.pop();
						}
						// If previous message was user, update it with combined full text; otherwise append
						if (updated.length > 0 && updated[updated.length - 1].role === 'user') {
							updated[updated.length - 1] = { role: 'user', content: response.combined_text };
						} else {
							updated.push({ role: 'user', content: response.combined_text });
						}
						return updated;
					});
					setMicHint('');
				}
				return;
			}

			// Backend STT Notice / Error: seamlessly fall back to browser Web Speech API
			if (response.type === 'stt_error') {
				console.warn('Backend STT notice:', response.message, 'switching to browser speech recognition');
				setSttProvider('browser');
				sttProviderRef.current = 'browser';
				setMicHint('Browser Voice Active • Speak freely');
				if (pageRef.current === 'CALL') {
					startListening();
				}
				return;
			}

			// Final transcript from Sarvam STT
			if (response.type === 'final_transcript') {
				if (response.text) {
					setMessages((current) => [...current, { role: 'user', content: response.text }]);
					setMicHint('');
					isTurnInFlightRef.current = true;
				}
				return;
			}

			// Handle FIFO Queue state when 2 concurrent calls are active
			if (response.type === 'queued') {
				setStatus('queued');
				setPage('QUEUED');
				setQueuePosition(response.position || 1);
				setQueueMessage(response.message || 'Agents are busy kindly wait');
				return;
			}

			// Active Call Starts (immediately or popped from queue)
			if (response.type === 'start') {
				setStatus('connected');
				setPage('CALL');
				const maxDuration = response.max_duration_seconds || DEFAULT_CALL_DURATION;
				startCallTimer(maxDuration);

				if (response.stt_provider) {
					setSttProvider(response.stt_provider);
					sttProviderRef.current = response.stt_provider;
				}

				const greeting =
					response.greeting ||
					(ttsProviderRef.current === 'sarvam' || ttsProviderRef.current === 'rumik'
						? 'Namaste! Bright Dental Clinic mein aapka swagat hai. Main Niaa bol rahi hoon, main aapki kaise madad kar sakti hoon?'
						: 'Hello, welcome to Bright Dental Clinic. I am Niaa, how may I help you today?');

				let greetingAdded = false;
				const addGreeting = () => {
					if (!greetingAdded) {
						greetingAdded = true;
						setMessages([{ role: 'assistant', content: greeting }]);
					}
				};

				if (voicesRef.current.length === 0 && 'speechSynthesis' in window) {
					voicesRef.current = window.speechSynthesis.getVoices();
				}

				speak(
					greeting,
					undefined,
					response.audio_b64,
					() => {
						addGreeting();
						if (sttProviderRef.current === 'browser') {
							startListening();
						}
					},
					response.tts_provider
				);
				setTimeout(addGreeting, 600);

				// Connect audio streaming for Sarvam STT or start browser speech recognition
				if (sttProviderRef.current === 'sarvam') {
					startAudioStreaming(socket, {
						onError: (err) => {
							console.warn('Sarvam streaming error, falling back to Web Speech:', err);
							setSttProvider('browser');
							sttProviderRef.current = 'browser';
							startListening();
						},
						onAudioLevel: (lvl) => {
							if (!isMutedRef.current && !isSpeakingRef.current) {
								setAudioLevel(lvl);
							}
						},
					}, mediaStreamRef.current || undefined).then(() => {
						setIsListening(true);
						setMicHint('Sarvam Realtime Voice Active • Speak naturally');
					}).catch((err) => {
						console.warn('Could not start Sarvam streaming:', err);
						if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
							setIsListening(false);
							setMicHint('');
							setError('Microphone permission blocked. In Safari, tap "aA" (or settings icon) in the address bar -> Website Settings -> Microphone -> Allow.');
							return;
						}
						setSttProvider('browser');
						sttProviderRef.current = 'browser';
						startListening();
					});
				} else {
					setMicHint('Browser Speech Active • Speak freely');
					startListening();
				}
				return;
			}

			if (response.type === 'payment_required') {
				paymentPendingRef.current = true;
				const bookingId = response.booking?.id || pendingBookingId;
				const payUrl = response.checkout_url || (bookingId ? `${BACKEND_URL}/payment-success?mock=1&booking=${bookingId}` : null);
				setCheckoutUrl(payUrl);
				setPendingBookingId(bookingId || null);
			}

			if (response.type === 'booking_confirmed') {
				paymentPendingRef.current = false;
				setCheckoutUrl(null);
				setPendingBookingId(null);
				stopListening();

				const booking = response.booking as BookingSummary | undefined;
				const dateStr = booking?.date || 'your requested date';
				const timeStr = booking?.time || 'your requested time';
				const bookingObj: BookingSummary = {
					date: dateStr,
					time: timeStr,
					name: booking?.name || profile.name,
				};
				setConfirmedBooking(bookingObj);
				setLastConfirmedBooking(bookingObj);

				const cleanResp = (response.response || '')
					.replace(/<think>[\s\S]*?<\/think>/gi, '')
					.replace(/<think>[\s\S]*$/gi, '')
					.replace(/^(?:Assistant|Agent|Receptionist|Bot):\s*/i, '')
					.trim();
				const confirmation = cleanResp || `Payment confirmed. Your appointment is booked for ${dateStr} at ${timeStr}.`;
				const goodbye = 'Thank you for calling Bright Dental Clinic. Have a lovely day!';
				setMessages((current) => [
					...current,
					{ role: 'assistant', content: confirmation },
					{ role: 'assistant', content: goodbye },
				]);
				setStatus('confirmed');

				// Speaks confirmation, then goodbye, and ONLY AFTER goodbye completely finishes speaking, ends the call
				if (confirmationTimerRef.current) clearTimeout(confirmationTimerRef.current);
				let confirmationEnded = false;
				const finishBookingCall = () => {
					if (!confirmationEnded) {
						confirmationEnded = true;
						endCall();
					}
				};

				speak(
					confirmation,
					() => {
						speak(
							goodbye,
							() => {
								// Snappy 0.3s pause after final words before redirecting
								setTimeout(finishBookingCall, 300);
							},
							undefined,
							undefined,
							response.tts_provider
						);
					},
					response.audio_b64,
					undefined,
					response.tts_provider
				);

				// Safety watchdog: 30 seconds (never cuts off speech prematurely)
				confirmationTimerRef.current = setTimeout(finishBookingCall, 30000);
				return;
			}

			const cleanResponse = (response.response || '')
				.replace(/<think>[\s\S]*?<\/think>/gi, '')
				.replace(/<think>[\s\S]*$/gi, '')
				.replace(/^(?:Assistant|Agent|Receptionist|Bot):\s*/i, '')
				.trim();

			if (response.type === 'call_end') {
				stopListening();
				callTimeExpiredPendingRef.current = false;
				isTurnInFlightRef.current = false;
				if (timerRef.current) {
					clearInterval(timerRef.current);
					timerRef.current = null;
				}

				const endMsg =
					cleanResponse ||
					'Thank you so much for calling Bright Dental Clinic! Take wonderful care of your smile, and have a lovely day!';

				const proceedWithCallEnd = () => {
					setStatus('ended');
					setMessages((current) => [...current, { role: 'assistant', content: endMsg }]);

					let ended = false;
					const autoEnd = () => {
						if (!ended) {
							ended = true;
							endCall();
						}
					};

					// Speaks full ending message, and only after completion waits 0.3s then ends call
					speak(
						endMsg,
						() => {
							setTimeout(autoEnd, 500);
						},
						response.audio_b64,
						undefined,
						response.tts_provider
					);
					// Safety watchdog: 25 seconds
					setTimeout(autoEnd, 25000);
				};

				if (isSpeakingRef.current) {
					// Wait for ongoing speech to complete before speaking ending message
					const waitInterval = setInterval(() => {
						if (!isSpeakingRef.current) {
							clearInterval(waitInterval);
							proceedWithCallEnd();
						}
					}, 250);
				} else {
					proceedWithCallEnd();
				}
				return;
			}

			if (cleanResponse) {
				setIsNiaaThinking(false);
				let messageAdded = false;
				const addMessage = () => {
					if (!messageAdded) {
						messageAdded = true;
						setMessages((current) => [...current, { role: 'assistant', content: cleanResponse }]);
					}
				};

				speak(
					cleanResponse,
					() => {
						isTurnInFlightRef.current = false;
						// If call timer reached 0 while this answer was being spoken, trigger final closing message now!
						if (callTimeExpiredPendingRef.current) {
							console.log('In-flight answer speech completed. Now triggering final call closing message.');
							callTimeExpiredPendingRef.current = false;
							handleCallTimeExpired();
						}
					},
					response.audio_b64,
					() => {
						addMessage();
					},
					response.tts_provider
				);
				// Safety watchdog: ensure message is displayed after 800ms even if audio delay occurs
				setTimeout(addMessage, 800);
			}

			setStatus('connected');
		};

		socket.onerror = () => {
			if (pageRef.current === 'CALL') {
				setStatus('error');
				setError('Could not connect to the voice agent. Check that the backend is running.');
			}
		};

		socket.onclose = () => {
			if (heartbeatRef.current) {
				clearInterval(heartbeatRef.current);
				heartbeatRef.current = null;
			}
			if (pageRef.current === 'CALL') {
				setStatus('ended');
			}
			stopListening();
			if (timerRef.current) {
				clearInterval(timerRef.current);
				timerRef.current = null;
			}
		};
	};

	const endCall = () => {
		callTimeExpiredPendingRef.current = false;
		isTurnInFlightRef.current = false;
		if (heartbeatRef.current) {
			clearInterval(heartbeatRef.current);
			heartbeatRef.current = null;
		}
		stopAudioStreaming();
		stopListening();
		if ('speechSynthesis' in window) {
			try {
				window.speechSynthesis.cancel();
			} catch { }
		}
		stopSarvamAudio();
		isSpeakingRef.current = false;
		setIsNiaaSpeaking(false);
		setIsNiaaThinking(false);
		if (timerRef.current) {
			clearInterval(timerRef.current);
			timerRef.current = null;
		}
		socketRef.current?.close();
		socketRef.current = null;
		paymentPendingRef.current = false;
		setStatus('idle');
		setCheckoutUrl(null);
		setPendingBookingId(null);
		if (confirmationTimerRef.current) {
			clearTimeout(confirmationTimerRef.current);
			confirmationTimerRef.current = null;
		}
		setConfirmedBooking(null);
		setMicHint('');
		setCallElapsedSeconds(0);
		setAudioLevel(0);
		setIsMuted(false);
		isMutedRef.current = false;
		setIsSpeakerMuted(false);
		setSpeakerMuted(false);
		setPage('INTRO');
	};

	if (page === 'PAYMENT_SUCCESS' || page === 'PAYMENT_CANCEL') {
		const success = page === 'PAYMENT_SUCCESS';
		return (
			<main className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-6">
				<section className="max-w-md text-center space-y-4">
					<p className="text-5xl">{success ? '✓' : '×'}</p>
					<h1 className="text-3xl font-semibold">Payment {success ? 'successful' : 'cancelled'}</h1>
					<p className="text-slate-300">Return to the clinic call window to continue.</p>
					<button
						onClick={() => {
							window.close();
							window.setTimeout(() => window.location.replace('/'), 100);
						}}
						className="rounded-lg bg-cyan-500 px-5 py-3 font-semibold text-slate-950"
					>
						Return
					</button>
				</section>
			</main>
		);
	}

	if (page === 'FORM') {
		return (
			<main className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-6">
				<form className="w-full max-w-md space-y-5 rounded-2xl border border-slate-800 bg-slate-900 p-8" onSubmit={async (event) => {
					event.preventDefault();
					if (!profile.name.trim() || !/^[6-9]\d{9}$/.test(profile.phone)) {
						setError('Enter your name and a valid 10-digit Indian phone number.');
						return;
					}
					localStorage.setItem('userProfile', JSON.stringify(profile));
					try {
						await fetch(`${BACKEND_URL}/store-profile`, {
							method: 'POST',
							headers: {
								'Content-Type': 'application/json',
								'ngrok-skip-browser-warning': 'true',
							},
							body: JSON.stringify(profile),
						});
					} catch { /* proceed */ }
					setPage('INTRO');
				}}>
					<div><p className="text-sm uppercase tracking-widest text-cyan-300">Bright Dental Clinic</p><h1 className="mt-2 text-3xl font-semibold">Start with your details</h1></div>
					<input className="w-full rounded-lg border border-slate-700 bg-slate-800 px-4 py-3 outline-none focus:border-cyan-400 text-white placeholder-slate-500" placeholder="Full name" value={profile.name} onChange={(event) => setProfile({ ...profile, name: event.target.value })} />
					<input className="w-full rounded-lg border border-slate-700 bg-slate-800 px-4 py-3 outline-none focus:border-cyan-400 text-white placeholder-slate-500" placeholder="Phone number" value={profile.phone} onChange={(event) => setProfile({ ...profile, phone: event.target.value })} />
					<input className="w-full rounded-lg border border-slate-700 bg-slate-800 px-4 py-3 outline-none focus:border-cyan-400 text-white placeholder-slate-500" placeholder="Email (optional)" type="email" value={profile.email} onChange={(event) => setProfile({ ...profile, email: event.target.value })} />
					{error && <p className="text-sm text-rose-300">{error}</p>}
					<button className="w-full rounded-lg bg-cyan-400 px-4 py-3 font-semibold text-slate-950 hover:bg-cyan-300 transition-colors" type="submit">Continue</button>
				</form>
			</main>
		);
	}

	if (page === 'INTRO') {
		return (
			<main className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-6">
				<section className="max-w-xl space-y-6 text-center">
					<p className="text-sm uppercase tracking-widest text-cyan-300">Bright Dental Clinic</p>
					<h1 className="text-5xl font-semibold tracking-tight">Your dental receptionist is ready.</h1>
					<p className="text-lg text-slate-300">
						Ask about treatments, timings, fees, or book an appointment by voice.
					</p>
					<div className="flex flex-wrap items-center justify-center gap-4 text-xs text-slate-400">
						<span className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/80 px-3 py-1.5">
							<Clock size={14} className="text-cyan-400" /> Max 3 mins per call
						</span>
						<span className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/80 px-3 py-1.5">
							<Users size={14} className="text-cyan-400" /> Max 2 concurrent lines
						</span>
						<span className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/80 px-3 py-1.5">
							<ShieldCheck size={14} className="text-emerald-400" /> AI Safety Guardrails
						</span>
					</div>
					{lastConfirmedBooking && (
						<div className="mx-auto max-w-md rounded-2xl border border-emerald-400/40 bg-emerald-400/10 p-4 text-center shadow-lg shadow-emerald-950/20">
							<div className="flex items-center justify-center gap-1.5 text-xs uppercase tracking-wider font-semibold text-emerald-400">
								<Sparkles size={14} /> Confirmed Appointment
							</div>
							<p className="mt-1 text-base font-semibold text-white">
								📅 {(() => {
									try {
										const d = new Date(lastConfirmedBooking.date + 'T12:00:00');
										return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
									} catch {
										return lastConfirmedBooking.date;
									}
								})()} at {lastConfirmedBooking.time}
							</p>
							{lastConfirmedBooking.doctor && (
								<p className="text-xs font-semibold text-cyan-300 mt-1">
									👨‍⚕️ Specialist: {lastConfirmedBooking.doctor}
								</p>
							)}
							<p className="text-xs text-slate-300 mt-0.5">
								Patient: {lastConfirmedBooking.name || profile.name}
							</p>
							<div className="mt-2.5">
								<a
									href={lastConfirmedBooking.calendar_url || `https://calendar.google.com/calendar/r/day/${lastConfirmedBooking.date.replace(/-/g, '/')}`}
									target="_blank"
									rel="noopener noreferrer"
									className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/20 border border-emerald-500/30 px-3 py-1 text-xs font-medium text-emerald-300 hover:bg-emerald-500/30 transition-all hover:scale-105"
								>
									<Calendar size={12} /> View {lastConfirmedBooking.date} on Google Calendar <ExternalLink size={11} />
								</a>
							</div>
						</div>
					)}
					<div className="flex flex-col sm:flex-row items-center justify-center gap-3">
						<button
							onClick={startCall}
							className="inline-flex items-center gap-3 rounded-full bg-cyan-400 px-7 py-4 font-semibold text-slate-950 hover:bg-cyan-300 transition-all duration-200 hover:scale-105 shadow-lg shadow-cyan-500/20"
						>
							<Phone size={20} /> Start voice call
						</button>
						<button
							type="button"
							onClick={cycleTtsProvider}
							title="Click to switch voice engine: Sarvam AI / Rumik AI / Browser Voice"
							className={`inline-flex items-center gap-2 rounded-full border px-4 py-3.5 text-xs font-semibold transition-all hover:scale-105 active:scale-95 cursor-pointer shadow-md ${
								ttsProvider === 'rumik'
									? 'border-cyan-500/50 bg-cyan-500/15 text-cyan-300 shadow-cyan-950/30'
									: ttsProvider === 'sarvam'
									? 'border-violet-500/50 bg-violet-500/15 text-violet-300 shadow-violet-950/30'
									: 'border-slate-700 bg-slate-800/90 text-slate-300 shadow-slate-950/30'
							}`}
						>
							<span className="text-slate-400">Voice Engine:</span>
							{ttsProvider === 'rumik' ? (
								<span className="text-cyan-300 font-bold">⚡ Rumik Silk (Aisha)</span>
							) : ttsProvider === 'sarvam' ? (
								<span className="text-violet-300 font-bold">🎙️ Sarvam AI (Simran)</span>
							) : (
								<span className="text-slate-200 font-bold">🔊 Browser Voice</span>
							)}
							<span className="text-[10px] text-slate-500 underline ml-0.5">tap to switch</span>
						</button>
					</div>
				</section>
			</main>
		);
	}

	// Queue Screen when both receptionists are occupied
	if (page === 'QUEUED') {
		return (
			<main className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-6">
				<section className="w-full max-w-md text-center space-y-6 rounded-3xl border border-amber-500/30 bg-slate-900/90 p-8 shadow-2xl backdrop-blur-xl">
					<div className="relative mx-auto flex h-24 w-24 items-center justify-center">
						<span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-20"></span>
						<span className="relative flex h-16 w-16 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/40 text-amber-400">
							<Users size={32} />
						</span>
					</div>

					<div className="space-y-2">
						<p className="text-xs uppercase tracking-widest text-amber-400">All lines currently in use</p>
						<h1 className="text-2xl font-bold tracking-tight text-white">{queueMessage}</h1>
						<p className="text-sm text-slate-300">
							Only 2 patients can talk to the receptionist at a time to ensure quality care.
						</p>
					</div>

					<div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
						<p className="text-xs text-slate-400 uppercase tracking-wider">Your Queue Position</p>
						<p className="mt-1 text-3xl font-extrabold text-amber-300">
							#{queuePosition} <span className="text-sm font-normal text-slate-400">in line</span>
						</p>
						<p className="mt-2 text-xs text-slate-400">
							⚡ Your call will pop in automatically as soon as an active call ends.
						</p>
					</div>

					<button
						onClick={endCall}
						className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800 px-6 py-2.5 text-sm font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
					>
						<PhoneOff size={16} /> Leave queue
					</button>
				</section>
			</main>
		);
	}

	// Active Call Screen replicating the exact UI from the reference picture
	return (
		<CallScreen
			messages={messages}
			isNiaaSpeaking={isNiaaSpeaking}
			isNiaaThinking={isNiaaThinking}
			isListening={isListening}
			audioLevel={audioLevel}
			status={status}
			error={error}
			timeLeft={timeLeft}
			callElapsedSeconds={callElapsedSeconds}
			continuousMode={continuousMode}
			ttsProvider={ttsProvider}
			confirmedBooking={confirmedBooking}
			checkoutUrl={checkoutUrl}
			pendingBookingId={pendingBookingId}
			onEndCall={endCall}
			onToggleMute={toggleMute}
			isMuted={isMuted}
			onToggleSpeaker={toggleSpeaker}
			isSpeakerMuted={isSpeakerMuted}
			onOpenPayment={openPaymentAndConfirm}
			onCycleTtsProvider={cycleTtsProvider}
			userName={profile.name}
			interimTranscript={micHint}
		/>
	);
}

export default App;
