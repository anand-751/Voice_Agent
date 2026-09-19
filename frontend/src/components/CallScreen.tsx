import React, { useState, useEffect } from 'react';
import {
	Mic,
	MicOff,
	PhoneOff,
	Volume2,
	VolumeX,
	MessageSquare,
	X,
	Sparkles,
	CreditCard,
	Calendar,
	ExternalLink,
	Clock,
	Maximize2,
	Minimize2,
} from 'lucide-react';
import InfinityLogo from './InfinityLogo';
import AudioWaveform from './AudioWaveform';
import type { TTSProvider } from '../services/tts';

export interface Message {
	role: 'user' | 'assistant';
	content: string;
}

export interface BookingSummary {
	id?: string;
	date: string;
	time: string;
	name?: string;
	doctor?: string;
	calendar_url?: string;
}

interface CallScreenProps {
	messages: Message[];
	isNiaaSpeaking: boolean;
	isNiaaThinking: boolean;
	isListening: boolean;
	audioLevel: number;
	status: string;
	error?: string;
	timeLeft: number;
	callElapsedSeconds: number;
	continuousMode: boolean;
	ttsProvider: TTSProvider;
	confirmedBooking: BookingSummary | null;
	checkoutUrl: string | null;
	pendingBookingId: string | null;
	onEndCall: () => void;
	onToggleMute: () => void;
	isMuted: boolean;
	onToggleSpeaker?: () => void;
	isSpeakerMuted?: boolean;
	onOpenPayment?: () => void;
	onCycleTtsProvider?: () => void;
	userName?: string;
	interimTranscript?: string;
}

export const CallScreen: React.FC<CallScreenProps> = ({
	messages,
	isNiaaSpeaking,
	isNiaaThinking,
	isListening,
	audioLevel,
	timeLeft,
	callElapsedSeconds,
	ttsProvider,
	confirmedBooking,
	checkoutUrl,
	pendingBookingId,
	onEndCall,
	onToggleMute,
	isMuted,
	onToggleSpeaker,
	isSpeakerMuted = false,
	onOpenPayment,
	onCycleTtsProvider,
	interimTranscript,
}) => {
	const [showTranscript, setShowTranscript] = useState(false);
	const [isFramedMode, setIsFramedMode] = useState(true);
	const [currentTime, setCurrentTime] = useState('9:41');

	// Live iOS clock
	useEffect(() => {
		const updateClock = () => {
			const now = new Date();
			setCurrentTime(
				now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
			);
		};
		updateClock();
		const interval = setInterval(updateClock, 10000);
		return () => clearInterval(interval);
	}, []);

	// Format mm:ss (e.g. 00:28)
	const formatDuration = (seconds: number) => {
		const mins = Math.floor(seconds / 60);
		const secs = seconds % 60;
		return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
	};

	// Determine latest assistant and user dialogue
	const lastAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant');
	const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user');

	// Clean caller speech (prefer live interim transcript if caller is actively speaking, otherwise last user message)
	const cleanInterim = (interimTranscript || '').replace(/^["']|["']$/g, '').trim();
	const callerSpeech = cleanInterim || lastUserMsg?.content || 'Say "Namaste" or ask about appointments...';

	// Clean assistant speech
	const assistantSpeech = lastAssistantMsg?.content || 'Namaste! Welcome to Bright Dental Clinic. How may I assist you today?';

	let headlineStatus = 'Listening and responding...';
	if (isNiaaSpeaking) {
		headlineStatus = 'Niaa is speaking...';
	} else if (isNiaaThinking) {
		headlineStatus = 'Thinking and processing...';
	} else if (isListening && !isMuted) {
		headlineStatus = 'Listening to you...';
	} else if (isMuted) {
		headlineStatus = 'Microphone muted';
	}

	// The Mobile Call Screen Body
	const renderPhoneContent = () => (
		<div className="relative w-full h-full flex flex-col justify-between overflow-hidden bg-[#050812] text-white select-none font-sans">
			{/* Ultra-smooth velvety background gradient (clean, minimal, no dot noise) */}
			<div
				className="absolute inset-0 pointer-events-none"
				style={{
					background:
						'radial-gradient(ellipse 85% 55% at 50% 36%, rgba(15, 38, 86, 0.45) 0%, rgba(7, 16, 38, 0.7) 45%, #040711 100%)',
				}}
			/>

			{/* TOP: iOS Status Bar + Clean Call Header */}
			<div className="relative z-20 pt-3.5 sm:pt-4 px-6 flex flex-col items-center">
				{/* iOS Status Bar Row */}
				<div className="w-full flex items-center justify-between text-xs text-white/80 font-medium px-1 mb-2">
					<span className="font-semibold">{currentTime}</span>

					{/* Dynamic Island / Notch replica */}
					<div className="hidden sm:flex items-center justify-center gap-2 bg-black/95 border border-white/10 px-4 py-1 rounded-full shadow-inner">
						<span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
						<span className="text-[11px] text-slate-300 font-mono tracking-wide">Live</span>
					</div>

					{/* Cellular Signal, WiFi, Battery */}
					<div className="flex items-center gap-1.5 text-white/80">
						{/* Cellular */}
						<svg className="w-4 h-3 fill-current" viewBox="0 0 18 14">
							<rect x="1" y="10" width="2.5" height="4" rx="0.5" />
							<rect x="5.5" y="7.5" width="2.5" height="6.5" rx="0.5" />
							<rect x="10" y="4.5" width="2.5" height="9.5" rx="0.5" />
							<rect x="14.5" y="1.5" width="2.5" height="12.5" rx="0.5" />
						</svg>
						{/* WiFi */}
						<svg className="w-4 h-3 fill-current" viewBox="0 0 20 16">
							<path d="M10 14a2 2 0 100-4 2 2 0 000 4zm6.36-6.36a9 9 0 00-12.72 0l1.41 1.41a7 7 0 019.9 0l1.41-1.41zm3.54-3.54a14 14 0 00-19.8 0l1.41 1.41a12 12 0 0116.98 0l1.41-1.41z" />
						</svg>
						{/* Battery */}
						<div className="w-5 h-2.5 rounded-sm border border-white/70 p-0.5 flex items-center">
							<div className="w-3/4 h-full bg-white rounded-2xs" />
						</div>
					</div>
				</div>

				{/* Live Call Header (clean, minimal, exactly as in screenshot) */}
				<div className="mt-4 sm:mt-5 text-center flex flex-col items-center">
					<div className="flex items-center gap-2.5">
						{/* Pulsing Red Dot */}
						<span className="relative flex h-3 w-3 items-center justify-center">
							<span className="animate-pulse-dot absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75" />
							<span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500 shadow-md shadow-red-500/80" />
						</span>
						<h1 className="text-base sm:text-lg font-medium tracking-wide text-white">
							AI Receptionist — Live Call
						</h1>
					</div>

					{/* Call Timer (00:28) */}
					<div className="mt-1 text-sm sm:text-base font-normal tracking-wider text-slate-400 font-mono">
						{formatDuration(callElapsedSeconds)}
					</div>
				</div>

				{/* Subtle Time Limit warning if nearing end */}
				{timeLeft <= 30 && (
					<div className="mt-2 inline-flex items-center gap-1 rounded-full bg-amber-500/20 border border-amber-500/40 px-2.5 py-0.5 text-xs text-amber-300 font-mono animate-pulse">
						<Clock size={11} /> {timeLeft}s remaining
					</div>
				)}
			</div>

			{/* CENTER: Audio Waveform + Circular Glowing Ring + Exact Infinity Logo */}
			<div className="relative z-10 flex-1 flex flex-col items-center justify-center my-auto py-2">
				<div className="relative w-full max-w-md flex items-center justify-center">
					{/* Electric Blue Audio Waveform extending to sides */}
					<AudioWaveform
						isSpeaking={isNiaaSpeaking || (isListening && !isMuted)}
						audioLevel={audioLevel}
					/>

					{/* Center Circular Orb */}
					<div className="relative z-10 flex items-center justify-center">
						{/* Soft ambient aura */}
						<div className="absolute -inset-4 rounded-full bg-blue-500/15 blur-2xl pointer-events-none animate-pulse" />

						{/* Neon Circular Ring */}
						<div className="relative w-48 h-48 sm:w-56 sm:h-56 rounded-full p-[2.5px] bg-gradient-to-b from-cyan-400 via-blue-600 to-blue-950 shadow-[0_0_30px_rgba(14,165,233,0.35)] animate-ring-breath flex items-center justify-center">
							{/* Glowing Cyan Top Arc highlight */}
							<div className="absolute top-0 inset-x-10 h-1 bg-gradient-to-r from-transparent via-cyan-300 to-transparent blur-[1px] rounded-full" />

							{/* Circular Container Disc with subtle illumination for authentic logo */}
							<div className="w-full h-full rounded-full bg-gradient-to-b from-[#0e1930] via-[#070d1c] to-[#040814] border border-blue-500/25 flex items-center justify-center shadow-inner relative overflow-hidden">
								{/* Subtle inner radial backlight to give crystal clarity to the dark & blue logo */}
								<div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(56,189,248,0.15)_0%,rgba(15,23,42,0.45)_65%,transparent_85%)]" />

								{/* EXACT AUTHENTIC COMPANY INFINITY LOGO with Animated Heartbeat */}
								<InfinityLogo
									isSpeaking={isNiaaSpeaking || (isListening && !isMuted)}
									isHeartbeatActive={true}
									audioLevel={audioLevel}
									size={126}
								/>
							</div>
						</div>
					</div>
				</div>

				{/* DUAL CONVERSATIONAL CAPTIONS (Shows BOTH what you are speaking and what AI speaks) */}
				<div className="mt-5 sm:mt-7 px-4 sm:px-6 w-full max-w-sm sm:max-w-md mx-auto space-y-2">
					{/* Status Sub-headline */}
					<div className="flex items-center justify-center gap-1.5 text-xs text-slate-400 font-medium">
						<span
							className={`w-2 h-2 rounded-full transition-all duration-300 ${
								isNiaaSpeaking
									? 'bg-cyan-400 animate-pulse shadow-[0_0_8px_rgba(34,211,238,0.8)]'
									: isListening && !isMuted
									? 'bg-emerald-400 animate-ping shadow-[0_0_8px_rgba(52,211,153,0.8)]'
									: isNiaaThinking
									? 'bg-amber-400 animate-pulse shadow-[0_0_8px_rgba(251,191,36,0.8)]'
									: 'bg-slate-600'
							}`}
						/>
						<span className="tracking-wide">{headlineStatus}</span>
					</div>

					{/* Dual Captions Container */}
					<div className="rounded-2xl border border-white/10 bg-slate-950/70 backdrop-blur-xl p-3 shadow-[0_12px_32px_rgba(0,0,0,0.5)] space-y-2 text-left">
						{/* You (User) Caption */}
						<div
							className={`flex items-start gap-2.5 p-2 rounded-xl transition-all duration-300 ${
								isListening && !isMuted
									? 'bg-emerald-500/10 border border-emerald-500/30'
									: 'bg-white/[0.02] border border-white/[0.04]'
							}`}
						>
							<div className="flex-shrink-0 mt-0.5">
								<span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 tracking-wider uppercase">
									<Mic size={10} />
									You
								</span>
							</div>
							<p
								className={`text-xs sm:text-[13px] leading-relaxed line-clamp-2 flex-1 transition-colors ${
									isListening && !isMuted && cleanInterim
										? 'text-emerald-200 font-medium'
										: lastUserMsg?.content
										? 'text-slate-200 font-normal'
										: 'text-slate-500 italic'
								}`}
							>
								{callerSpeech}
							</p>
						</div>

						{/* Niaa (AI Agent) Caption */}
						<div
							className={`flex items-start gap-2.5 p-2 rounded-xl transition-all duration-300 ${
								isNiaaSpeaking
									? 'bg-cyan-500/15 border border-cyan-400/40 shadow-[0_0_20px_rgba(6,182,212,0.18)]'
									: 'bg-white/[0.02] border border-white/[0.04]'
							}`}
						>
							<div className="flex-shrink-0 mt-0.5">
								<span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 tracking-wider uppercase">
									<Sparkles size={10} />
									Niaa
								</span>
							</div>
							<div className="text-xs sm:text-[13px] leading-relaxed line-clamp-2 flex-1 text-cyan-50/90 font-normal">
								{isNiaaThinking ? (
									<span className="inline-flex items-center gap-1.5 text-cyan-300 animate-pulse font-medium">
										<Sparkles size={12} className="animate-spin" />
										Checking details & calendar...
									</span>
								) : (
									assistantSpeech
								)}
							</div>
						</div>
					</div>
				</div>
			</div>

			{/* RESERVATION / PAYMENT POPUPS (Non-intrusive bottom glass banners) */}
			{confirmedBooking && (
				<div className="relative z-30 mx-5 mb-3 rounded-2xl border border-emerald-500/40 bg-emerald-950/80 p-3.5 backdrop-blur-xl shadow-xl animate-in fade-in">
					<div className="flex items-center justify-between">
						<div className="flex items-center gap-1.5 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
							<Sparkles size={14} /> Confirmed Appointment
						</div>
						{confirmedBooking.calendar_url && (
							<a
								href={confirmedBooking.calendar_url}
								target="_blank"
								rel="noreferrer"
								className="text-[11px] text-emerald-300 underline flex items-center gap-1"
							>
								<Calendar size={11} /> Calendar <ExternalLink size={10} />
							</a>
						)}
					</div>
					<p className="mt-1 text-sm font-semibold text-white">
						📅 {confirmedBooking.date} at {confirmedBooking.time}
					</p>
				</div>
			)}

			{(checkoutUrl || pendingBookingId) && !confirmedBooking && (
				<div className="relative z-30 mx-5 mb-3 rounded-2xl border border-amber-500/40 bg-amber-950/80 p-3.5 backdrop-blur-xl shadow-xl flex items-center justify-between gap-3 animate-in fade-in">
					<div>
						<p className="text-xs font-semibold text-amber-300 uppercase tracking-wider">Slot Reserved</p>
						<p className="text-xs text-slate-300 mt-0.5">₹500 booking fee to confirm</p>
					</div>
					<button
						type="button"
						onClick={onOpenPayment}
						className="inline-flex items-center gap-1.5 rounded-full bg-amber-400 px-3.5 py-1.5 text-xs font-bold text-slate-950 hover:bg-amber-300 transition-transform active:scale-95"
					>
						<CreditCard size={13} /> Pay Now
					</button>
				</div>
			)}

			{/* BOTTOM CALL CONTROLS: Mute, End Call, Speaker */}
			<div className="relative z-20 pb-8 sm:pb-9 pt-2 px-8">
				<div className="flex items-center justify-around max-w-xs sm:max-w-sm mx-auto">
					{/* 1. MUTE */}
					<div className="flex flex-col items-center gap-2">
						<button
							type="button"
							onClick={onToggleMute}
							aria-label={isMuted ? 'Unmute microphone' : 'Mute microphone'}
							className={`w-16 h-16 rounded-full flex items-center justify-center transition-all duration-200 active:scale-95 shadow-lg backdrop-blur-md cursor-pointer ${
								isMuted
									? 'bg-rose-500/20 border-2 border-rose-500 text-rose-400 shadow-rose-950/40'
									: 'bg-white/10 hover:bg-white/15 border border-white/10 text-white hover:scale-105'
							}`}
						>
							{isMuted ? <MicOff size={24} /> : <Mic size={24} />}
						</button>
						<span className="text-xs font-medium text-slate-300 tracking-wide">
							{isMuted ? 'Unmute' : 'Mute'}
						</span>
					</div>

					{/* 2. END CALL */}
					<div className="flex flex-col items-center gap-2">
						<button
							type="button"
							onClick={onEndCall}
							aria-label="End call"
							className="w-[72px] h-[72px] rounded-full bg-[#ea3838] hover:bg-red-600 active:scale-95 text-white flex items-center justify-center shadow-lg shadow-red-500/35 transition-all duration-200 hover:scale-105 cursor-pointer"
						>
							<PhoneOff size={28} className="stroke-[2.5]" />
						</button>
						<span className="text-xs font-medium text-slate-200 tracking-wide">
							End Call
						</span>
					</div>

					{/* 3. SPEAKER */}
					<div className="flex flex-col items-center gap-2">
						<button
							type="button"
							onClick={onToggleSpeaker}
							aria-label={isSpeakerMuted ? 'Turn speaker on' : 'Mute speaker'}
							className={`w-16 h-16 rounded-full flex items-center justify-center transition-all duration-200 active:scale-95 shadow-lg backdrop-blur-md cursor-pointer ${
								isSpeakerMuted
									? 'bg-amber-500/20 border-2 border-amber-500 text-amber-400'
									: 'bg-white/10 hover:bg-white/15 border border-white/10 text-white hover:scale-105'
							}`}
						>
							{isSpeakerMuted ? <VolumeX size={24} /> : <Volume2 size={24} />}
						</button>
						<span className="text-xs font-medium text-slate-300 tracking-wide">
							{isSpeakerMuted ? 'Muted' : 'Speaker'}
						</span>
					</div>
				</div>

				{/* Home Bar Indicator */}
				<div className="w-32 h-1 bg-white/20 rounded-full mx-auto mt-6" />
			</div>

			{/* SLIDE-UP TRANSCRIPT DRAWER */}
			{showTranscript && (
				<div className="absolute inset-0 z-40 bg-slate-950/92 backdrop-blur-xl flex flex-col p-5 animate-in fade-in duration-200">
					<div className="flex items-center justify-between pb-3 border-b border-slate-800">
						<div className="flex items-center gap-2">
							<MessageSquare size={17} className="text-cyan-400" />
							<h3 className="font-semibold text-white text-base">Conversation Transcript</h3>
						</div>
						<button
							onClick={() => setShowTranscript(false)}
							className="p-1.5 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-colors cursor-pointer"
						>
							<X size={18} />
						</button>
					</div>

					{onCycleTtsProvider && (
						<div className="py-2.5 border-b border-slate-800/80 flex items-center justify-between text-xs">
							<span className="text-slate-400">Voice Engine:</span>
							<button
								type="button"
								onClick={onCycleTtsProvider}
								className="px-2.5 py-1 rounded-full bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 font-medium hover:bg-cyan-500/25 transition-colors cursor-pointer"
							>
								{ttsProvider === 'rumik'
									? '⚡ Rumik Silk (Aisha)'
									: '🎙️ Sarvam AI (Simran)'}
							</button>
						</div>
					)}

					<div className="flex-1 overflow-y-auto custom-scrollbar py-4 space-y-3">
						{messages.length === 0 ? (
							<p className="text-center text-sm text-slate-500 mt-10">
								Live conversation messages will appear here.
							</p>
						) : (
							messages.map((msg, i) => (
								<div
									key={i}
									className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
								>
									<div
										className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-xs sm:text-sm leading-relaxed ${
											msg.role === 'user'
												? 'bg-cyan-500 text-slate-950 font-medium rounded-br-none'
												: 'bg-slate-800 text-slate-100 border border-slate-700/60 rounded-bl-none'
										}`}
									>
										{msg.content}
									</div>
								</div>
							))
						)}
					</div>

					<button
						onClick={() => setShowTranscript(false)}
						className="w-full mt-3 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-colors cursor-pointer"
					>
						Close Transcript
					</button>
				</div>
			)}
		</div>
	);

	// Full View Mode (edge-to-edge on screen or mobile)
	if (!isFramedMode) {
		return (
			<main className="fixed inset-0 w-screen h-screen bg-[#03050a] flex items-center justify-center overflow-hidden">
				{/* Top Floating Control Bar */}
				<div className="fixed top-4 right-5 z-50 flex items-center gap-2">
					<button
						onClick={() => setIsFramedMode(true)}
						className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/15 border border-white/10 text-xs text-white backdrop-blur-xl transition-all cursor-pointer shadow-lg"
					>
						<Minimize2 size={12} />
						<span>Phone Frame</span>
					</button>
					<button
						onClick={() => setShowTranscript(!showTranscript)}
						className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/15 border border-white/10 text-xs text-white backdrop-blur-xl transition-all cursor-pointer shadow-lg"
					>
						<MessageSquare size={12} className="text-cyan-400" />
						<span>Transcript</span>
					</button>
				</div>

				<div className="w-full h-full max-w-md mx-auto">
					{renderPhoneContent()}
				</div>
			</main>
		);
	}

	// Desktop Showcase Mode: Premium iPhone 16 Pro Titanium Bezel with external floating controls
	return (
		<main className="min-h-screen w-full bg-[#030509] flex items-center justify-center p-3 sm:p-6 lg:p-8 relative">
			{/* Top-Right Discreet Desktop Controls (outside phone screen) */}
			<div className="fixed top-5 right-6 z-50 flex items-center gap-2.5">
				<button
					onClick={() => setIsFramedMode(false)}
					className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-xs text-slate-300 hover:text-white backdrop-blur-xl transition-all cursor-pointer shadow-md"
					title="Switch to Full View"
				>
					<Maximize2 size={12} />
					<span>Full View</span>
				</button>
				<button
					onClick={() => setShowTranscript(true)}
					className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-xs text-slate-300 hover:text-white backdrop-blur-xl transition-all cursor-pointer shadow-md"
					title="Open Transcript"
				>
					<MessageSquare size={12} className="text-cyan-400" />
					<span>Transcript</span>
				</button>
			</div>

			{/* iPhone 16 Pro Frame */}
			<div className="relative w-full max-w-[385px] h-[810px] rounded-[52px] p-[9px] bg-gradient-to-b from-[#2a3444] via-[#161c28] to-[#0d121c] shadow-[0_30px_90px_rgba(0,0,0,0.9),0_0_50px_rgba(14,165,233,0.12)] ring-1 ring-white/10 flex flex-col items-center justify-center transition-all duration-300">
				{/* Titanium edge bevel highlights */}
				<div className="absolute inset-0 rounded-[52px] border border-white/15 pointer-events-none" />
				<div className="absolute inset-[2.5px] rounded-[49.5px] border border-black/90 pointer-events-none" />

				{/* Earpiece speaker slot */}
				<div className="absolute top-3.5 w-12 h-1 bg-[#121620] rounded-full z-30" />

				{/* Display */}
				<div className="w-full h-full rounded-[43px] overflow-hidden relative shadow-inner">
					{renderPhoneContent()}
				</div>
			</div>
		</main>
	);
};

export default CallScreen;
