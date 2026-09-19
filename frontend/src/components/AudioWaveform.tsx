import React, { useEffect, useState } from 'react';

interface AudioWaveformProps {
	isSpeaking?: boolean; // When AI or user voice is active
	audioLevel?: number; // 0.0 to 1.0 real-time volume
}

// Natural voice frequency envelope heights (smooth curve tapering gracefully at edges)
const LEFT_ENVELOPE = [
	0.12, 0.18, 0.28, 0.42, 0.58, 0.38, 0.72, 0.90, 0.65, 0.82, 0.55, 0.88, 0.70, 0.82, 0.58, 0.68, 0.40, 0.24
];
const RIGHT_ENVELOPE = [
	0.24, 0.40, 0.68, 0.58, 0.82, 0.70, 0.88, 0.55, 0.82, 0.65, 0.90, 0.72, 0.38, 0.58, 0.42, 0.28, 0.18, 0.12
];

export const AudioWaveform: React.FC<AudioWaveformProps> = ({
	isSpeaking = false,
	audioLevel = 0,
}) => {
	const [timePhase, setTimePhase] = useState(0);

	useEffect(() => {
		let animId: number;
		let phase = 0;

		const animate = () => {
			phase += isSpeaking ? 0.07 + audioLevel * 0.08 : 0.02;
			setTimePhase(phase);
			animId = requestAnimationFrame(animate);
		};

		animId = requestAnimationFrame(animate);
		return () => cancelAnimationFrame(animId);
	}, [isSpeaking, audioLevel]);

	const renderWing = (envelope: number[], isLeft: boolean) => {
		return envelope.map((baseHeight, idx) => {
			const distFromCenter = isLeft ? envelope.length - 1 - idx : idx;
			const edgeFade = Math.max(0.2, 1 - (distFromCenter / envelope.length) * 0.7);
			const ripple = Math.sin(timePhase + idx * 0.4);

			let scale: number;
			if (isSpeaking) {
				const energy = Math.max(audioLevel, 0.3);
				scale = Math.min(1.0, Math.max(0.12, baseHeight * 0.45 + energy * 0.55 * (0.6 + 0.4 * ripple)));
			} else {
				scale = Math.max(0.1, baseHeight * 0.32 + 0.08 * ripple);
			}

			const heightPx = Math.max(4, Math.round(scale * 68));
			const opacity = Math.min(1, Math.max(0.25, scale * edgeFade * 1.1));

			return (
				<div
					key={`${isLeft ? 'l' : 'r'}-${idx}`}
					className="w-[2.5px] sm:w-[3px] rounded-full transition-all duration-75"
					style={{
						height: `${heightPx}px`,
						opacity,
						background: isSpeaking
							? 'linear-gradient(180deg, #38bdf8 0%, #0284c7 50%, #1d4ed8 100%)'
							: 'linear-gradient(180deg, #0284c7 0%, #0369a1 60%, #1e3a8a 100%)',
						boxShadow: isSpeaking
							? `0 0 ${3 + scale * 6}px rgba(56, 189, 248, 0.7)`
							: '0 0 3px rgba(2, 132, 199, 0.25)',
					}}
				/>
			);
		});
	};

	return (
		<div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex items-center justify-between pointer-events-none px-3 sm:px-5 z-0">
			{/* Left Wing */}
			<div className="flex items-center justify-end gap-[3px] sm:gap-[3.5px] flex-1 pr-4 sm:pr-8">
				{renderWing(LEFT_ENVELOPE, true)}
			</div>

			{/* Center Spacer for Circle */}
			<div className="w-[190px] sm:w-[230px] shrink-0" />

			{/* Right Wing */}
			<div className="flex items-center justify-start gap-[3px] sm:gap-[3.5px] flex-1 pl-4 sm:pr-8">
				{renderWing(RIGHT_ENVELOPE, false)}
			</div>
		</div>
	);
};

export default AudioWaveform;
