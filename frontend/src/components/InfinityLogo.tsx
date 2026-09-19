import React from 'react';

interface InfinityLogoProps {
	isSpeaking?: boolean;
	isHeartbeatActive?: boolean;
	audioLevel?: number; // 0.0 to 1.0 energy level
	className?: string;
	size?: number; // width in px
}

export const InfinityLogo: React.FC<InfinityLogoProps> = ({
	isSpeaking = false,
	isHeartbeatActive = true,
	audioLevel = 0,
	className = '',
	size = 126,
}) => {
	// Dynamic scale based on real-time audio volume
	const dynamicScale = isSpeaking ? 1 + Math.min(audioLevel * 0.16, 0.14) : 1;

	// Refined glow that defines the authentic dark loop edge and amplifies the blue loop
	const dynamicFilter = isSpeaking
		? `drop-shadow(0 0 1px rgba(255, 255, 255, 0.55)) drop-shadow(0 0 ${12 + audioLevel * 18}px rgba(56, 189, 248, ${0.55 + audioLevel * 0.35}))`
		: 'drop-shadow(0 0 1px rgba(255, 255, 255, 0.35)) drop-shadow(0 4px 14px rgba(14, 165, 233, 0.3))';

	return (
		<div
			className={`relative flex items-center justify-center select-none ${className}`}
			style={{
				width: `${size}px`,
				height: `${size * 0.57}px`,
			}}
		>
			{/* Ambient radial backlight illuminating the dark and blue logo loops */}
			<div
				className={`absolute inset-0 rounded-full blur-xl pointer-events-none transition-all duration-300 ${
					isHeartbeatActive ? 'animate-heartbeat-glow' : ''
				}`}
				style={{
					background:
						'radial-gradient(ellipse at center, rgba(56, 189, 248, 0.4) 0%, rgba(30, 58, 110, 0.25) 55%, transparent 80%)',
					transform: `scale(${dynamicScale * 1.25})`,
				}}
			/>

			{/* Exact Authentic Company Logo Asset (100% original dark & steel-blue colors) */}
			<img
				src="/company_logo_authentic.png"
				alt="Company Infinity Logo"
				className={`relative z-10 w-full h-full object-contain pointer-events-none transition-transform duration-150 ${
					isHeartbeatActive ? 'animate-heartbeat' : ''
				}`}
				style={{
					filter: dynamicFilter,
					transform: `scale(${dynamicScale})`,
				}}
			/>
		</div>
	);
};

export default InfinityLogo;
