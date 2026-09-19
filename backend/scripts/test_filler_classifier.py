"""Verification suite for Context-Aware Latency Filler Classifier."""

import asyncio
import sys
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm import LLMService
from app.config import get_settings


def test_contextual_filler_classifier():
	llm = LLMService()
	settings = get_settings()

	test_cases = [
		# 1. Chit-chat, greetings, pleasantries -> MUST BE None (NO filler audio, answers immediately!)
		("Aap kaisi hain? Main theek hoon, aap bataiye.", None),
		("Haan ji haan ji", None),
		("theek hai", None),
		("theek", None),
		("Namaste!", None),
		("Hello kaise ho?", None),
		("Shukriya ji", None),
		("dhanyavaad", None),
		("ok", None),
		("हेलो निया, कैसी हो तुम? आप बेसिकली मुझे थोड़ा हेडेक हो रहा है, तो आप हेल्प कर सकते हैं?", None),

		# 2. Payment verification -> MUST BE 'payment'
		("पेमेंट होगी।", "payment"),
		("maine payment kar di hai", "payment"),
		("paid", "payment"),
		("fees transfer kar di", "payment"),
		("गूगल पे कर दिया है", "payment"),

		# 3. Slot checking / booking -> MUST BE 'slots'
		("Kal Monday ko slots available hain kya?", "slots"),
		("10:00 AM ka time available hai?", "slots"),
		("theek hai book kar dijiye", "slots"),
		("ओके, रोहित वर्मा के साथ कमिंग ट्यूसडे पे आप मेरा बुक कीजिए स्लॉट।", "slots"),
		("मुझे 5 पीएम का चाहिए वैसे तो।", "slots"),
		("5 बजे का चाहिए मुझे कल शाम।", "slots"),

		# 4. Clinic / Doctor / Treatment info -> MUST BE 'info'
		("Mujhe bleeding ho rahi hai teeth mein", "info"),
		("Doctor Rohit kab milenge?", "info"),
		("Doctor Ananya ke baare mein batao", "info"),
		("Cleaning ka kitna charge hai?", "info"),
		("Clinic ka address kahan par hai?", "info"),
		("Toothache aur bleeding gums ka ilaj", "info"),
		("ओके, ठीक है। डेंटल केयर से रिलेटेड मुझे टीथ वाइटनिंग करवाना है, तो उसके लिए आप हेल्प कीजिए मेरी।", "info"),
		("ओके, मुझे ब्रेसेस भी ट्राई बेसिकली लगवाने हैं, तो उसके लिए भी आप हेल्प करें, बताइए कि क्या बेटर रहेगा।", "info"),

		# 5. Irritation & Anger -> MUST BE None (NO delay filler audio, de-escalate immediately!)
		("निया तुम मुझे बहुत इरिटेटिंग लगी, क्या तुम अच्छे से रिप्लाई नहीं कर सकती हो?", None),
		("Tum mujhe irritate kar rahi ho, achhe se baat karo", None),

		# 6. Short thinking / clarifications -> MUST BE 'short'
		("Ek sawaal poochna tha", "short"),
		("Mujhe ek query hai", "short"),
	]

	async def _run():
		print("\n--- Testing Context-Aware Filler Intent Classification ---")
		passed = 0
		for text, expected in test_cases:
			intent = await llm.classify_filler_intent(text)
			print(f"Utterance: \"{text}\" -> Intent: {intent} (expected: {expected})")
			assert intent == expected, f"Failed for '{text}': got {intent}, expected {expected}"
			passed += 1

		print(f"\n✓ All {passed}/{len(test_cases)} filler classification test cases passed successfully!")

		# 6. Test Fallback with Groq client None
		print("\n--- Testing Rigid Regex Fallback Mode (Groq Disabled) ---")
		orig_client = llm._client
		try:
			llm._client = None
			fallback_cases = [
				("Aap kaisi hain?", None),
				("Haan ji", None),
				("पेमेंट होगी", "payment"),
				("Kal ke slots kya hain?", "slots"),
				("Bleeding gums ki problem hai", "info"),
				("Doctor Rohit kab aate hain?", "info"),
				("Ek sawaal tha?", "short"),
			]
			for text, expected in fallback_cases:
				intent = await llm.classify_filler_intent(text)
				print(f"[Fallback] \"{text}\" -> Intent: {intent} (expected: {expected})")
				assert intent == expected, f"Fallback failed for '{text}': got {intent}, expected {expected}"
			print(f"✓ All {len(fallback_cases)} fallback regex checks passed successfully!")
		finally:
			llm._client = orig_client

	asyncio.run(_run())


if __name__ == "__main__":
	test_contextual_filler_classifier()
