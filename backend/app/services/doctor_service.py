"""Doctor service managing clinic specialists, schedules, and clinical problem matching."""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo


@dataclass
class Doctor:
	name: str
	title: str
	specialty: str
	description: str
	working_days: List[str]  # e.g. ["Tuesday", "Thursday", "Saturday"]
	working_days_str: str
	keywords: List[str] = field(default_factory=list)


# Official clinic specialists from Bright Dental Clinic roster
DOCTORS: List[Doctor] = [
	Doctor(
		name="Dr. Ananya Sharma",
		title="BDS MDS (Endodontics)",
		specialty="Endodontics & Cosmetic Dentistry",
		description="Specialist in root canal treatments (RCT), tooth pain relief, cosmetic whitening, and dental restorations. 12 years of experience.",
		working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
		working_days_str="Monday through Saturday (10:00 AM – 7:00 PM)",
		keywords=[
			"root canal", "rct", "toothache", "tooth pain", "tooth ache", "gum pain",
			"cavity", "cavities", "filling", "fillings", "decay", "swelling",
			"cleaning", "scaling", "polishing", "teeth whitening", "whitening",
			"crown", "crowns", "cap", "veneer", "veneers", "checkup", "consultation",
			"general", "sensitivity", "broken tooth", "chipped tooth",
		],
	),
	Doctor(
		name="Dr. Rohit Verma",
		title="BDS MDS (Orthodontics)",
		specialty="Orthodontics",
		description="Specialist in teeth alignment, braces (metal & ceramic), clear aligners (Invisalign), and bite correction. Visits on Tuesday, Thursday, and Saturday.",
		working_days=["Tuesday", "Thursday", "Saturday"],
		working_days_str="Tuesday, Thursday, and Saturday (10:00 AM – 7:00 PM)",
		keywords=[
			"braces", "aligner", "aligners", "clear aligners", "invisalign",
			"crooked", "crooked teeth", "gap", "gaps", "spacing", "overlapping",
			"teeth alignment", "straighten", "straightening", "bite", "overbite",
			"underbite", "jaw alignment", "orthodontic", "orthodontics", "ortho",
		],
	),
]


def get_all_doctors() -> List[Doctor]:
	return DOCTORS


def find_doctor_by_name(query: str) -> Optional[Doctor]:
	"""Look up doctor by distinctive name, full name, or specialty."""
	if not query:
		return None
	q = query.strip().lower()

	# 1. Exact full name match
	for doc in DOCTORS:
		if q == doc.name.lower() or doc.name.lower() in q:
			return doc

	# 2. Distinctive name parts (exclude titles like 'dr', 'dr.', 'doctor', 'bds', 'mds')
	ignore_words = {"dr", "dr.", "doctor", "bds", "mds"}
	for doc in DOCTORS:
		name_parts = [p for p in doc.name.lower().split() if p not in ignore_words]
		for part in name_parts:
			if re.search(rf"\b{re.escape(part)}\b", q):
				return doc

	# 3. Specialty match
	for doc in DOCTORS:
		if q in doc.specialty.lower() or doc.specialty.lower() in q:
			return doc
		if "ortho" in q and "orthodontics" in doc.specialty.lower():
			return doc
		if ("endo" in q or "cosmetic" in q) and "endodontics" in doc.specialty.lower():
			return doc

	return None


def recommend_doctor_by_problem(problem_text: str) -> Optional[Doctor]:
	"""Match user complaint or inquiry to the best specialist."""
	if not problem_text:
		return None
	text = problem_text.lower()

	# Check Orthodontics keywords first (more specific)
	ortho_doc = DOCTORS[1]  # Dr. Rohit Verma
	for kw in ortho_doc.keywords:
		if kw in text:
			return ortho_doc

	# Check Endodontics keywords
	endo_doc = DOCTORS[0]  # Dr. Ananya Sharma
	for kw in endo_doc.keywords:
		if kw in text:
			return endo_doc

	return None


def is_doctor_available_on_date(doctor: Doctor, date_str: str) -> tuple[bool, str]:
	"""Check if the doctor practices on the day corresponding to date_str (YYYY-MM-DD)."""
	try:
		dt = date.fromisoformat(date_str)
		day_name = dt.strftime("%A")  # "Monday", "Tuesday", etc.
		return (day_name in doctor.working_days, day_name)
	except Exception:
		return (True, "")


def get_upcoming_available_dates(
	doctor: Doctor, from_date_str: str, max_dates: int = 3
) -> List[dict]:
	"""Find the nearest upcoming dates on which the doctor is practicing in the clinic."""
	try:
		start = date.fromisoformat(from_date_str)
	except Exception:
		start = date.today()

	available = []
	for offset in range(0, 14):
		candidate = start + timedelta(days=offset)
		day_name = candidate.strftime("%A")
		if day_name in doctor.working_days and day_name != "Sunday":
			available.append({
				"date": candidate.isoformat(),
				"day": day_name,
				"formatted": f"{day_name}, {candidate.strftime('%b %d')}",
			})
			if len(available) >= max_dates:
				break
	return available
