"""TOON (Token-Oriented Object Notation) encoder.

A compact, token-efficient serialization format designed for LLM prompts.
Replaces verbose JSON syntax ({}, quotes, repeated keys) with tabular CSV-style
headers and YAML-like key-value pairs, reducing prompt tokens by 30%–60% while
remaining lossless and fully human-readable.
"""

from typing import Any, Dict, List


def _format_scalar(val: Any) -> str:
	if val is None:
		return "null"
	if isinstance(val, bool):
		return "true" if val else "false"
	return str(val).strip()


def encode_toon(data: Any, indent: int = 0) -> str:
	"""Encode arbitrary Python data structures into TOON format."""
	prefix = "  " * indent

	if data is None:
		return f"{prefix}null"

	if isinstance(data, (str, int, float, bool)):
		return f"{prefix}{_format_scalar(data)}"

	# 1. Dictionary / Mapping
	if isinstance(data, dict):
		if not data:
			return f"{prefix}{{}}"

		lines = []
		for k, v in data.items():
			# List of primitives (e.g. free_slots: ["10:00", "10:30", "11:00"])
			if isinstance(v, list) and (not v or all(isinstance(x, (str, int, float, bool)) for x in v)):
				items_str = ", ".join(_format_scalar(x) for x in v)
				lines.append(f"{prefix}{k}[{len(v)}]: {items_str}")

			# List of uniform dictionaries (tabular data)
			elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
				# Check if all items share the same keys
				first_keys = list(v[0].keys())
				is_uniform = all(list(x.keys()) == first_keys for x in v) and all(
					all(not isinstance(val, (dict, list)) for val in x.values()) for x in v
				)
				if is_uniform and first_keys:
					cols_header = ",".join(first_keys)
					lines.append(f"{prefix}{k}[{len(v)}]{{{cols_header}}}:")
					for row in v:
						row_vals = ",".join(_format_scalar(row.get(col, "")) for col in first_keys)
						lines.append(f"{prefix}  {row_vals}")
				else:
					# Non-uniform list of dicts
					lines.append(f"{prefix}{k}[{len(v)}]:")
					for item in v:
						lines.append(encode_toon(item, indent + 1))

			# Nested Dictionary
			elif isinstance(v, dict):
				lines.append(f"{prefix}{k}:")
				lines.append(encode_toon(v, indent + 1))

			# Scalar Value
			else:
				lines.append(f"{prefix}{k}: {_format_scalar(v)}")

		return "\n".join(lines)

	# 2. List / Sequence
	if isinstance(data, list):
		if not data:
			return f"{prefix}[]"

		# Primitives list
		if all(isinstance(x, (str, int, float, bool)) for x in data):
			items_str = ", ".join(_format_scalar(x) for x in data)
			return f"{prefix}[{len(data)}]: {items_str}"

		# Uniform dicts list
		if all(isinstance(x, dict) for x in data):
			first_keys = list(data[0].keys())
			is_uniform = all(list(x.keys()) == first_keys for x in data) and all(
				all(not isinstance(val, (dict, list)) for val in x.values()) for x in data
			)
			if is_uniform and first_keys:
				cols_header = ",".join(first_keys)
				lines = [f"{prefix}[{len(data)}]{{{cols_header}}}:"]
				for row in data:
					row_vals = ",".join(_format_scalar(row.get(col, "")) for col in first_keys)
					lines.append(f"{prefix}  {row_vals}")
				return "\n".join(lines)

		# Heterogeneous list
		lines = [f"{prefix}[{len(data)}]:"]
		for item in data:
			lines.append(encode_toon(item, indent + 1))
		return "\n".join(lines)

	return f"{prefix}{_format_scalar(data)}"
