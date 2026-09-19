from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolResult:
	"""What a tool hands back to the graph."""
	data: dict = field(default_factory=dict)    # facts for the Responder LLM
	payload_type: str = "response"              # ws message type hint
	extra: dict = field(default_factory=dict)   # structured fields for the payload
	pending_booking_id: Optional[str] = None
	clear_pending_booking: bool = False
