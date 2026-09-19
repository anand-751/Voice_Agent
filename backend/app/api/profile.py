from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()


class Profile(BaseModel):
	name: str = ""
	phone: str = ""
	email: str = ""


@router.post("/store-profile")
async def store_profile(profile: Profile, request: Request):
	request.app.state.bookings.save_profile(
		name=profile.name, phone=profile.phone, email=profile.email)
	return {"ok": True}
