"""Gemini or OpenAI calls for slots, menus, reviews, and follow-ups.

Keeps provider choice behind one parse/text interface so the orchestrator stays API-agnostic.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from agent.models import (
    MenuItem,
    PlaceReview,
    RestaurantDossier,
    ReviewInsight,
    SlotExtraction,
    Slots,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_RETRY_ATTEMPTS = 5

_retry_hook: Callable[[int, int, int], None] | None = None


def set_retry_hook(hook: Callable[[int, int, int], None] | None) -> None:
    global _retry_hook
    _retry_hook = hook


def provider() -> str:
    return (os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini")


def require_llm_ready() -> None:
    chosen = provider()
    if chosen == "openai":
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is missing.")
        return
    if chosen == "gemini":
        if not os.getenv("GEMINI_API_KEY", "").strip():
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is missing. "
                "Use an AI Studio key, not GOOGLE_PLACES_API_KEY."
            )
        return
    raise RuntimeError(f"Unknown LLM_PROVIDER={chosen}. Use gemini or openai.")


def _openai() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())


def _parse_openai(model: type[BaseModel], system: str, user: str) -> Any:
    completion = _openai().beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=model,
        temperature=0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed object.")
    return parsed


def _text_openai(system: str, user: str) -> str:
    completion = _openai().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""


def _gemini_client():
    from google import genai

    return genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())


def _gemini_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 503):
        return True
    text = str(exc).lower()
    return "503" in text or "429" in text or "unavailable" in text or "high demand" in text


def _gemini_generate(contents: str, config):
    client = _gemini_client()
    last: Exception | None = None
    for attempt in range(GEMINI_RETRY_ATTEMPTS):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last = exc
            if not _gemini_retryable(exc) or attempt == GEMINI_RETRY_ATTEMPTS - 1:
                raise
            wait = 2**attempt
            if _retry_hook:
                _retry_hook(attempt + 1, GEMINI_RETRY_ATTEMPTS, wait)
            time.sleep(wait)
    raise last or RuntimeError("Gemini request failed.")


def _gemini_blocked(response) -> bool:
    prompt = getattr(response, "prompt_feedback", None)
    block = getattr(prompt, "block_reason", None) if prompt else None
    if block:
        return True
    cands = getattr(response, "candidates", None) or []
    for cand in cands:
        finish = getattr(cand, "finish_reason", None)
        if str(finish) in ("SAFETY", "FinishReason.SAFETY", "2"):
            return True
    return False


def _parse_gemini(model: type[BaseModel], system: str, user: str) -> Any:
    from google.genai import types

    payload = user
    last_err: Exception | str | None = None
    config = types.GenerateContentConfig(
        system_instruction=system + " Respond with JSON only.",
        temperature=0,
        response_mime_type="application/json",
        response_schema=model,
    )
    for _ in range(2):
        response = _gemini_generate(payload, config)
        if _gemini_blocked(response):
            last_err = "Gemini safety filter blocked the response."
            continue
        text = getattr(response, "text", None) or ""
        if not text.strip():
            last_err = "Gemini returned an empty response."
            continue
        try:
            return model.model_validate_json(text)
        except ValidationError as exc:
            last_err = exc
            payload = payload + "\nFix the JSON to match the schema exactly."
    raise RuntimeError(f"Gemini JSON parse failed: {last_err}")


def _text_gemini(system: str, user: str) -> str:
    from google.genai import types

    response = _gemini_generate(
        user,
        types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
        ),
    )
    if _gemini_blocked(response):
        return ""
    return getattr(response, "text", None) or ""


def _parse(model: type[BaseModel], system: str, user: str) -> Any:
    require_llm_ready()
    if provider() == "openai":
        return _parse_openai(model, system, user)
    return _parse_gemini(model, system, user)


def _text(system: str, user: str) -> str:
    require_llm_ready()
    if provider() == "openai":
        return _text_openai(system, user)
    return _text_gemini(system, user)


def extract_slots(user_message: str, current: Slots, has_search: bool) -> SlotExtraction:
    system = (
        "Extract travel dining slots from the user. Do not invent values. "
        "Only fill a field if the user clearly stated it. "
        "meal must be breakfast, lunch, or dinner — otherwise leave it null. "
        "diet is the user's stated diet: veg, vegetarian, vegan, nonveg, non-veg, "
        "non-vegetarian, halal, or their own words. veg means vegetarian. "
        "nonveg, non-veg, and non-vegetarian are complete diets; "
        "set diet to 'non-vegetarian' and diet_unclear false. "
        "If they said 'no meat' but vegan vs vegetarian is unclear, set diet_unclear true "
        "and leave diet null. "
        "If they hinted near me / this area without a place, set neighborhood_hinted_but_vague. "
        "If they hinted cheap/expensive without a bound, set budget_hinted_but_vague. "
        "is_follow_up is true when they are refining an existing restaurant list. "
        "wants_new_search is true when they change city or ask to search again. "
        "follow_up_unclear is true when a follow-up cannot be acted on without a question. "
        "If follow_up_unclear, write one short follow_up_question."
    )
    user = json.dumps(
        {
            "current_slots": current.model_dump(),
            "has_existing_search": has_search,
            "message": user_message,
        }
    )
    return _parse(SlotExtraction, system, user)


def clarify_question(
    slots: Slots,
    missing: list[str],
    extras: list[str],
) -> str:
    system = (
        "You help a traveller find restaurants. Ask only for the listed missing facts "
        "in one short message. Do not assume a city, meal, or diet. Do not greet at length."
    )
    user = json.dumps(
        {"missing_required": missing, "optional_if_they_want": extras, "known": slots.model_dump()}
    )
    text = _text(system, user).strip()
    return text or "Please share city, meal (breakfast/lunch/dinner), and diet."


class MenuUrlPick(BaseModel):
    url: Optional[str] = None


def pick_food_menu_url(
    homepage_url: str, links: list[tuple[str, str]], meal: str = ""
) -> str | None:
    if not links:
        return None
    payload = [{"label": label or "(no label)", "url": url} for label, url in links[:20]]
    meal_hint = (meal or "").strip()
    system = (
        "Pick at most one URL that is the restaurant food menu (HTML or PDF). "
        "Ignore booking, shop, gallery, careers, Instagram, language switchers, and drink-only pages "
        "unless that is the only menu. If the list is a menu index (several meal or language food "
        "pages, few dish prices), pick the food page that matches the requested meal "
        "(breakfast, lunch, or dinner). If none is a menu, return url null. "
        "url must be copied exactly from the list."
    )
    try:
        picked = _parse(
            MenuUrlPick,
            system,
            json.dumps(
                {
                    "homepage": homepage_url,
                    "meal": meal_hint or None,
                    "links": payload,
                }
            ),
        )
    except RuntimeError:
        return None
    url = (picked.url or "").strip()
    allowed = {item[1] for item in links}
    if url in allowed:
        return url
    return None


class ExtractedMenu(BaseModel):
    items: list[MenuItem] = Field(default_factory=list)
    scrape_note: str = ""


def parse_menu(markdown: str, diet: str, meal: str) -> ExtractedMenu:
    system = (
        "Extract regular menu items from restaurant page markdown. "
        "Translate names and descriptions to English. Keep original_name. "
        "fits_diet is true only if the item clearly matches the user's diet; "
        "if unsure, set false. fits_meal is true if the item reasonably suits the meal. "
        "Include prices when present. Ignore drinks unless they are a meal. "
        "Cap at 16 items that best fit diet and meal. If the page is not a menu, "
        "return few/no items and explain in scrape_note."
    )
    user = json.dumps(
        {
            "diet": diet,
            "meal": meal,
            "markdown": markdown[:12000],
        }
    )
    try:
        return _parse(ExtractedMenu, system, user)
    except RuntimeError as exc:
        return ExtractedMenu(scrape_note=str(exc))


class RestaurantAnalysis(BaseModel):
    why: str = ""
    suggested_dishes: list[str] = Field(default_factory=list)
    reviews_positive: list[ReviewInsight] = Field(default_factory=list)
    reviews_negative: list[ReviewInsight] = Field(default_factory=list)
    context_reviews: list[ReviewInsight] = Field(default_factory=list)


def analyze_restaurant(
    place_name: str,
    slots: Slots,
    items: list[MenuItem],
    reviews: list[PlaceReview],
) -> RestaurantAnalysis:
    system = (
        "Recommend this restaurant only from the given menu items and Google reviews. "
        "Split reviews into positive and negative (use mixed if needed). "
        "context_reviews must quote reviews that speak to the user's meal, diet, or notes. "
        "why is 2-3 sentences. suggested_dishes are item names from the menu list. "
        "Do not invent reviews or prices."
    )
    user = json.dumps(
        {
            "place": place_name,
            "slots": slots.model_dump(),
            "items": [i.model_dump() for i in items],
            "reviews": [r.model_dump() for r in reviews],
        }
    )
    try:
        return _parse(RestaurantAnalysis, system, user)
    except RuntimeError:
        dishes = [i.translated_name for i in items if i.fits_diet][:5]
        return RestaurantAnalysis(
            why=(
                f"{place_name} is in the shortlist, but review synthesis was skipped "
                "(model returned empty or blocked output)."
            ),
            suggested_dishes=dishes,
        )


def follow_up_answer(user_message: str, slots: Slots, dossiers: list[RestaurantDossier]) -> str:
    system = (
        "Answer using only the cached restaurant dossiers. If you cannot, ask one clarifying question. "
        "Keep the reply concise. Include dish names and prices when you cite items."
    )
    payload = []
    for d in dossiers:
        payload.append(
            {
                "name": d.place.name,
                "address": d.place.address,
                "rating": d.place.rating,
                "website": d.homepage_url or d.place.website_uri,
                "menu_url": d.menu_url,
                "why": d.why,
                "items": [i.model_dump() for i in d.items[:12]],
                "suggested_dishes": d.suggested_dishes,
            }
        )
    user = json.dumps({"slots": slots.model_dump(), "question": user_message, "restaurants": payload})
    text = _text(system, user).strip()
    return text or "I could not answer from the cached menus. Could you rephrase the follow-up?"
