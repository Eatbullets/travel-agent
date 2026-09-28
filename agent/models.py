"""Shared Pydantic types for slots, traces, menus, and recommendations.

Gives Places, Firecrawl, the LLM, and Streamlit a single shape for session data.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Meal = Literal["breakfast", "lunch", "dinner"]


class Slots(BaseModel):
    city: Optional[str] = None
    meal: Optional[Meal] = None
    diet: Optional[str] = None
    neighborhood: Optional[str] = None
    budget: Optional[str] = None
    notes: Optional[str] = None


class SlotExtraction(BaseModel):
    city: Optional[str] = None
    meal: Optional[Meal] = None
    diet: Optional[str] = None
    diet_unclear: bool = False
    neighborhood: Optional[str] = None
    neighborhood_hinted_but_vague: bool = False
    budget: Optional[str] = None
    budget_hinted_but_vague: bool = False
    notes: Optional[str] = None
    wants_new_search: bool = False
    is_follow_up: bool = False
    follow_up_unclear: bool = False
    follow_up_question: Optional[str] = None


class TraceEvent(BaseModel):
    title: str
    detail: str
    used_session_memory: bool = False


class PlaceReview(BaseModel):
    rating: Optional[float] = None
    text: str = ""
    author: str = ""


class PlaceCandidate(BaseModel):
    place_id: str
    name: str
    address: str = ""
    rating: Optional[float] = None
    user_rating_count: int = 0
    price_level: Optional[str] = None
    website_uri: Optional[str] = None
    google_maps_uri: Optional[str] = None
    serves_vegetarian_food: Optional[bool] = None
    serves_breakfast: Optional[bool] = None
    serves_lunch: Optional[bool] = None
    serves_dinner: Optional[bool] = None
    business_status: Optional[str] = None
    reviews: list[PlaceReview] = Field(default_factory=list)
    score: float = 0.0


class MenuItem(BaseModel):
    original_name: str
    translated_name: str
    price: Optional[str] = None
    section: Optional[str] = None
    description: Optional[str] = None
    fits_diet: bool = True
    fits_meal: bool = True


class ReviewInsight(BaseModel):
    polarity: Literal["positive", "negative", "mixed"]
    quote: str
    relevance: str


class RestaurantDossier(BaseModel):
    place: PlaceCandidate
    homepage_url: Optional[str] = None
    menu_url: Optional[str] = None
    homepage_markdown: str = ""
    menu_markdown: str = ""
    scrape_notes: str = ""
    items: list[MenuItem] = Field(default_factory=list)
    reviews_positive: list[ReviewInsight] = Field(default_factory=list)
    reviews_negative: list[ReviewInsight] = Field(default_factory=list)
    context_reviews: list[ReviewInsight] = Field(default_factory=list)
    why: str = ""
    suggested_dishes: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    query_used: str
    shortlist_rule: str
    restaurants: list[RestaurantDossier] = Field(default_factory=list)
    traces: list[TraceEvent] = Field(default_factory=list)
    skipped_optional: list[str] = Field(default_factory=list)


class TurnResult(BaseModel):
    kind: Literal["clarify", "follow_up_question", "search", "follow_up"]
    assistant_message: str
    search: Optional[SearchResult] = None
    used_session_memory: bool = False
