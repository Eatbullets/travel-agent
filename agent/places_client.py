"""Google Places API (New) client.

Text Search finds restaurants; Place Details loads website, flags, and review samples.
"""

from __future__ import annotations

import os

import httpx

from agent.models import PlaceCandidate, PlaceReview

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"

SEARCH_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.websiteUri",
        "places.googleMapsUri",
        "places.servesVegetarianFood",
        "places.servesBreakfast",
        "places.servesLunch",
        "places.servesDinner",
        "places.businessStatus",
    ]
)

DETAILS_MASK = ",".join(
    [
        "id",
        "displayName",
        "formattedAddress",
        "rating",
        "userRatingCount",
        "priceLevel",
        "websiteUri",
        "googleMapsUri",
        "servesVegetarianFood",
        "servesBreakfast",
        "servesLunch",
        "servesDinner",
        "businessStatus",
        "reviews",
    ]
)


def _api_key() -> str:
    key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GOOGLE_PLACES_API_KEY is missing.")
    return key


def _text(value) -> str:
    if isinstance(value, dict):
        return value.get("text") or ""
    return value or ""


def _parse_place(raw: dict) -> PlaceCandidate:
    reviews: list[PlaceReview] = []
    for item in raw.get("reviews") or []:
        text = _text(item.get("text"))
        author = ""
        attr = item.get("authorAttribution") or {}
        author = attr.get("displayName") or ""
        reviews.append(
            PlaceReview(
                rating=item.get("rating"),
                text=text,
                author=author,
            )
        )
    return PlaceCandidate(
        place_id=raw.get("id") or "",
        name=_text(raw.get("displayName")) or "Unknown",
        address=raw.get("formattedAddress") or "",
        rating=raw.get("rating"),
        user_rating_count=int(raw.get("userRatingCount") or 0),
        price_level=raw.get("priceLevel"),
        website_uri=raw.get("websiteUri"),
        google_maps_uri=raw.get("googleMapsUri"),
        serves_vegetarian_food=raw.get("servesVegetarianFood"),
        serves_breakfast=raw.get("servesBreakfast"),
        serves_lunch=raw.get("servesLunch"),
        serves_dinner=raw.get("servesDinner"),
        business_status=raw.get("businessStatus"),
        reviews=reviews,
    )


def text_search(query: str, max_results: int = 12) -> list[PlaceCandidate]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _api_key(),
        "X-Goog-FieldMask": SEARCH_MASK,
    }
    body = {
        "textQuery": query,
        "includedType": "restaurant",
        "maxResultCount": max_results,
        "languageCode": "en",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(SEARCH_URL, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
    return [_parse_place(p) for p in data.get("places") or []]


def place_details(place_id: str) -> PlaceCandidate:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _api_key(),
        "X-Goog-FieldMask": DETAILS_MASK,
    }
    url = DETAILS_URL.format(place_id=place_id)
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    return _parse_place(data)
