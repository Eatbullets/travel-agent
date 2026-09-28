"""Turn handler: clarify, search, or reuse session memory.

Wires Places, Firecrawl, and the LLM and emits the pipeline traces shown in the UI.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from agent import firecrawl_client, llm, places_client
from agent.models import (
    PlaceCandidate,
    RestaurantDossier,
    SearchResult,
    Slots,
    TraceEvent,
    TurnResult,
)

SHORTLIST_N = 3
SHORTLIST_RULE = (
    "Drop permanently closed places. Prefer Places flags that match the stated diet and meal. "
    "Rank by rating * log(1 + review count). Keep the top 3 for menu scraping."
)


def _md_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _md_section(heading: str, bullets: list[str]) -> str:
    body = _md_list(bullets)
    if body:
        return f"**{heading}**\n\n{body}"
    return f"**{heading}**"


def _md_sections(*parts: tuple[str, list[str]]) -> str:
    return "\n\n".join(_md_section(heading, bullets) for heading, bullets in parts)


def _llm_label() -> str:
    return "OpenAI" if llm.provider() == "openai" else "Gemini"


@dataclass
class SessionMemory:
    slots: Slots = field(default_factory=Slots)
    search: SearchResult | None = None
    traces: list[TraceEvent] = field(default_factory=list)


def missing_required(slots: Slots) -> list[str]:
    missing: list[str] = []
    if not slots.city:
        missing.append("city")
    if not slots.meal:
        missing.append("meal (breakfast, lunch, or dinner)")
    if not slots.diet:
        missing.append("diet (vegetarian, vegan, or other)")
    return missing


def merge_slots(current: Slots, extracted) -> Slots:
    data = current.model_dump()
    for key in ("city", "meal", "diet", "neighborhood", "budget", "notes"):
        value = getattr(extracted, key, None)
        if value:
            data[key] = value
    return Slots(**data)


_NON_VEG_RE = re.compile(r"\bnon[-\s]?veg(?:etarian)?\b", re.I)
_VEGAN_RE = re.compile(r"\bvegan\b", re.I)
_VEGETARIAN_RE = re.compile(r"\bvegetarian\b|\bveg\b", re.I)


def _wants_vegetarian_boost(diet: str | None) -> bool:
    if not diet:
        return False
    text = diet.lower()
    if _NON_VEG_RE.search(text):
        return False
    return bool(_VEGAN_RE.search(text) or _VEGETARIAN_RE.search(text))


def diet_stated_in_text(text: str) -> str | None:
    if _NON_VEG_RE.search(text):
        return "non-vegetarian"
    if _VEGAN_RE.search(text):
        return "vegan"
    if _VEGETARIAN_RE.search(text):
        return "vegetarian"
    return None


def _score(place: PlaceCandidate, slots: Slots) -> float:
    rating = place.rating or 0.0
    count = place.user_rating_count or 0
    score = rating * math.log(1 + count)
    if _wants_vegetarian_boost(slots.diet) and place.serves_vegetarian_food:
        score += 2.0
    meal_flag = {
        "breakfast": place.serves_breakfast,
        "lunch": place.serves_lunch,
        "dinner": place.serves_dinner,
    }.get(slots.meal or "")
    if meal_flag:
        score += 1.5
    return score


def shortlist(places: list[PlaceCandidate], slots: Slots) -> list[PlaceCandidate]:
    open_places = [
        p
        for p in places
        if (p.business_status or "OPERATIONAL") != "CLOSED_PERMANENTLY"
    ]
    for place in open_places:
        place.score = _score(place, slots)
    open_places.sort(key=lambda p: p.score, reverse=True)
    return open_places[:SHORTLIST_N]


def _places_query(slots: Slots) -> str:
    parts = [slots.diet or "restaurant", "restaurants in", slots.city or ""]
    if slots.neighborhood:
        parts.append(slots.neighborhood)
    if slots.meal:
        parts.append(f"for {slots.meal}")
    if slots.budget:
        parts.append(slots.budget)
    return " ".join(p for p in parts if p).strip()


def _next_menu_url(
    page_url: str,
    markdown: str,
    scraped_keys: set[str],
    meal: str,
) -> tuple[str | None, str]:
    links = [
        (label, url)
        for label, url in firecrawl_client.collect_page_links(page_url, markdown)
        if firecrawl_client.page_key(url) not in scraped_keys
    ]
    pdf = firecrawl_client.first_pdf_url(links)
    if pdf:
        return pdf, "PDF link"
    picked = llm.pick_food_menu_url(page_url, links, meal=meal)
    if picked and firecrawl_client.page_key(picked) not in scraped_keys:
        return picked, "LLM picked a menu URL"
    return None, ""


def _scrape_one(place: PlaceCandidate, slots: Slots) -> RestaurantDossier:
    dossier = RestaurantDossier(place=place, homepage_url=place.website_uri)
    if not place.website_uri:
        dossier.scrape_notes = "No official website from Places; skipped Firecrawl."
        return dossier
    try:
        homepage = firecrawl_client.scrape_markdown(place.website_uri)
    except Exception as exc:
        dossier.scrape_notes = f"Homepage scrape failed: {exc}"
        return dossier
    dossier.homepage_markdown = homepage
    dossier.menu_url = place.website_uri
    dossier.menu_markdown = homepage
    if firecrawl_client.looks_like_menu(homepage):
        dossier.scrape_notes = "Menu-like content found on the homepage (one scrape)."
        return dossier

    meal = slots.meal or ""
    scraped_keys = {firecrawl_client.page_key(place.website_uri)}
    current_url = place.website_uri
    current_md = homepage
    notes: list[str] = []
    for hop in range(1, 3):
        next_url, how = _next_menu_url(current_url, current_md, scraped_keys, meal)
        if not next_url:
            if hop == 1:
                notes.append(
                    "No separate menu URL (hash-only or LLM found none); used homepage markdown."
                )
            else:
                notes.append("Index page had no further food-menu URL; kept last scrape.")
            break
        try:
            menu_md = firecrawl_client.scrape_markdown(next_url)
        except Exception as exc:
            notes.append(f"Menu scrape failed ({exc}); kept previous markdown.")
            break
        if not menu_md:
            notes.append("Menu page scrape returned empty markdown; kept previous markdown.")
            break
        dossier.menu_url = next_url
        dossier.menu_markdown = menu_md
        kind = "PDF" if firecrawl_client.is_pdf_url(next_url) else "HTML"
        ordinal = "Second" if hop == 1 else "Third"
        notes.append(f"{ordinal} scrape ({kind}): {next_url} ({how}).")
        scraped_keys.add(firecrawl_client.page_key(next_url))
        if firecrawl_client.looks_like_menu(menu_md) or firecrawl_client.is_pdf_url(next_url):
            break
        current_url = next_url
        current_md = menu_md
    dossier.scrape_notes = " ".join(notes)
    return dossier


def _enrich_dossiers(dossiers: list[RestaurantDossier], slots: Slots) -> None:
    for dossier in dossiers:
        markdown = dossier.menu_markdown or dossier.homepage_markdown
        if markdown:
            extracted = llm.parse_menu(markdown, slots.diet or "", slots.meal or "")
            dossier.items = extracted.items
            if extracted.scrape_note:
                dossier.scrape_notes = (
                    f"{dossier.scrape_notes} {extracted.scrape_note}".strip()
                )
        analysis = llm.analyze_restaurant(
            dossier.place.name,
            slots,
            dossier.items,
            dossier.place.reviews,
        )
        dossier.why = analysis.why
        dossier.suggested_dishes = analysis.suggested_dishes
        dossier.reviews_positive = analysis.reviews_positive
        dossier.reviews_negative = analysis.reviews_negative
        dossier.context_reviews = analysis.context_reviews


def run_search(slots: Slots) -> SearchResult:
    traces: list[TraceEvent] = []
    skipped: list[str] = []
    if not slots.neighborhood:
        skipped.append("No neighborhood stated; searched the city, not a district.")
    if not slots.budget:
        skipped.append("No budget stated; did not filter on price.")

    query = _places_query(slots)
    traces.append(
        TraceEvent(
            title="Identify restaurants",
            detail=_md_section(
                "Search",
                [
                    "Source: Google Places Text Search",
                    f"Query: `{query}`",
                    "Type filter: restaurant",
                ],
            ),
        )
    )
    found = places_client.text_search(query)
    traces.append(
        TraceEvent(
            title="Restaurants found (before shortlist)",
            detail=_md_section(
                f"{len(found)} restaurants returned",
                [p.name for p in found[:12]] if found else ["none"],
            ),
        )
    )
    chosen = shortlist(found, slots)
    traces.append(
        TraceEvent(
            title="Shortlist",
            detail=_md_sections(
                (
                    "How they were ranked",
                    [
                        "Drop permanently closed places",
                        "Prefer Places flags that match the stated diet and meal",
                        "Rank by rating × log(1 + review count)",
                        "Keep the top 3 for menu scraping",
                    ],
                ),
                (
                    "Selected",
                    [f"{p.name} (score {p.score:.2f})" for p in chosen] or ["none"],
                ),
            ),
        )
    )

    detailed: list[PlaceCandidate] = []
    for place in chosen:
        try:
            detailed.append(places_client.place_details(place.place_id))
        except Exception:
            detailed.append(place)
    traces.append(
        TraceEvent(
            title="Reviews",
            detail=_md_section(
                "Source",
                [
                    "Loaded Google Places reviews (typically up to 5 per restaurant)",
                    "Did not scrape Google Maps",
                ],
            ),
        )
    )

    dossiers: list[RestaurantDossier] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_scrape_one, place, slots) for place in detailed]
        for future in as_completed(futures):
            dossiers.append(future.result())
    dossiers.sort(key=lambda d: d.place.score, reverse=True)

    traces.append(
        TraceEvent(
            title="Read menus",
            detail=_md_sections(
                (
                    "How menus were read",
                    [
                        "Firecrawl markdown scrape, 2 at a time",
                        "Homepage first; up to two extra pages if a menu link is found and the last page still has few prices",
                    ],
                ),
                (
                    "Per restaurant",
                    [
                        f"{d.place.name}: {d.menu_url or 'no URL'} — {d.scrape_notes}"
                        for d in dossiers
                    ]
                    or ["none"],
                ),
            ),
        )
    )

    _enrich_dossiers(dossiers, slots)
    model_name = _llm_label()
    traces.append(
        TraceEvent(
            title="Translate and filter",
            detail=_md_section(
                "What the model did",
                [
                    f"{model_name} translated menu names to English",
                    "Kept items that fit the stated diet and meal",
                ],
            ),
        )
    )
    traces.append(
        TraceEvent(
            title="Review synthesis",
            detail=_md_section(
                "What the model did",
                [
                    f"{model_name} split Places quotes into positive and negative",
                    "Pulled quotes that match this meal, diet, or notes",
                ],
            ),
        )
    )
    traces.append(
        TraceEvent(
            title="Suggest options",
            detail=_md_section(
                "Shown on each card",
                [
                    "Dishes and prices come from the extracted regular menu",
                    "Restaurant website and menu page when found",
                ],
            ),
        )
    )

    return SearchResult(
        query_used=query,
        shortlist_rule=SHORTLIST_RULE,
        restaurants=dossiers,
        traces=traces,
        skipped_optional=skipped,
    )


def _refilter_for_meal(memory: SessionMemory) -> SearchResult:
    assert memory.search is not None
    traces = [
        TraceEvent(
            title="Session memory",
            detail=_md_section(
                "What was reused",
                [
                    "Same city restaurant set reused",
                    "No new Firecrawl scrapes",
                    "Re-filtered cached menus for the updated meal",
                ],
            ),
            used_session_memory=True,
        )
    ]
    _enrich_dossiers(memory.search.restaurants, memory.slots)
    traces.append(
        TraceEvent(
            title="Re-rank dishes",
            detail=_md_section(
                "What the model did",
                [f"{_llm_label()} re-read cached menus for the new meal"],
            ),
            used_session_memory=True,
        )
    )
    return SearchResult(
        query_used=memory.search.query_used,
        shortlist_rule=memory.search.shortlist_rule,
        restaurants=memory.search.restaurants,
        traces=traces,
        skipped_optional=memory.search.skipped_optional,
    )


def handle_turn(user_message: str, memory: SessionMemory) -> TurnResult:
    llm.require_llm_ready()
    extracted = llm.extract_slots(user_message, memory.slots, memory.search is not None)
    previous_city = memory.slots.city
    previous_meal = memory.slots.meal
    memory.slots = merge_slots(memory.slots, extracted)
    if not memory.slots.diet:
        inferred = diet_stated_in_text(user_message)
        if inferred:
            memory.slots.diet = inferred
    if extracted.diet_unclear and not memory.slots.diet:
        memory.slots.diet = None

    extras: list[str] = []
    if extracted.neighborhood_hinted_but_vague and not memory.slots.neighborhood:
        extras.append("neighborhood or landmark (optional — skip to search the whole city)")
    if extracted.budget_hinted_but_vague and not memory.slots.budget:
        extras.append("budget (optional — skip to ignore price)")

    missing = missing_required(memory.slots)
    if missing:
        question = llm.clarify_question(memory.slots, missing, extras)
        return TurnResult(kind="clarify", assistant_message=question)

    city_changed = bool(
        previous_city
        and memory.slots.city
        and previous_city.lower() != memory.slots.city.lower()
    )
    meal_changed = bool(
        previous_meal and memory.slots.meal and previous_meal != memory.slots.meal
    )

    reuse_cache = (
        memory.search is not None
        and not extracted.wants_new_search
        and not city_changed
    )
    if reuse_cache:
        if extracted.follow_up_unclear:
            q = extracted.follow_up_question or "Could you be more specific about what to change?"
            return TurnResult(
                kind="follow_up_question",
                assistant_message=q,
                used_session_memory=True,
            )
        if meal_changed:
            result = _refilter_for_meal(memory)
            memory.search = result
            return TurnResult(
                kind="follow_up",
                assistant_message=(
                    "Updated suggestions for the new meal using the restaurants "
                    "already in session memory (no new Firecrawl scrapes)."
                ),
                search=result,
                used_session_memory=True,
            )
        answer = llm.follow_up_answer(
            user_message, memory.slots, memory.search.restaurants
        )
        return TurnResult(
            kind="follow_up",
            assistant_message=answer,
            search=memory.search,
            used_session_memory=True,
        )

    result = run_search(memory.slots)
    if extras:
        result.skipped_optional = list(result.skipped_optional) + [
            "You hinted at area or budget without a specific value; searched without those filters."
        ]
    memory.search = result
    prefix = "Search complete. Pipeline steps are below."
    if result.skipped_optional:
        prefix += " " + " ".join(result.skipped_optional)
    return TurnResult(kind="search", assistant_message=prefix, search=result)
