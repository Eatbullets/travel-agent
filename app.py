"""Streamlit UI for the traveller restaurant agent.

Loads env, runs chat, and renders pipeline traces and recommendation cards via the orchestrator.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent.llm import provider, set_retry_hook
from agent.models import RestaurantDossier, SearchResult, TraceEvent
from agent.orchestrator import SessionMemory, handle_turn

load_dotenv()

st.set_page_config(page_title="Traveller restaurant agent", layout="wide")
st.title("Traveller restaurant agent")
st.caption(
    "Name a city, a meal (breakfast / lunch / dinner), and your diet. "
    "Nothing is assumed — if something is missing, the agent asks."
)

with st.sidebar:
    st.subheader("Limitations")
    st.markdown(
        "- Places reviews are a small sample.\n"
        "- PDF / image / Instagram menus often fail.\n"
        "- Firecrawl: 3 venues, markdown only (free-tier cap).\n"
        "- Diet tagging can miss hidden animal products.\n"
        "- Memory lasts for this tab only."
    )
    st.subheader("LLM")
    st.write(provider())
    st.subheader("Keys loaded")
    st.write(
        {
            "Gemini": bool(os.getenv("GEMINI_API_KEY")),
            "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
            "Firecrawl": bool(os.getenv("FIRECRAWL_API_KEY")),
            "Places": bool(os.getenv("GOOGLE_PLACES_API_KEY")),
        }
    )


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Tell me which city you are visiting, whether this is breakfast, "
                    "lunch, or dinner, and your diet (vegetarian, vegan, or other)."
                ),
            }
        ]
    if "memory" not in st.session_state:
        st.session_state.memory = SessionMemory()
    if "last_search" not in st.session_state:
        st.session_state.last_search = None
    if "used_memory" not in st.session_state:
        st.session_state.used_memory = False


def render_traces(traces: list[TraceEvent]) -> None:
    st.subheader("How this answer was built")
    for event in traces:
        label = event.title
        if event.used_session_memory:
            label += " (session memory)"
        with st.expander(label, expanded=False):
            st.markdown(event.detail)


def render_restaurant(dossier: RestaurantDossier) -> None:
    place = dossier.place
    st.markdown(f"### {place.name}")
    meta = []
    if place.rating:
        meta.append(f"★ {place.rating} ({place.user_rating_count} ratings)")
    if place.address:
        meta.append(place.address)
    if meta:
        st.caption(" · ".join(meta))
    if dossier.why:
        st.write(dossier.why)
    if dossier.suggested_dishes:
        st.write("**Suggested dishes:** " + ", ".join(dossier.suggested_dishes))

    diet_items = [i for i in dossier.items if i.fits_diet]
    show = diet_items or dossier.items
    if show:
        st.write("**Menu items and prices**")
        rows = []
        for item in show[:10]:
            rows.append(
                {
                    "Item": item.translated_name,
                    "Original": item.original_name,
                    "Price": item.price or "—",
                    "Section": item.section or "—",
                    "Fits meal": "yes" if item.fits_meal else "no",
                }
            )
        st.table(rows)
    else:
        st.info(dossier.scrape_notes or "No menu items extracted.")

    cols = st.columns(3)
    website = dossier.homepage_url or place.website_uri
    if website:
        cols[0].markdown(f"[Restaurant website]({website})")
    if dossier.menu_url:
        cols[1].markdown(f"[Menu page]({dossier.menu_url})")
    if place.google_maps_uri:
        cols[2].markdown(f"[Google Maps]({place.google_maps_uri})")

    with st.expander("Reviews"):
        st.write("**Positive**")
        for r in dossier.reviews_positive[:3]:
            st.write(f"- ({r.polarity}) {r.quote}")
        st.write("**Negative**")
        for r in dossier.reviews_negative[:3]:
            st.write(f"- ({r.polarity}) {r.quote}")
        st.write("**Matching this meal / context**")
        for r in dossier.context_reviews[:3]:
            st.write(f"- {r.quote} — {r.relevance}")
        if not dossier.reviews_positive and not dossier.reviews_negative:
            st.write("No review text returned by Places for this restaurant.")


def render_search(result: SearchResult, used_memory: bool) -> None:
    if used_memory:
        st.info("This turn used session memory (no new Firecrawl scrapes).")
    if result.skipped_optional:
        for note in result.skipped_optional:
            st.caption(note)
    render_traces(result.traces)
    st.subheader("Suggestions")
    for dossier in result.restaurants:
        with st.container(border=True):
            render_restaurant(dossier)


_init_state()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("City, meal, diet, or a follow-up…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=True)
        status.write("Calling the model…")

        def on_retry(n: int, total: int, wait: int) -> None:
            status.write(
                f"Gemini is busy (high demand / 503). Retry {n}/{total} in {wait}s…"
            )
            status.update(label=f"Retrying Gemini ({n}/{total})…", state="running")

        set_retry_hook(on_retry)
        try:
            turn = handle_turn(prompt, st.session_state.memory)
            status.update(label="Done", state="complete")
        except Exception:
            status.update(label="Failed", state="error")
            raise
        finally:
            set_retry_hook(None)
        st.write(turn.assistant_message)
        st.session_state.messages.append(
            {"role": "assistant", "content": turn.assistant_message}
        )
        if turn.search:
            st.session_state.last_search = turn.search
        st.session_state.used_memory = turn.used_session_memory

if st.session_state.last_search:
    render_search(st.session_state.last_search, st.session_state.used_memory)
