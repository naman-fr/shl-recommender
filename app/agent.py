"""
Conversational Agent with Dialogue State Tracking and Policy Layer.

Architecture follows conversational recommender system best practices:
1. Dialogue State Extractor — turns conversation history into structured slots
2. Policy Layer — decides whether to CLARIFY, RECOMMEND, REFINE, COMPARE, or REFUSE
3. Retrieval Orchestrator — calls HybridRetriever with extracted slots
4. Response Generator — produces grounded, catalog-only responses

Key design decisions:
- Explicit dialogue policy prevents the LLM from improvising unsafely
- All recommendations are verified against the catalog before returning (grounding)
- Conversation context is maintained per session (max 20 turns)
- The LLM's job is limited to extracting intent + generating natural-language from evidence
"""

import json
import os
import re
import logging
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from app.retriever import HybridRetriever
from app.models import AssessmentCard

logger = logging.getLogger(__name__)


class Intent(Enum):
    GREETING = "greeting"
    RECOMMEND = "recommend"
    CLARIFY_RESPONSE = "clarify_response"
    REFINE = "refine"
    COMPARE = "compare"
    DETAIL = "detail"
    HELP = "help"
    OFF_TOPIC = "off_topic"


class Action(Enum):
    ASK_CLARIFY = "ask_clarify"
    RETRIEVE_AND_RESPOND = "retrieve_and_respond"
    REFINE_RESULTS = "refine_results"
    COMPARE_ASSESSMENTS = "compare_assessments"
    SHOW_DETAIL = "show_detail"
    GREET = "greet"
    SHOW_HELP = "show_help"
    REFUSE = "refuse"


@dataclass
class DialogueSlots:
    """Structured representation of user requirements extracted from conversation."""
    role: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    job_level: Optional[str] = None
    max_duration: Optional[int] = None
    remote_only: bool = False
    adaptive_only: bool = False
    category: Optional[str] = None
    language: Optional[str] = None
    compare_names: list[str] = field(default_factory=list)
    detail_name: Optional[str] = None
    raw_query: str = ""

    def has_enough_info(self) -> bool:
        """Check if we have enough slots filled to make recommendations."""
        return bool(self.role or self.skills or self.raw_query or self.category
                     or self.job_level)

    def to_search_query(self) -> str:
        parts = []
        if self.role:
            parts.append(self.role)
        if self.skills:
            parts.extend(self.skills)
        if self.raw_query:
            parts.append(self.raw_query)
        return " ".join(parts)


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class Session:
    session_id: str
    history: list[ConversationTurn] = field(default_factory=list)
    slots: DialogueSlots = field(default_factory=DialogueSlots)
    last_recommendations: list[dict] = field(default_factory=list)
    turn_count: int = 0

    def add_turn(self, role: str, content: str):
        self.history.append(ConversationTurn(role=role, content=content))
        if len(self.history) > 20:
            self.history = self.history[-20:]
        self.turn_count += 1

    def get_history_text(self) -> str:
        return "\n".join(
            f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}"
            for t in self.history
        )


# ── System prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the SHL Assessment Recommender, an expert AI assistant that helps HR professionals find the right SHL assessments.

CRITICAL RULES:
1. ONLY recommend assessments from the retrieved catalog results provided to you.
2. NEVER invent assessment names, URLs, or details not in the catalog.
3. Every recommendation MUST include the exact name and URL from the catalog.
4. If the user's query is vague, ask 1-2 clarifying questions.
5. If the query is off-topic (not about SHL assessments), politely redirect.
6. When comparing, use ONLY catalog data fields (description, duration, categories, etc.).

RESPONSE FORMAT for recommendations:
- Brief summary of what you found
- For each assessment: Name (as clickable link), what it measures, duration
- Ask if they want to refine or see alternatives

Available categories: Knowledge & Skills, Personality & Behavior, Simulations, Ability & Aptitude, Competencies, Biodata & Situational Judgment, Development & 360, Assessment Exercises
Available job levels: Entry-Level, Graduate, Mid-Professional, Professional Individual Contributor, Front Line Manager, Supervisor, Manager, Director, Executive, General Population"""


class SHLAgent:
    """Main agent with dialogue state tracking and policy-driven responses."""

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or HybridRetriever()
        self.sessions: dict[str, Session] = {}
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and api_key != "your_gemini_api_key_here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._llm = genai.GenerativeModel(
                    "gemini-2.0-flash",
                    system_instruction=SYSTEM_PROMPT,
                )
                logger.info("Gemini LLM initialized")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}. Using fallback mode.")
        else:
            logger.info("No GEMINI_API_KEY. Using fallback rule-based mode.")

    def _get_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    # ── Main chat endpoint ───────────────────────────────────────────────

    async def chat(self, session_id: str, user_message: str) -> tuple[str, list[dict]]:
        session = self._get_session(session_id)
        session.add_turn("user", user_message)

        # Step 1: Extract intent
        intent = self._detect_intent(user_message, session)

        # Step 2: Update dialogue slots
        self._update_slots(user_message, session)

        # Step 3: Policy — decide action
        action = self._decide_action(intent, session)

        # Step 4: Execute action
        response_text, results = await self._execute_action(action, user_message, session)

        session.add_turn("assistant", response_text)
        session.last_recommendations = results

        cards = self._build_cards(results[:10])
        return response_text, cards

    # ── Intent Detection ─────────────────────────────────────────────────

    def _detect_intent(self, msg: str, session: Session) -> Intent:
        m = msg.lower().strip()

        # Greetings
        if m in ("hi", "hello", "hey", "hi!", "hello!", "hey!", "good morning",
                 "good afternoon", "good evening"):
            return Intent.GREETING

        # Help
        if m in ("help", "what can you do", "what can you do?", "?", "capabilities"):
            return Intent.HELP

        # Compare
        if any(w in m for w in ("compare", "versus", "vs", "difference between", "vs.")):
            return Intent.COMPARE

        # Detail about specific assessment
        if any(w in m for w in ("tell me about", "details about", "describe", "what is")):
            return Intent.DETAIL

        # Refinement (user has previous results and is narrowing)
        refinement_words = ["also", "instead", "but", "only", "filter", "narrow",
                            "shorter", "longer", "cheaper", "different", "more",
                            "less than", "under", "within", "exclude", "remove"]
        if session.last_recommendations and any(w in m for w in refinement_words):
            return Intent.REFINE

        # Off-topic
        off_topic = ["weather", "recipe", "joke", "news", "sports", "movie", "music",
                      "game", "politics", "stock", "crypto"]
        assessment_words = ["assess", "test", "shl", "hire", "skill", "role", "job",
                            "candidate", "eval", "measure", "screen"]
        if any(w in m for w in off_topic) and not any(w in m for w in assessment_words):
            return Intent.OFF_TOPIC

        # If we're in a clarification flow, this is a response
        if session.turn_count > 1 and not session.slots.has_enough_info():
            return Intent.CLARIFY_RESPONSE

        return Intent.RECOMMEND

    # ── Slot Extraction ──────────────────────────────────────────────────

    def _update_slots(self, msg: str, session: Session):
        m = msg.lower()
        slots = session.slots

        # Always update raw query
        slots.raw_query = msg

        # Job level
        jl_map = {
            "entry level": "Entry-Level", "entry-level": "Entry-Level",
            "junior": "Entry-Level", "fresher": "Entry-Level",
            "graduate": "Graduate", "grad": "Graduate",
            "mid level": "Mid-Professional", "mid-level": "Mid-Professional",
            "mid professional": "Mid-Professional",
            "senior": "Professional Individual Contributor",
            "professional": "Professional Individual Contributor",
            "individual contributor": "Professional Individual Contributor",
            "front line manager": "Front Line Manager",
            "supervisor": "Supervisor",
            "manager": "Manager", "managerial": "Manager",
            "director": "Director",
            "executive": "Executive", "c-level": "Executive",
            "general": "General Population",
        }
        for kw, level in jl_map.items():
            if kw in m:
                slots.job_level = level
                break

        # Duration
        dur_match = re.search(r"(?:under|less than|max|within|shorter than)\s*(\d+)\s*(?:min)?", m)
        if dur_match:
            slots.max_duration = int(dur_match.group(1))
        elif any(w in m for w in ("quick", "short", "brief", "fast")) and not slots.max_duration:
            slots.max_duration = 15

        # Remote
        if any(w in m for w in ("remote", "online", "virtual")):
            slots.remote_only = True

        # Adaptive
        if "adaptive" in m:
            slots.adaptive_only = True

        # Category
        cat_map = {
            "coding": "Simulations", "programming": "Simulations",
            "simulation": "Simulations",
            "personality": "Personality & Behavior", "behavioral": "Personality & Behavior",
            "cognitive": "Ability & Aptitude", "aptitude": "Ability & Aptitude",
            "reasoning": "Ability & Aptitude",
            "knowledge": "Knowledge & Skills", "technical": "Knowledge & Skills",
            "competency": "Competencies",
            "judgment": "Biodata & Situational Judgment", "sjt": "Biodata & Situational Judgment",
            "development": "Development & 360", "360": "Development & 360",
            "exercise": "Assessment Exercises",
        }
        for kw, cat in cat_map.items():
            if kw in m:
                slots.category = cat
                break

        # Language
        lang_map = {"spanish": "Spanish", "french": "French", "german": "German",
                     "portuguese": "Portuguese", "chinese": "Chinese", "arabic": "Arabic"}
        for kw, lang in lang_map.items():
            if kw in m:
                slots.language = lang
                break

        # Skills extraction (look for common tech/domain terms)
        skill_patterns = [
            r'\bjava\b', r'\bpython\b', r'\bsql\b', r'\bc\+\+\b', r'\bc#\b',
            r'\bjavascript\b', r'\breact\b', r'\bangular\b', r'\bnode\b',
            r'\bdata science\b', r'\bmachine learning\b', r'\bdevops\b',
            r'\bleadership\b', r'\bcustomer service\b', r'\bsales\b',
            r'\baccounting\b', r'\bfinance\b', r'\bnursing\b', r'\bengineering\b',
        ]
        for pat in skill_patterns:
            if re.search(pat, m):
                skill = re.search(pat, m).group()
                if skill not in slots.skills:
                    slots.skills.append(skill)

    # ── Policy Layer ─────────────────────────────────────────────────────

    def _decide_action(self, intent: Intent, session: Session) -> Action:
        """Explicit policy: decide action based on intent and state."""
        if intent == Intent.GREETING:
            return Action.GREET
        if intent == Intent.HELP:
            return Action.SHOW_HELP
        if intent == Intent.OFF_TOPIC:
            return Action.REFUSE
        if intent == Intent.COMPARE:
            return Action.COMPARE_ASSESSMENTS
        if intent == Intent.DETAIL:
            return Action.SHOW_DETAIL

        # For RECOMMEND / CLARIFY_RESPONSE / REFINE:
        if not session.slots.has_enough_info() and session.turn_count <= 2:
            return Action.ASK_CLARIFY

        if intent == Intent.REFINE:
            return Action.REFINE_RESULTS

        return Action.RETRIEVE_AND_RESPOND

    # ── Action Execution ─────────────────────────────────────────────────

    async def _execute_action(
        self, action: Action, msg: str, session: Session
    ) -> tuple[str, list[dict]]:

        if action == Action.GREET:
            return self._greet(), []

        if action == Action.SHOW_HELP:
            return self._help_text(), []

        if action == Action.REFUSE:
            return self._refuse_text(), []

        if action == Action.ASK_CLARIFY:
            return self._clarify_text(session), []

        if action == Action.COMPARE_ASSESSMENTS:
            return await self._do_compare(msg, session)

        if action == Action.SHOW_DETAIL:
            return await self._do_detail(msg, session)

        # RETRIEVE_AND_RESPOND or REFINE_RESULTS
        return await self._do_recommend(msg, session)

    async def _do_recommend(self, msg: str, session: Session) -> tuple[str, list[dict]]:
        slots = session.slots
        query = slots.to_search_query()

        results = self.retriever.search(
            query=query, top_k=10,
            job_level=slots.job_level, language=slots.language,
            max_duration=slots.max_duration,
            remote_only=slots.remote_only, adaptive_only=slots.adaptive_only,
            category=slots.category,
        )

        if self._llm:
            text = await self._llm_response(session, msg, results)
        else:
            text = self._fallback_recommend(results, session)

        return text, results

    async def _do_compare(self, msg: str, session: Session) -> tuple[str, list[dict]]:
        # Try to find assessment names in the message or use last recommendations
        results = session.last_recommendations[:2] if session.last_recommendations else []

        if not results:
            # Search for what user wants to compare
            search_results = self.retriever.search(msg, top_k=2)
            results = search_results

        if len(results) < 2:
            return ("I need at least two assessments to compare. Could you name the "
                    "assessments you'd like to compare, or let me recommend some first?"), []

        if self._llm:
            text = await self._llm_response(session, msg, results)
        else:
            text = self._fallback_compare(results[:2])

        return text, results[:2]

    async def _do_detail(self, msg: str, session: Session) -> tuple[str, list[dict]]:
        results = self.retriever.search(msg, top_k=1)
        if not results:
            return "I couldn't find that assessment. Could you provide the exact name?", []

        item = results[0]
        if self._llm:
            text = await self._llm_response(session, msg, results[:1])
        else:
            text = self._fallback_detail(item)

        return text, results[:1]

    # ── LLM Response ─────────────────────────────────────────────────────

    async def _llm_response(self, session: Session, msg: str, results: list[dict]) -> str:
        try:
            ctx = self._format_catalog_context(results[:10])
            prompt = f"""## Conversation History
{session.get_history_text()}

## Retrieved Assessments from Catalog (ONLY use these)
{ctx}

## Current User Message
{msg}

## Active Filters
{json.dumps({k: v for k, v in session.slots.__dict__.items() if v and k != 'raw_query'}, default=str)}

Respond helpfully. ONLY recommend assessments listed above. Use exact names and URLs."""

            response = self._llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._fallback_recommend(results, session)

    def _format_catalog_context(self, results: list[dict]) -> str:
        if not results:
            return "No matching assessments found."
        lines = []
        for i, item in enumerate(results, 1):
            lines.append(
                f"{i}. {item['name']}\n"
                f"   URL: {item['link']}\n"
                f"   Description: {item.get('description', 'N/A')}\n"
                f"   Duration: {item.get('duration', 'N/A')} | Remote: {item.get('remote', 'N/A')} | "
                f"Adaptive: {item.get('adaptive', 'N/A')}\n"
                f"   Job Levels: {', '.join(item.get('job_levels', []))}\n"
                f"   Categories: {', '.join(item.get('keys', []))}"
            )
        return "\n\n".join(lines)

    # ── Fallback responses ───────────────────────────────────────────────

    def _greet(self) -> str:
        return ("Hello! 👋 I'm the **SHL Assessment Recommender**. I help you find the right "
                "SHL assessments for hiring and talent development.\n\n"
                "Tell me about:\n• The **role** you're hiring for\n"
                "• The **skills** you want to assess\n"
                "• Requirements like duration, remote support, or job level")

    def _help_text(self) -> str:
        return ("I can help with:\n\n"
                "🔍 **Find assessments** — by role, skills, or job level\n"
                "📋 **Filter** — by duration, remote, adaptive, language, category\n"
                "⚖️ **Compare** — two or more assessments side by side\n"
                "📝 **Details** — about a specific assessment\n\n"
                "**Try:** \"Java tests for mid-level developers\" or "
                "\"Compare personality assessments under 20 minutes\"")

    def _refuse_text(self) -> str:
        return ("I'm specifically designed to help with SHL assessment recommendations. 😊\n\n"
                "Ask me about finding, comparing, or filtering SHL assessments!")

    def _clarify_text(self, session: Session) -> str:
        missing = []
        if not session.slots.role and not session.slots.skills:
            missing.append("What **role or skills** are you looking to assess?")
        if not session.slots.job_level:
            missing.append("What **job level** (e.g., entry-level, mid-professional, manager)?")
        return ("I'd like to give you the best recommendations. Could you help me with:\n\n"
                + "\n".join(f"• {q}" for q in missing[:2]))

    def _fallback_recommend(self, results: list[dict], session: Session) -> str:
        if not results:
            filters = {k: v for k, v in session.slots.__dict__.items() if v and k != 'raw_query'}
            if filters:
                return (f"No assessments match your criteria ({', '.join(f'{k}={v}' for k, v in filters.items())}). "
                        "Try broadening your search.")
            return "I couldn't find matching assessments. Please provide more details about the role or skills."

        parts = [f"I found **{len(results)} assessment{'s' if len(results) > 1 else ''}** matching your needs:\n"]
        for i, item in enumerate(results[:5], 1):
            desc = item.get("description", "")
            if len(desc) > 140:
                desc = desc[:137] + "..."
            dur = item.get("duration", "N/A")
            rem = "✅ Remote" if item.get("remote", "").lower() == "yes" else ""
            adp = "🔄 Adaptive" if item.get("adaptive", "").lower() == "yes" else ""
            cats = ", ".join(item.get("keys", []))
            parts.append(f"**{i}. [{item['name']}]({item['link']})**\n"
                         f"   {desc}\n"
                         f"   ⏱️ {dur} | {rem}{' | ' + adp if adp else ''} | 📂 {cats}\n")

        if len(results) > 5:
            parts.append(f"\n*...and {len(results) - 5} more results available.*")
        parts.append("\nWould you like to refine, compare, or get details on any of these?")
        return "\n".join(parts)

    def _fallback_compare(self, items: list[dict]) -> str:
        a, b = items[0], items[1]
        return (f"## Comparison: {a['name']} vs {b['name']}\n\n"
                f"| Feature | {a['name']} | {b['name']} |\n"
                f"|---------|------------|------------|\n"
                f"| Duration | {a.get('duration','N/A')} | {b.get('duration','N/A')} |\n"
                f"| Remote | {a.get('remote','N/A')} | {b.get('remote','N/A')} |\n"
                f"| Adaptive | {a.get('adaptive','N/A')} | {b.get('adaptive','N/A')} |\n"
                f"| Categories | {', '.join(a.get('keys',[]))} | {', '.join(b.get('keys',[]))} |\n"
                f"| Job Levels | {', '.join(a.get('job_levels',[])) or 'N/A'} | {', '.join(b.get('job_levels',[])) or 'N/A'} |\n\n"
                f"**{a['name']}**: {a.get('description','')[:200]}\n\n"
                f"**{b['name']}**: {b.get('description','')[:200]}\n\n"
                f"[View {a['name']}]({a['link']}) | [View {b['name']}]({b['link']})")

    def _fallback_detail(self, item: dict) -> str:
        return (f"## {item['name']}\n\n"
                f"**Description:** {item.get('description', 'N/A')}\n\n"
                f"**Duration:** {item.get('duration', 'N/A')}\n"
                f"**Remote Testing:** {item.get('remote', 'N/A')}\n"
                f"**Adaptive:** {item.get('adaptive', 'N/A')}\n"
                f"**Job Levels:** {', '.join(item.get('job_levels', [])) or 'N/A'}\n"
                f"**Languages:** {', '.join(item.get('languages', [])) or 'N/A'}\n"
                f"**Categories:** {', '.join(item.get('keys', []))}\n\n"
                f"🔗 [View in SHL Catalog]({item['link']})")

    def _build_cards(self, results: list[dict]) -> list[dict]:
        return [AssessmentCard(
            name=item["name"], url=item["link"],
            description=item.get("description", ""),
            duration=item.get("duration", "N/A"),
            remote=item.get("remote", "N/A"),
            adaptive=item.get("adaptive", "N/A"),
            job_levels=item.get("job_levels", []),
            categories=item.get("keys", []),
        ).model_dump() for item in results]
