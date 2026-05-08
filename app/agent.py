"""
Conversational Agent with Dialogue State Tracking and Policy Layer.

Designed for the SHL automated evaluator:
- Schema: recommendations EMPTY when clarifying, 1-10 when recommending, end_of_conversation flag
- Behavior probes: refuse off-topic, don't recommend on turn 1 for vague queries, honor edits
- Turn cap: max 8 turns total (user + assistant)
- Grounding: ALL recommendations from catalog only, never hallucinated

Architecture:
  1. Dialogue State Extractor — slots from conversation history
  2. Policy Layer — clarify / recommend / refine / compare / refuse
  3. HybridRetriever — BM25 + TF-IDF ranked results
  4. Response Generator — Gemini LLM or rule-based fallback
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

MAX_TURNS = 8  # Evaluator caps at 8 total turns (user + assistant)


class Intent(Enum):
    GREETING = "greeting"
    RECOMMEND = "recommend"
    CLARIFY_RESPONSE = "clarify_response"
    REFINE = "refine"
    COMPARE = "compare"
    DETAIL = "detail"
    HELP = "help"
    OFF_TOPIC = "off_topic"
    THANKS = "thanks"


class Action(Enum):
    ASK_CLARIFY = "ask_clarify"
    RETRIEVE_AND_RESPOND = "retrieve_and_respond"
    REFINE_RESULTS = "refine_results"
    COMPARE_ASSESSMENTS = "compare_assessments"
    SHOW_DETAIL = "show_detail"
    GREET = "greet"
    SHOW_HELP = "show_help"
    REFUSE = "refuse"
    END = "end"


@dataclass
class DialogueSlots:
    """Structured user requirements extracted from conversation."""
    role: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    job_level: Optional[str] = None
    max_duration: Optional[int] = None
    remote_only: bool = False
    adaptive_only: bool = False
    category: Optional[str] = None
    language: Optional[str] = None
    raw_query: str = ""

    def has_enough_info(self) -> bool:
        return bool(self.role or len(self.skills) >= 1 or self.category or self.job_level)

    def has_strong_info(self) -> bool:
        """Has enough info for a confident recommendation."""
        filled = sum([
            bool(self.role), len(self.skills) >= 1, bool(self.category),
            bool(self.job_level), bool(self.max_duration), self.remote_only,
        ])
        # 1 specific skill/role is enough for a recommendation
        return filled >= 1 or bool(self.role)

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
    has_recommended: bool = False

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

    def total_turns(self) -> int:
        return len(self.history)


SYSTEM_PROMPT = """You are the SHL Assessment Recommender, an expert AI that helps HR professionals find SHL assessments.

ABSOLUTE RULES:
1. ONLY recommend assessments from the RETRIEVED CATALOG RESULTS provided below. Never invent names or URLs.
2. Every recommendation MUST include the exact assessment name and URL from the catalog.
3. If the query is vague, ask 1-2 SHORT clarifying questions. Do NOT recommend yet.
4. If off-topic (not about SHL assessments, hiring, testing), politely refuse and redirect.
5. When comparing, use ONLY catalog data fields.
6. Keep responses concise — no more than 3-4 sentences plus the recommendation list.
7. When you provide recommendations, list them clearly with name, duration, and what they measure.

RESPONSE FORMAT when recommending:
- Brief 1-sentence summary
- Numbered list: assessment name, what it measures, duration
- Ask if they want to refine

When you have enough context, provide your recommended shortlist and consider the conversation complete."""


class SHLAgent:
    """Agent with dialogue state tracking, policy layer, and evaluator-compliant schema."""

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
                logger.info("Gemini LLM initialized successfully")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")
        else:
            logger.info("No GEMINI_API_KEY. Using fallback mode.")

    def _get_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    # ── Main chat ────────────────────────────────────────────────────────

    async def chat(self, session_id: str, user_message: str) -> tuple[str, list[dict], bool]:
        """
        Returns (response_text, recommendations, end_of_conversation).
        recommendations is EMPTY when clarifying, 1-10 when recommending.
        """
        session = self._get_session(session_id)
        session.add_turn("user", user_message)

        # Detect intent
        intent = self._detect_intent(user_message, session)

        # Update slots from message
        self._update_slots(user_message, session)

        # Policy decision
        action = self._decide_action(intent, session)

        # Execute
        response_text, results, end_conv = await self._execute_action(action, user_message, session)

        session.add_turn("assistant", response_text)
        if results:
            session.last_recommendations = results
            session.has_recommended = True

        return response_text, self._build_cards(results), end_conv

    # ── Intent Detection ─────────────────────────────────────────────────

    def _detect_intent(self, msg: str, session: Session) -> Intent:
        m = msg.lower().strip()

        if m in ("hi", "hello", "hey", "hi!", "hello!", "hey!",
                 "good morning", "good afternoon", "good evening"):
            return Intent.GREETING

        if m in ("help", "what can you do", "what can you do?", "?"):
            return Intent.HELP

        if any(w in m for w in ("thank", "thanks", "that's all", "no more", "bye",
                                 "goodbye", "that is all", "done", "perfect")):
            return Intent.THANKS

        if any(w in m for w in ("compare", "versus", "vs ", "vs.", "difference between")):
            return Intent.COMPARE

        # Off-topic — check BEFORE detail to avoid "what is the weather" matching DETAIL
        off_topic = ["weather", "recipe", "joke", "news", "sports", "movie", "music",
                      "game", "politics", "stock", "crypto", "restaurant", "travel",
                      "song", "food", "play", "watch"]
        assess_words = ["assess", "test", "shl", "hire", "skill", "role", "job",
                        "candidate", "eval", "measure", "screen", "interview",
                        "recruit", "talent", "aptitude", "personality", "competenc",
                        "simulation", "cognitive", "ability"]
        if any(w in m for w in off_topic) and not any(w in m for w in assess_words):
            return Intent.OFF_TOPIC

        if any(w in m for w in ("tell me about", "details about", "describe",
                                 "more info", "explain")):
            return Intent.DETAIL
        # "what is X" only for detail if it mentions assessment-related terms
        if "what is" in m and any(w in m for w in assess_words):
            return Intent.DETAIL

        # Refinement (has previous results + narrowing language)
        refine_words = ["also", "instead", "but", "only", "filter", "narrow",
                        "shorter", "longer", "different", "remove", "exclude",
                        "less than", "under", "within", "change", "actually",
                        "no ", "not ", "without", "prefer"]
        if session.has_recommended and any(w in m for w in refine_words):
            return Intent.REFINE

        # If answering a clarifying question
        if not session.has_recommended and session.turn_count > 2:
            return Intent.CLARIFY_RESPONSE

        return Intent.RECOMMEND

    # ── Slot Extraction ──────────────────────────────────────────────────

    def _update_slots(self, msg: str, session: Session):
        m = msg.lower()
        slots = session.slots
        slots.raw_query = msg

        # Job level
        jl_map = {
            "entry level": "Entry-Level", "entry-level": "Entry-Level",
            "junior": "Entry-Level", "fresher": "Entry-Level",
            "graduate": "Graduate", "grad ": "Graduate",
            "mid level": "Mid-Professional", "mid-level": "Mid-Professional",
            "mid professional": "Mid-Professional", "mid-career": "Mid-Professional",
            "senior": "Professional Individual Contributor",
            "professional": "Professional Individual Contributor",
            "individual contributor": "Professional Individual Contributor",
            "front line manager": "Front Line Manager",
            "first line manager": "Front Line Manager",
            "supervisor": "Supervisor",
            "manager": "Manager", "managerial": "Manager",
            "director": "Director",
            "executive": "Executive", "c-level": "Executive", "c-suite": "Executive",
            "general population": "General Population",
        }
        for kw, level in jl_map.items():
            if kw in m:
                slots.job_level = level
                break

        # Duration
        dur_match = re.search(r"(?:under|less than|max|within|shorter than|no more than)\s*(\d+)\s*(?:min)?", m)
        if dur_match:
            slots.max_duration = int(dur_match.group(1))
        elif any(w in m for w in ("quick", "short", "brief", "fast")) and not slots.max_duration:
            slots.max_duration = 20

        # Remote
        if any(w in m for w in ("remote", "online", "virtual", "proctored remotely")):
            slots.remote_only = True

        # Adaptive / IRT
        if any(w in m for w in ("adaptive", "irt", "computer adaptive")):
            slots.adaptive_only = True

        # Category
        cat_map = {
            "coding": "Simulations", "programming": "Simulations",
            "simulation": "Simulations", "hands-on": "Simulations",
            "personality": "Personality & Behavior", "behavioral": "Personality & Behavior",
            "behaviour": "Personality & Behavior", "opq": "Personality & Behavior",
            "cognitive": "Ability & Aptitude", "aptitude": "Ability & Aptitude",
            "ability": "Ability & Aptitude", "reasoning": "Ability & Aptitude",
            "numerical": "Ability & Aptitude", "verbal": "Ability & Aptitude",
            "knowledge": "Knowledge & Skills", "technical": "Knowledge & Skills",
            "competency": "Competencies", "competencies": "Competencies",
            "judgment": "Biodata & Situational Judgment", "sjt": "Biodata & Situational Judgment",
            "situational": "Biodata & Situational Judgment",
            "development": "Development & 360", "360": "Development & 360",
            "exercise": "Assessment Exercises", "role play": "Assessment Exercises",
        }
        for kw, cat in cat_map.items():
            if kw in m:
                slots.category = cat
                break

        # Language
        lang_map = {"spanish": "Spanish", "french": "French", "german": "German",
                     "portuguese": "Portuguese", "chinese": "Chinese", "arabic": "Arabic",
                     "japanese": "Japanese", "korean": "Korean", "dutch": "Dutch",
                     "italian": "Italian", "russian": "Russian"}
        for kw, lang in lang_map.items():
            if kw in m:
                slots.language = lang
                break

        # Skills extraction
        skill_patterns = [
            r'\bjava\b', r'\bpython\b', r'\bsql\b', r'\bc\+\+\b', r'\bc#\b',
            r'\bjavascript\b', r'\b\.net\b', r'\breact\b', r'\bangular\b',
            r'\bdata science\b', r'\bmachine learning\b', r'\bdevops\b',
            r'\bleadership\b', r'\bcustomer service\b', r'\bsales\b',
            r'\baccounting\b', r'\bfinance\b', r'\bnursing\b', r'\bengineering\b',
            r'\bcall center\b', r'\bcontact center\b', r'\bdata entry\b',
            r'\btyping\b', r'\bmechanical\b', r'\belectrical\b',
        ]
        for pat in skill_patterns:
            match = re.search(pat, m)
            if match:
                skill = match.group()
                if skill not in slots.skills:
                    slots.skills.append(skill)

        # Role extraction (longer phrases that indicate a role)
        role_patterns = [
            r'(software (?:developer|engineer))', r'(data (?:scientist|analyst|engineer))',
            r'(project manager)', r'(business analyst)', r'(sales (?:representative|manager))',
            r'(customer service (?:rep|representative|agent))',
            r'(call center (?:agent|operator))', r'(web developer)',
            r'(full stack developer)', r'(front.?end developer)',
            r'(back.?end developer)', r'(devops engineer)',
            r'(system administrator)', r'(network engineer)',
            r'(hr (?:manager|specialist))', r'(financial analyst)',
        ]
        for pat in role_patterns:
            match = re.search(pat, m)
            if match:
                slots.role = match.group()
                break

    # ── Policy Layer ─────────────────────────────────────────────────────

    def _decide_action(self, intent: Intent, session: Session) -> Action:
        """
        Policy enforces:
        - Clarify when vague (especially on turn 1)
        - Recommend only when enough constraints exist
        - Refine rather than restart
        - Compare from catalog fields
        - Refuse off-scope / prompt-injection
        """
        if intent == Intent.GREETING:
            return Action.GREET
        if intent == Intent.HELP:
            return Action.SHOW_HELP
        if intent == Intent.OFF_TOPIC:
            return Action.REFUSE
        if intent == Intent.THANKS:
            return Action.END
        if intent == Intent.COMPARE:
            return Action.COMPARE_ASSESSMENTS
        if intent == Intent.DETAIL:
            return Action.SHOW_DETAIL

        # BEHAVIOR PROBE: Don't recommend on turn 1 for vague queries
        if session.turn_count <= 2 and not session.slots.has_strong_info():
            return Action.ASK_CLARIFY

        # If we're near the turn cap, just recommend what we have
        if session.total_turns() >= MAX_TURNS - 2:
            return Action.RETRIEVE_AND_RESPOND

        # Need more info?
        if not session.slots.has_enough_info():
            return Action.ASK_CLARIFY

        if intent == Intent.REFINE:
            return Action.REFINE_RESULTS

        return Action.RETRIEVE_AND_RESPOND

    # ── Action Execution ─────────────────────────────────────────────────

    async def _execute_action(
        self, action: Action, msg: str, session: Session
    ) -> tuple[str, list[dict], bool]:
        """Returns (response_text, results, end_of_conversation)."""

        if action == Action.GREET:
            return self._greet(), [], False

        if action == Action.SHOW_HELP:
            return self._help_text(), [], False

        if action == Action.REFUSE:
            return self._refuse_text(), [], False

        if action == Action.END:
            return self._end_text(session), [], True

        if action == Action.ASK_CLARIFY:
            return self._clarify_text(session), [], False

        if action == Action.COMPARE_ASSESSMENTS:
            text, results = await self._do_compare(msg, session)
            return text, results, True

        if action == Action.SHOW_DETAIL:
            text, results = await self._do_detail(msg, session)
            return text, results, False

        # RETRIEVE_AND_RESPOND or REFINE_RESULTS
        text, results = await self._do_recommend(msg, session)
        # End conversation after providing recommendations
        end = len(results) > 0
        return text, results, end

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

        return text, results[:10]

    async def _do_compare(self, msg: str, session: Session) -> tuple[str, list[dict]]:
        results = session.last_recommendations[:2] if session.last_recommendations else []
        if not results:
            results = self.retriever.search(msg, top_k=2)

        if len(results) < 2:
            return "I need at least two assessments to compare. Could you tell me which ones?", []

        if self._llm:
            text = await self._llm_response(session, msg, results[:2])
        else:
            text = self._fallback_compare(results[:2])
        return text, results[:2]

    async def _do_detail(self, msg: str, session: Session) -> tuple[str, list[dict]]:
        results = self.retriever.search(msg, top_k=1)
        if not results:
            return "I couldn't find that assessment. Could you provide the exact name?", []

        if self._llm:
            text = await self._llm_response(session, msg, results[:1])
        else:
            text = self._fallback_detail(results[0])
        return text, results[:1]

    # ── LLM Response ─────────────────────────────────────────────────────

    async def _llm_response(self, session: Session, msg: str, results: list[dict]) -> str:
        try:
            ctx = self._format_catalog_context(results[:10])
            slots_info = {k: v for k, v in session.slots.__dict__.items()
                         if v and k != 'raw_query'}

            prompt = f"""## Conversation History
{session.get_history_text()}

## Retrieved Assessments from Catalog (ONLY use these — never invent)
{ctx}

## Extracted User Requirements
{json.dumps(slots_info, default=str)}

## Current User Message
{msg}

Provide your response. ONLY recommend from the assessments listed above. Use their exact names and URLs.
If making recommendations, list them concisely with name, what it measures, and duration.
Keep the response brief and professional."""

            response = self._llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._fallback_recommend(results, session)

    def _format_catalog_context(self, results: list[dict]) -> str:
        if not results:
            return "No matching assessments found in catalog."
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

    # ── Fallback (no LLM) ───────────────────────────────────────────────

    def _greet(self) -> str:
        return ("Hello! I'm the SHL Assessment Recommender. I help you find the right "
                "SHL assessments for your hiring and development needs.\n\n"
                "To get started, tell me:\n"
                "- What role are you hiring for?\n"
                "- What skills or competencies do you need to assess?\n"
                "- Any preferences (test duration, remote, job level)?")

    def _help_text(self) -> str:
        return ("I can help you find, compare, and filter SHL assessments.\n\n"
                "Try queries like:\n"
                "- 'I need a Java test for mid-level developers'\n"
                "- 'Short personality assessments under 20 minutes'\n"
                "- 'Compare the Python and Java tests'")

    def _refuse_text(self) -> str:
        return ("I'm specifically designed to help with SHL assessment recommendations. "
                "I can help you find assessments for specific roles, skills, or job levels. "
                "What role or skills would you like to assess?")

    def _end_text(self, session: Session) -> str:
        return ("You're welcome! I hope the recommendations are helpful. "
                "Feel free to come back if you need more assessment suggestions.")

    def _clarify_text(self, session: Session) -> str:
        slots = session.slots
        questions = []
        if not slots.role and not slots.skills:
            questions.append("What **role** are you hiring for, or what **skills** do you need to assess?")
        if not slots.job_level:
            questions.append("What **job level** is this for (e.g., entry-level, mid-professional, manager)?")
        if not slots.max_duration and not slots.remote_only:
            questions.append("Do you have any preferences for **test duration** or **remote testing**?")

        return ("I'd like to give you the best recommendations. Could you help me with:\n\n"
                + "\n".join(f"- {q}" for q in questions[:2]))

    def _fallback_recommend(self, results: list[dict], session: Session) -> str:
        if not results:
            return ("I couldn't find assessments matching those criteria. "
                    "Could you try broadening your requirements?")

        n = min(len(results), 10)
        parts = [f"Based on your requirements, here are my top {n} recommendations:\n"]
        for i, item in enumerate(results[:10], 1):
            desc = item.get("description", "")
            if len(desc) > 120:
                desc = desc[:117] + "..."
            dur = item.get("duration", "N/A")
            remote = "Remote" if item.get("remote", "").lower() == "yes" else "In-person"
            parts.append(f"{i}. **{item['name']}** — {desc}\n"
                         f"   Duration: {dur} | {remote} | {', '.join(item.get('keys', []))}\n"
                         f"   {item['link']}")

        parts.append("\nWould you like to refine these, compare any, or get more details?")
        return "\n".join(parts)

    def _fallback_compare(self, items: list[dict]) -> str:
        a, b = items[0], items[1]
        return (f"**Comparison: {a['name']} vs {b['name']}**\n\n"
                f"| Feature | {a['name']} | {b['name']} |\n"
                f"|---------|------------|------------|\n"
                f"| Duration | {a.get('duration','N/A')} | {b.get('duration','N/A')} |\n"
                f"| Remote | {a.get('remote','N/A')} | {b.get('remote','N/A')} |\n"
                f"| Adaptive | {a.get('adaptive','N/A')} | {b.get('adaptive','N/A')} |\n"
                f"| Categories | {', '.join(a.get('keys',[]))} | {', '.join(b.get('keys',[]))} |\n\n"
                f"**{a['name']}**: {a.get('description','')[:200]}\n\n"
                f"**{b['name']}**: {b.get('description','')[:200]}")

    def _fallback_detail(self, item: dict) -> str:
        return (f"**{item['name']}**\n\n"
                f"{item.get('description', 'N/A')}\n\n"
                f"Duration: {item.get('duration', 'N/A')} | "
                f"Remote: {item.get('remote', 'N/A')} | "
                f"Adaptive: {item.get('adaptive', 'N/A')}\n"
                f"Job Levels: {', '.join(item.get('job_levels', [])) or 'N/A'}\n"
                f"Categories: {', '.join(item.get('keys', []))}\n"
                f"URL: {item['link']}")

    def _build_cards(self, results: list[dict]) -> list[dict]:
        if not results:
            return []
        return [AssessmentCard(
            name=item["name"], url=item["link"],
            description=item.get("description", ""),
            duration=item.get("duration", "N/A"),
            remote=item.get("remote", "N/A"),
            adaptive=item.get("adaptive", "N/A"),
            job_levels=item.get("job_levels", []),
            categories=item.get("keys", []),
        ).model_dump() for item in results[:10]]
