import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import LLM_CONFIG, SEARCH_CONFIG
from llm_client import LLMClient, LLMConfig, LLMError


@dataclass
class RouteResult:
    search_str: str
    filter: Dict[str, bool]


class ChatbotAgent:
    def __init__(
        self,
        db,
        llm_config: LLMConfig | None = None,
        llm_client: LLMClient | None = None,
        search_top_k: int | None = None,
        context_chars: int | None = None,
    ):
        self.db = db
        self.llm_config = llm_config or LLMConfig.from_settings(LLM_CONFIG)
        self.llm = llm_client or LLMClient(self.llm_config)
        self.search_top_k = search_top_k or int(os.getenv("SEARCH_TOP_K", str(SEARCH_CONFIG["top_k"])))
        self.context_chars = context_chars or int(os.getenv("LLM_CONTEXT_CHARS", str(SEARCH_CONFIG["context_chars"])))

    def _route_query(self, user_query: str) -> RouteResult:
        query = user_query.strip()
        lower = query.lower()
        meta_filter: Dict[str, bool] = {}

        if re.search(r"\b(intern|internship|interns)\b", lower):
            meta_filter["is_intern"] = True

        return RouteResult(search_str=query, filter=meta_filter)

    def _build_context(self, context_chunks: list) -> str:
        context_lines = []
        for c in context_chunks:
            section = c.get("section", "unknown")
            text = c.get("text", "")
            if len(text) > self.context_chars:
                text = text[: self.context_chars].rstrip() + "..."
            context_lines.append(
                f"[Candidate ID: {c['candidate_id']} | Section: {section}]\nText: {text}"
            )
        return "\n\n".join(context_lines)

    def query_synthesizer(self, user_query: str, context_chunks: list) -> str:
        if not context_chunks:
            return "Not found in candidate pool."

        context_str = self._build_context(context_chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an HR Assistant. Answer the recruiter's query using only the provided context. "
                    "Cite the candidate_id for every claim. Quote direct lines from the resume text exactly. "
                    'If the answer cannot be found, say "Not found in candidate pool".'
                ),
            },
            {
                "role": "user",
                "content": f"Query: {user_query}\n\nContext:\n{context_str}\n\nAnswer:",
            },
        ]

        return self.llm.chat(messages)

    def chat(self, user_query: str) -> str:
        routing = self._route_query(user_query)

        search_res = self.db.semantic_search(
            routing.search_str,
            where_filter=routing.filter or None,
            top_k=self.search_top_k,
        )

        chunks = []
        if search_res and search_res.get("ids") and search_res["ids"][0]:
            for i, text in enumerate(search_res["documents"][0]):
                meta = search_res["metadatas"][0][i]
                chunks.append(
                    {
                        "candidate_id": meta["candidate_id"],
                        "section": meta.get("section", "unknown"),
                        "text": text,
                    }
                )

        return self.query_synthesizer(user_query, chunks)
