import json
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class ChatbotAgent:
    def __init__(self, db, llm_api_key: str = None, base_url: str = None):
        self.db = db
        self.base_url = base_url or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        self.api_key = llm_api_key or os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0.0
        )

    def _parse_json_response(self, raw_text: str):
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found")

        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj

    def query_router(self, user_query: str):
        prompt = PromptTemplate(
            input_variables=["query"],
            template='''
You are a Query Router for an HR database.
Return valid JSON only, with no markdown or explanation.
Allowed filter keys: "is_intern" (bool).
Return exactly this shape:
{"search_str": "string", "filter": {"is_intern": true}}
If there is no filter, use an empty object.
Query: {query}
            '''
        )
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"query": user_query})
            return self._parse_json_response(resp.content)
        except Exception:
            return {"search_str": user_query, "filter": {}}

    def query_synthesizer(self, user_query: str, context_chunks: list):
        if not context_chunks:
            return "Not found in candidate pool."
            
        context_str = ""
        for c in context_chunks:
            section = c.get("section", "unknown")
            context_str += f"\n[Candidate ID: {c['candidate_id']} | Section: {section}]\nText: {c['text']}\n"

        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template='''
            You are an HR Assistant. Answer the recruiter's query using ONLY the provided context.
            STRICT RULES:
            1. Cite the candidate_id for every claim.
            2. Quote direct lines from the resume text exactly.
            3. If answer cannot be found, say "Not found in candidate pool".

            Query: {query}
            Context:
            {context}
            Answer:
            '''
        )
        chain = prompt | self.llm
        resp = chain.invoke({"query": user_query, "context": context_str})
        return resp.content

    def chat(self, user_query: str):
        routing = self.query_router(user_query)
        search_str = routing.get("search_str", user_query)
        meta_filter = routing.get("filter", None)

        # ChromaDB rejects empty dicts {} for the where clause.
        if not meta_filter:
            meta_filter = None

        search_res = self.db.semantic_search(search_str, where_filter=meta_filter, top_k=5)

        chunks = []
        if search_res and search_res["ids"] and search_res["ids"][0]:
            for i, text in enumerate(search_res["documents"][0]):
                meta = search_res["metadatas"][0][i]
                chunks.append({
                    "candidate_id": meta["candidate_id"],
                    "section": meta.get("section", "unknown"),
                    "text": text
                })

        return self.query_synthesizer(user_query, chunks)
