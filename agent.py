import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class ChatbotAgent:
    def __init__(self, db, llm_api_key: str):
        self.db = db
        self.llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0.0
        )

    def query_router(self, user_query: str):
        prompt = PromptTemplate(
            input_variables=["query"],
            template='''
            You are a Query Router for an HR DB. Convert the query into JSON.
            Allowed keys for metadata filter: "is_intern" (bool).
            Return a dictionary with two keys: "search_str": string for semantic search, "filter": dictionary for metadata constraints.
            Query: {query}
            JSON:
            '''
        )
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"query": user_query})
            cleaned = resp.content.strip().strip('```json').strip('```')
            return json.loads(cleaned)
        except:
            return {"search_str": user_query, "filter": {}}

    def query_synthesizer(self, user_query: str, context_chunks: list):
        if not context_chunks:
            return "Not found in candidate pool."
            
        context_str = ""
        for c in context_chunks:
            context_str += f"\n[Candidate ID: {c['candidate_id']}]\nText: {c['text']}\n"

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
                chunks.append({"candidate_id": meta["candidate_id"], "text": text})

        return self.query_synthesizer(user_query, chunks)