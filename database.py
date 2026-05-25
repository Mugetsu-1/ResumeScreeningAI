import chromadb
from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

class VectorDB:
    def __init__(self, collection_name: str = "resumes"):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def ingest_parsed_df(self, df: pd.DataFrame):
        """
        Takes Phase 1 DataFrame, chunks by logical section, and inserts vectors + metadata.
        """
        if df.empty:
            return

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for _, row in df.iterrows():
            cid = row["candidate_id"]
            segments = row["segments"]
            
            base_meta = {
                "candidate_id": cid,
                "email": str(row["email"]),
                "skills": ",".join(row["skills"]) if row["skills"] else "none",
                "years_of_exp": float(row["years_of_exp"]),
                "is_intern": bool(row["is_intern"])
            }
            
            for sec_name, sec_text in segments.items():
                if not sec_text.strip():
                    continue
                    
                chunk_id = f"{cid}_{sec_name}"
                ids.append(chunk_id)
                texts.append(sec_text)
                
                meta = base_meta.copy()
                meta["section"] = sec_name
                metadatas.append(meta)

        if texts:
            # Batch encode
            embeddings = self.encoder.encode(
                texts, 
                batch_size=32, 
                device=self.device, 
                normalize_embeddings=True
            ).tolist()
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

    def semantic_search(self, query: str, where_filter: Dict[str, Any] = None, top_k: int = 10) -> Dict[str, Any]:
        query_emb = self.encoder.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        return results