import chromadb
from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

class VectorDB:
    def __init__(self, collection_name: str = "resumes", persist_directory: str = "chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.encoder = None

    def _get_encoder(self):
        if self.encoder is None:
            try:
                self.encoder = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load embedding model '{self.model_name}'. "
                    "Install the model once with internet access or cache it locally before running the app."
                ) from exc
        return self.encoder

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
            encoder = self._get_encoder()
            embeddings = encoder.encode(
                texts, 
                batch_size=32, 
                device=self.device, 
                normalize_embeddings=True
            ).tolist()
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

    def semantic_search(self, query: str, where_filter: Dict[str, Any] = None, top_k: int = 10) -> Dict[str, Any]:
        encoder = self._get_encoder()
        query_emb = encoder.encode([query], normalize_embeddings=True).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        return results
