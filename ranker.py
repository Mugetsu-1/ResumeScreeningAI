import pandas as pd
from typing import List, Dict, Any

class CandidateRanker:
    """Implement exact ranking formula"""
    def __init__(self, db, parser):
        self.db = db
        self.parser = parser

    def rank_candidates(self, job_desc: str, req_skills: List[str] = None, req_years_exp: float = 0.0) -> pd.DataFrame:
        if not job_desc.strip():
            return pd.DataFrame()
            
        # Use simple extraction for JD if req_skills not provided
        if req_skills is None:
            extracted = self.parser.extract_entities(job_desc)
            req_skills = extracted["skills"]

        # Get candidates sorted by semantic similarity. Top 20 chunks.
        # Note: distances in Chroma are often L2. We map to similarity: 1 / (1 + distance)
        search_res = self.db.semantic_search(job_desc, top_k=20)
        
        candidates_scores = {}
        
        if search_res and search_res["ids"] and search_res["ids"][0]:
            ids = search_res["ids"][0]
            dists = search_res["distances"][0]
            metas = search_res["metadatas"][0]
            
            for i, chunk_id in enumerate(ids):
                meta = metas[i]
                cid = meta["candidate_id"]
                dist = dists[i]
                # convert L2 distance roughly to similarity 0.0 - 1.0 interval
                sim_score = 1.0 / (1.0 + dist)
                
                if cid not in candidates_scores:
                    candidates_scores[cid] = {
                        "candidate_id": cid,
                        "email": str(meta.get("email", "")),
                        "skills": [s for s in meta["skills"].split(",") if s and s != "none"],
                        "years_of_exp": float(meta.get("years_of_exp", 0.0) or 0.0),
                        "is_intern": bool(meta.get("is_intern", False)),
                        "max_semantic_score": sim_score
                    }
                else:
                    candidates_scores[cid]["max_semantic_score"] = max(candidates_scores[cid]["max_semantic_score"], sim_score)

        if not candidates_scores:
            return pd.DataFrame()

        results = []
        req_skills_set = set(req_skills) if req_skills else set()
        
        for cid, data in candidates_scores.items():
            cand_skills_set = set(data["skills"])
            
            # Skill Overlap calculation safely
            if req_skills_set:
                overlap_ratio = len(req_skills_set.intersection(cand_skills_set)) / len(req_skills_set)
            else:
                overlap_ratio = 1.0 if not cand_skills_set else 0.0
                
            # Experience Fit
            exp_fit = min(data["years_of_exp"] / req_years_exp, 1.0) if req_years_exp > 0 else 1.0
            
            sem_score = data["max_semantic_score"]
            
            # Final Score = (0.5 * Semantic_Score) + (0.3 * Skill_Overlap_Ratio) + (0.2 * Experience_Fit_Score)
            final_score = (0.5 * sem_score) + (0.3 * overlap_ratio) + (0.2 * exp_fit)
            
            results.append({
                "candidate_id": cid,
                "email": data["email"],
                "final_score": round(final_score, 4),
                "semantic_score": round(sem_score, 4),
                "skill_overlap": round(overlap_ratio, 4),
                "exp_fit": round(exp_fit, 4),
                "matched_skills": sorted(req_skills_set.intersection(cand_skills_set)),
                "missing_skills": sorted(req_skills_set.difference(cand_skills_set))
            })
            
        df = pd.DataFrame(results).sort_values("final_score", ascending=False)
        return df
