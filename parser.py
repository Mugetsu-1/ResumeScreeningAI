import concurrent.futures
import io
import re
import warnings
from typing import Any, Dict, List

import docx
import pandas as pd
import pdfplumber
import spacy

try:
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["tagger", "parser", "lemmatizer", "attribute_ruler"],
    )
except Exception:
    warnings.warn(
        "spaCy model 'en_core_web_sm' is not available. Falling back to a blank English model.",
        RuntimeWarning,
    )
    nlp = spacy.blank("en")

class ResumeParser:
    def __init__(self):
        self.skill_map = {
            "scikit-learn": ["sklearn", "scikit learn", "scikit-learn"],
            "machine learning": ["ml", "machine learning"],
            "pandas": ["pandas"],
            "numpy": ["numpy"],
            "python": ["python", "python3"],
            "sql": ["sql", "mysql", "postgresql"],
            "tensorflow": ["tensorflow"],
            "pytorch": ["pytorch", "torch"],
            "react": ["react", "reactjs", "react.js"],
            "aws": ["aws", "amazon web services"]
        }

        self.syn_to_canon: Dict[str, str] = {}
        for canonical, syns in self.skill_map.items():
            for syn in syns:
                self.syn_to_canon[syn.lower()] = canonical

        # Pre-compile a single regex so resume scans stay fast and deterministic.
        skill_terms = sorted(self.syn_to_canon.keys(), key=len, reverse=True)
        master_pattern = r"\b(" + "|".join(map(re.escape, skill_terms)) + r")\b"
        self.skills_regex = re.compile(master_pattern, re.IGNORECASE)

        self.section_headers = re.compile(
            r"^(EXPERIENCE|EDUCATION|SKILLS|PROJECTS|SUMMARY|OBJECTIVE|CERTIFICATIONS)[\s:]*$",
            re.IGNORECASE,
        )
        self.phone_regex = re.compile(
            r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        )
        self.email_regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        text = ""
        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    chunk = page.extract_text()
                    if chunk:
                        text += chunk + "\n"
        elif filename.lower().endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs])
        return text

    def extract_entities(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {"name": "Unknown", "email": None, "phone": None, "skills": []}

        # Contacts
        emails = self.email_regex.findall(text)
        phones = self.phone_regex.findall(text)
        email = emails[0] if emails else None
        phone = phones[0] if phones else None

        # Name estimation from spaCy first, then a light heuristic fallback.
        name = "Unknown"
        top_doc = nlp(text[:1000])
        for ent in getattr(top_doc, "ents", []):
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                break
        if name == "Unknown":
            name = self._infer_name(text)

        found_synonyms = {match.group(0).lower() for match in self.skills_regex.finditer(text)}
        extracted_skills = sorted({self.syn_to_canon[syn] for syn in found_synonyms})

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "skills": extracted_skills
        }

    def _infer_name(self, text: str) -> str:
        for line in text.splitlines()[:10]:
            candidate = line.strip(" \t-•")
            if not candidate:
                continue
            if self.email_regex.search(candidate) or self.phone_regex.search(candidate):
                continue
            words = candidate.split()
            if 2 <= len(words) <= 4 and all(word[:1].isupper() for word in words):
                return candidate
        return "Unknown"

    def segment_text(self, text: str) -> Dict[str, str]:
        sections = {
            "contact": [],
            "skills": [],
            "experience": [],
            "education": [],
            "projects": [],
            "other": [],
        }
        current_section = "contact"

        for line in text.split("\n"):
            line_str = line.strip()
            if not line_str:
                continue

            if self.section_headers.fullmatch(line_str) or (
                len(line_str.split()) <= 2 and line_str.isupper()
            ):
                sec_name = line_str.lower()
                if "exp" in sec_name:
                    current_section = "experience"
                elif "edu" in sec_name:
                    current_section = "education"
                elif "skill" in sec_name:
                    current_section = "skills"
                elif "proj" in sec_name:
                    current_section = "projects"
                else:
                    current_section = "other"
            else:
                sections[current_section].append(line_str)

        return {k: "\n".join(v) for k, v in sections.items()}

    def parse_resumes(self, files_info: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        files_info: [{"candidate_id": "1", "filename": "x.pdf", "bytes": b"..."}]
        """
        parsed_data = []

        def process_file(f):
            try:
                raw_text = self.extract_text(f["bytes"], f["filename"])
                if not raw_text.strip():
                    return None

                entities = self.extract_entities(raw_text)
                segments = self.segment_text(raw_text)

                exp_match = re.search(
                    r"(\d+)(?:\+| years?)(?: of)? (?:professional )?experience",
                    raw_text.lower(),
                )
                years_exp = float(exp_match.group(1)) if exp_match else 0.0

                is_intern = bool(re.search(r"\bintern\b|\binternship\b", raw_text.lower()))

                return {
                    "candidate_id": f["candidate_id"],
                    "name": entities["name"],
                    "email": entities["email"] or "",
                    "phone": entities["phone"] or "",
                    "skills": entities["skills"],
                    "years_of_exp": years_exp,
                    "is_intern": is_intern,
                    "segments": segments,
                    "raw_text": raw_text
                }
            except Exception as exc:
                warnings.warn(f"Skipping {f.get('filename', 'unknown file')}: {exc}", RuntimeWarning)
                return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_file, files_info)
            for row in results:
                if row is not None:
                    parsed_data.append(row)

        return pd.DataFrame(parsed_data)
