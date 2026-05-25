import re
import pandas as pd
import pdfplumber
import docx
import io
import spacy
import concurrent.futures
from typing import Dict, List, Any

# Ensure Spacy is downloaded
try:
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])

class ResumeParser:
    def __init__(self):
        self.skill_map = {
            "scikit-learn": ["sklearn", "scikit learn", "scikit-learn"],
            "machine learning": ["ml", "machine learning"],
            "python": ["python", "python3"],
            "sql": ["sql", "mysql", "postgresql"],
            "react": ["react", "reactjs", "react.js"],
            "aws": ["aws", "amazon web services"]
        }
        
        # [SPEED BOOST 2] Pre-compile a single O(1) master regex for all skills instead of O(N*M) nested loops
        for canonical, syns in self.skill_map.items():
            for syn in syns:
                self.syn_to_canon[syn.lower()] = canonical
        master_pattern = r'\b(' + '|'.join(map(re.escape, self.syn_to_canon.keys())) + r')\b'
        self.skills_regex = re.compile(master_pattern, re.IGNORECASE)
        
        self.section_headers = re.compile(
            r'^(EXPERIENCE|EDUCATION|SKILLS|PROJECTS|SUMMARY|OBJECTIVE|CERTIFICATIONS)[\s:]*$', 
            re.IGNORECASE
        )
        self.phone_regex = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        self.email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        text = ""
        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    chunk = page.extract_text()
                    if chunk: text += chunk + "\n"
        elif filename.lower().endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs])
        return text

    def extract_entities(self, text: str) -> Dict[str, Any]:
        doc = nlp(text)
        
        # Contacts
        emails = self.email_regex.findall(text)
        phones = self.phone_regex.findall(text)
        email = emails[0] if emails else None
        phone = phones[0] if phones else None
        
        # Name estimation (first PERSON entity in top 500 chars)
        name = "Unknown"
        top_doc = nlp(text[:500])
        for ent in top_doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                break

        # Skills
        # [SPEED BOOST 2] Extract all skills in a single regex sweep rather than nested loops
        extracted_skills = list(set(self.syn_to_canon[syn] for syn in found_synonyms))

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "skills": extracted_skills
        }

    def segment_text(self, text: str) -> Dict[str, str]:
        sections = {"contact": "", "skills": "", "experience": "", "education": "", "projects": "", "other": ""}
        current_section = "contact"
        sections[current_section] = []

        for line in text.split('\n'):
            line_str = line.strip()
            if not line_str: continue

            if self.section_headers.match(line_str) or (len(line_str.split()) <= 2 and line_str.isupper()):
                sec_name = line_str.lower()
                if "exp" in sec_name: current_section = "experience"
                elif "edu" in sec_name: current_section = "education"
                elif "skill" in sec_name: current_section = "skills"
                elif "proj" in sec_name: current_section = "projects"
                else: current_section = "other"
                sections[current_section] = []
            else:
                sections[current_section].append(line_str)

        return {k: "\n".join(v) if isinstance(v, list) else v for k, v in sections.items()}

    def parse_resumes(self, files_info: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        files_info: [{"candidate_id": "1", "filename": "x.pdf", "bytes": b"..."}]
        """
        parsed_data = []
        
        # [SPEED BOOST 3] Use ThreadPoolExecutor to process resumes in bulk concurrently
        def process_file(f):
            entities = self.extract_entities(raw_text)
            segments = self.segment_text(raw_text)
            
            exp_match = re.search(r'(\d+)(?:\+| years?)(?: of)? (?:professional )?experience', raw_text.lower())
            years_exp = float(exp_match.group(1)) if exp_match else 0.0
            
            is_intern = bool(re.search(r'\bintern\b|\binternship\b', raw_text.lower()))

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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_file, files_info)
            for row in results:
                parsed_data.append(row)
            
        return pd.DataFrame(parsed_data)