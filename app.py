import streamlit as st
import pandas as pd
import plotly.express as px

from parser import ResumeParser
from database import VectorDB
from ranker import CandidateRanker
from agent import ChatbotAgent

st.set_page_config(page_title="AI Resume Screening", layout="wide")

@st.cache_resource
def init_system():
    parser = ResumeParser()
    db = VectorDB()
    ranker = CandidateRanker(db, parser)
    agent = ChatbotAgent(db)
    return parser, db, ranker, agent

parser, db, ranker, agent = init_system()

st.title("AI Resume Screening")
st.caption(
    "Upload PDF or DOCX resumes, rank candidates against a job description, and query the candidate pool with a local OpenAI-compatible model."
)

# Sidebar
st.sidebar.title("Data Ingestion")
uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF/DOCX)", accept_multiple_files=True)
if st.sidebar.button("Ingest Resumes"):
    if uploaded_files:
        files_info = []
        for i, f in enumerate(uploaded_files):
            files_info.append({
                "candidate_id": f"C_{i+1}_{f.name}",
                "filename": f.name,
                "bytes": f.getvalue()
            })
        try:
            with st.spinner("Parsing and indexing resumes..."):
                df_parsed = parser.parse_resumes(files_info)
                db.ingest_parsed_df(df_parsed)
            st.sidebar.success(f"Ingested {len(uploaded_files)} resumes.")
        except Exception as exc:
            st.sidebar.error(f"Failed to ingest resumes: {exc}")
    else:
        st.sidebar.warning("Upload resumes first.")

job_desc = st.sidebar.text_area("Job Description", "Looking for Python and SQL engineers with ML projects.")
req_exp = st.sidebar.number_input("Required Years of Exp", min_value=0.0, value=2.0)

# Main UI
tab1, tab2 = st.tabs(["Rankings Dashboard", "RAG Chatbot"])

with tab1:
    st.header("Ranked Candidates")
    if st.button("Score & Rank Candidates"):
        try:
            with st.spinner("Calculating candidate scores..."):
                ranked_df = ranker.rank_candidates(job_desc, req_years_exp=req_exp)
                if not ranked_df.empty:
                    st.dataframe(ranked_df)

                    # Plotly chart of missing skills across the ranked candidates.
                    all_missing = []
                    for ms in ranked_df["missing_skills"]:
                        all_missing.extend(ms)

                    if all_missing:
                        missing_s = pd.Series(all_missing).value_counts().reset_index()
                        missing_s.columns = ["Skill", "Count"]
                        fig = px.bar(missing_s, x="Skill", y="Count", title="Missing Skills Distribution")
                        st.plotly_chart(fig)
                else:
                    st.info("No candidates ranked yet or the database is empty.")
        except Exception as exc:
            st.error(f"Failed to rank candidates: {exc}")

with tab2:
    st.header("Candidate Assistant")
    st.write("Ask questions like: `Find Python candidates with AWS experience` or `Show interns with ML projects`.")
    query = st.chat_input("Ask about candidates (e.g. Find Python interns with ML projects)")
    if query:
        st.chat_message("user").write(query)
        with st.spinner("Agent is searching and synthesizing..."):
            try:
                response = agent.chat(query)
                st.chat_message("assistant").write(response)
            except Exception:
                st.chat_message("assistant").write(
                    "I couldn't reach the configured LLM. Check the provider, base URL, API key, and model settings."
                )
