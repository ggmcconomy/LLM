###############################################################################
# method2.py - Refined for RAG Coverage & Better Visuals
#
# This script:
#   - Pulls data from Mural (optional),
#   - Loads the final "clean_risks.csv" from Method 1,
#   - Uses severity/probability to create RAG thresholds,
#   - Color-codes a DataFrame for quick scanning,
#   - Creates pivot charts (stakeholder vs. risk_type) with Altair,
#   - Optionally performs advanced semantic coverage checks with embeddings,
#   - Provides coverage feedback & new risk brainstorming.
###############################################################################

import os
import json
import uuid
import requests
import pandas as pd
import numpy as np
import streamlit as st
import sys
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from datetime import datetime
import altair as alt
import re

# If you get "torch.classes" error with streamlit file watcher, skip it
sys.modules['torch.classes'] = None

from sentence_transformers import SentenceTransformer
import faiss
import openai

###############################################################################
# Streamlit Page Config
###############################################################################
st.set_page_config(page_title="Method 2 - RAG Coverage & Visualization", layout="wide")
st.title("AI Risk Coverage & RAG Dashboard (Method 2)")

###############################################################################
# Load Secrets (e.g., Mural, OpenAI)
###############################################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "myworkspace")
except KeyError as e:
    st.warning(f"Missing secret: {e}. Please ensure .streamlit/secrets.toml is set.")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
    # If you're not using Mural, you can skip.

openai.api_key = OPENAI_API_KEY

###############################################################################
# 1) Utility Functions
###############################################################################
def clean_html_text(html_text):
    """Strip HTML tags and return plain text."""
    if not html_text:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text(separator=" ").strip()
    except Exception as e:
        st.error(f"HTML cleaning error: {str(e)}")
        return ""

def rag_category(row):
    """Assign RAG category based on combined_score or severity/probability."""
    # Example: 
    #   Red = combined_score >= 13
    #   Amber = between 9..12
    #   Green = below 9
    cscore = row.get("combined_score", 0)
    if cscore >= 13:
        return "Red"
    elif cscore >= 9:
        return "Amber"
    else:
        return "Green"

def style_rag(val):
    """Return a color style for a given RAG string."""
    if val == "Red":
        return "background-color: #f8d0d0"  # light red
    elif val == "Amber":
        return "background-color: #fcebcd"  # light orange
    elif val == "Green":
        return "background-color: #c9f5d8"  # light green
    return ""

def style_rag_dataframe(df):
    """Apply color-coding to a DataFrame column named 'rag'."""
    if "rag" in df.columns:
        return df.style.apply(lambda col: [style_rag(v) for v in col] if col.name=="rag" else ["" for _ in col], axis=0)
    return df

def log_feedback(risk_description, user_feedback, reason=""):
    """Log user feedback to CSV or other store."""
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_description": risk_description,
        "feedback": user_feedback,
        "reason": reason
    }
    df = pd.DataFrame([data])
    fname = "feedback_log.csv"
    if os.path.exists(fname):
        old = pd.read_csv(fname)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(fname, index=False)

###############################################################################
# 2) Mural OAuth & Pull (Optional) - same as before
###############################################################################
# ... (you can keep your existing Mural OAuth logic if you want)...

###############################################################################
# 3) Main Coverage Analysis
###############################################################################
st.subheader("Load or Paste Human-Finalized Risks (Optional)")
user_input = st.text_area("Paste your finalized risks from Mural or local input:", height=120, placeholder="One risk per line...")

st.subheader("Load CSV from Method 1 (clean_risks.csv)")
csv_file = st.text_input("CSV File Path", value="clean_risks.csv")
if st.button("Load & Show RAG Table"):
    try:
        df = pd.read_csv(csv_file)
        st.success(f"Loaded {df.shape[0]} lines from {csv_file}.")

        # Ensure we have the columns we need
        required_cols = ["risk_id","risk_description","risk_type","severity","probability","combined_score"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}. Check your CSV from Method 1.")
            st.stop()

        # Assign RAG category
        df["rag"] = df.apply(rag_category, axis=1)

        # Show color-coded table
        st.markdown("### RAG-Assigned Risk Table")
        styled_table = style_rag_dataframe(df)
        st.dataframe(styled_table, use_container_width=True)

        # Let user download the updated CSV (with rag column)
        csv_data = df.to_csv(index=False)
        st.download_button("Download Updated CSV (RAG)", data=csv_data, file_name="clean_risks_with_rag.csv", mime="text/csv")

        # Let's produce a pivot chart or two with Altair:
        st.markdown("### Risk Pivot: Stakeholder vs. Risk Type (avg combined_score)")

        if "stakeholder" in df.columns and "risk_type" in df.columns:
            # Make a pivot. We'll do a groupby then show an altair heatmap
            pivot_data = df.groupby(["stakeholder","risk_type"]).agg({"combined_score":"mean"}).reset_index()
            chart = alt.Chart(pivot_data).mark_rect().encode(
                x=alt.X("risk_type:N", sort=alt.SortField("risk_type", order="ascending")),
                y=alt.Y("stakeholder:N", sort=alt.SortField("stakeholder", order="ascending")),
                color=alt.Color("combined_score:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0, 25], clamp=True)),
                tooltip=["stakeholder","risk_type","combined_score"]
            ).properties(width=500, height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No 'stakeholder' or 'risk_type' columns found for pivot chart.")

        st.markdown("### Scatter: Probability vs. Severity")
        scatter_chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0,5])),
            y=alt.Y("severity:Q", scale=alt.Scale(domain=[0,5])),
            color=alt.Color("rag:N", scale=alt.Scale(domain=["Green","Amber","Red"], range=["green","orange","red"])),
            tooltip=["risk_description","stakeholder","risk_type","severity","probability","combined_score","rag"]
        ).interactive()
        st.altair_chart(scatter_chart, use_container_width=True)

    except FileNotFoundError:
        st.error(f"File not found: {csv_file}")
    except Exception as ex:
        st.error(f"Error loading CSV or generating RAG: {str(ex)}")

###############################################################################
# 4) Semantic Coverage (Optional advanced step)
###############################################################################
st.subheader("Optional: Semantic Coverage Check")
st.write("If you want to check how closely your *human-provided* risks match (or miss) the discovered ones, you can embed them and do a nearest-neighbor search on the final CSV embeddings from Method 1. This is more robust than naive substring matching.")

embed_model_name = st.text_input("Embed model (for coverage check)", value="all-MiniLM-L6-v2")
embeddings_file = st.text_input("Embeddings File (from Method 1)", value="embeddings.npy")
index_file = st.text_input("FAISS Index File (from Method 1)", value="faiss_index.faiss")

if st.button("Run Semantic Coverage Check"):
    if not user_input.strip():
        st.warning("No user input lines found. Paste some above.")
    else:
        # Load embeddings & index
        try:
            existing_df = pd.read_csv(csv_file)
            embeddings = np.load(embeddings_file)
            index = faiss.read_index(index_file)
        except Exception as e:
            st.error(f"Error loading embeddings/index: {str(e)}")
            st.stop()

        # We embed user lines
        lines = [l.strip() for l in user_input.split("\n") if l.strip()]
        embedder = SentenceTransformer(embed_model_name)
        user_embeds = embedder.encode(lines, show_progress_bar=False)
        user_embeds = np.array(user_embeds, dtype="float32")

        # Search nearest neighbors
        k = 3  # top 3 matches
        distances, indices = index.search(user_embeds, k)
        # For each user line, show best matches
        st.markdown("### Semantic Coverage Results")
        coverage_records = []
        for i, line in enumerate(lines):
            st.markdown(f"**User Risk {i+1}:** {line}")
            nn_idx = indices[i]
            nn_dists = distances[i]
            for rank, (idx_n, dist_n) in enumerate(zip(nn_idx, nn_dists), start=1):
                row_info = existing_df.iloc[idx_n]
                st.write(f"Match {rank}: {row_info['risk_description']} (distance={dist_n:.4f}, type={row_info.get('risk_type','N/A')})")
                coverage_records.append({
                    "user_line": line,
                    "match_rank": rank,
                    "matched_risk_id": row_info["risk_id"],
                    "matched_risk_description": row_info["risk_description"],
                    "distance": dist_n
                })
            st.write("---")

        # Optionally convert coverage_records to DataFrame
        df_coverage = pd.DataFrame(coverage_records)
        if not df_coverage.empty:
            st.markdown("#### Download Coverage Matches as CSV")
            st.download_button("Download Coverage CSV", df_coverage.to_csv(index=False), "semantic_coverage.csv", "text/csv")
        else:
            st.info("No coverage matches found or no user lines available.")

###############################################################################
# 5) Coverage Feedback & Brainstorming
###############################################################################
st.subheader("Coverage Feedback & Brainstorming")
if st.button("Generate Coverage Feedback & Suggestions"):
    # Very simple approach: pass the user input + the loaded CSV to GPT for a final summary
    # *This is optional; you may prefer a more advanced approach.*
    try:
        df_loaded = pd.read_csv(csv_file)
        domain = df_loaded['domain'].iloc[0] if 'domain' in df_loaded.columns else "AI domain"
        lines_str = user_input if user_input.strip() else "(no user lines)"

        prompt = f"""
You are a coverage analysis expert for {domain}.
The user-provided lines:
{lines_str}

We have discovered many risks in the CSV (not shown in full).
Provide a short textual feedback on coverage gaps or overlooked angles.
Then suggest 3 new risk ideas that are not obviously covered.
"""
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a coverage analysis expert focusing on AI risk."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        st.markdown("**Coverage Feedback & Suggestions:**")
        st.write(response.choices[0].message.content.strip())

    except Exception as e:
        st.error(f"Error generating coverage feedback: {str(e)}")

st.markdown("---")
st.info("End of Method 2 (Refined). Explore your RAG coverage & pivot charts above, or do further analysis as needed.")
