#############################
# method2.py
#
# Purpose:
#  - Pull final discovered risks from "clean_risks.csv" (Method 1 output).
#  - Optionally pull "human workshop" risks from Mural or text area.
#  - Compare user-provided final lines vs. discovered set => identify coverage gaps.
#  - Show coverage bar charts, color-coded RAG table, short gap feedback from GPT (optional).
#############################

import os
import sys
import uuid
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from collections import Counter

# For embeddings + similarity search (optional)
from sentence_transformers import SentenceTransformer
import faiss

# For optional LLM-based feedback
import openai

#############################
# 1) Basic Setup
#############################
st.set_page_config(page_title="Method 2 - Coverage & Gaps", layout="wide")
st.title("AI Risk Analysis Dashboard (Method 2)")

# Load secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY

    # Mural config
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "someworkspace")
except KeyError as e:
    st.error(f"Missing secret: {e}. Check your .streamlit/secrets.toml.")
    st.stop()

plt.style.use('ggplot')

#############################
# 2) Mural OAuth Helpers (Optional)
#############################
def get_authorization_url():
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    return "https://app.mural.co/api/public/v1/authorization/oauth2/?" + urlencode(params)

def exchange_code_for_token(code):
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "redirect_uri": MURAL_REDIRECT_URI,
        "code": code,
        "grant_type": "authorization_code"
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Auth failed: {r.status_code}")
            return None
    except Exception as e:
        st.error(f"Auth error: {str(e)}")
        return None

def refresh_access_token(refresh_token):
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Refresh token failed: {r.status_code}")
            return None
    except Exception as e:
        st.error(f"Token refresh error: {str(e)}")
        return None

# Initialize session tokens
if "mural_access_token" not in st.session_state:
    st.session_state["mural_access_token"] = None
    st.session_state["mural_refresh_token"] = None
    st.session_state["mural_expires_in"] = None
    st.session_state["mural_token_ts"] = None

# Check query params for mural code=...
params = st.experimental_get_query_params()
auth_code = params.get("code")
if auth_code and not st.session_state["mural_access_token"]:
    token_data = exchange_code_for_token(auth_code[0])
    if token_data:
        st.session_state["mural_access_token"] = token_data["access_token"]
        st.session_state["mural_refresh_token"] = token_data.get("refresh_token")
        st.session_state["mural_expires_in"] = token_data.get("expires_in", 900)
        st.session_state["mural_token_ts"] = datetime.now().timestamp()
        st.experimental_set_query_params()  # clear
        st.success("Mural authenticated!")
        st.experimental_rerun()

# If we have a token, check if near expiry
if st.session_state["mural_access_token"]:
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state["mural_token_ts"]) > (st.session_state["mural_expires_in"] - 60):
        # Refresh
        new_data = refresh_access_token(st.session_state["mural_refresh_token"])
        if new_data:
            st.session_state["mural_access_token"] = new_data["access_token"]
            st.session_state["mural_refresh_token"] = new_data.get("refresh_token", st.session_state["mural_refresh_token"])
            st.session_state["mural_expires_in"] = new_data.get("expires_in", 900)
            st.session_state["mural_token_ts"] = datetime.now().timestamp()

#############################
# 3) Functions
#############################
def clean_html_text(html_text):
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ").strip()

def pull_mural_stickies(mural_id, auth_token):
    """
    Pull sticky notes from a Mural board.
    Return list of plain text lines.
    """
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    try:
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        s.mount('https://', HTTPAdapter(max_retries=retries))
        resp = s.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            widgets = resp.json().get("value", [])
            notes = []
            for w in widgets:
                # check if sticky note
                wtype = w.get("type","").lower().replace(" ","_")
                if wtype == "sticky_note":
                    raw_text = w.get("text") or w.get("htmlText") or ""
                    txt = clean_html_text(raw_text)
                    if txt:
                        notes.append(txt)
            return notes
        else:
            st.error(f"Failed pulling mural widgets: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Error pulling mural data: {str(e)}")
        return []

def create_side_by_side_bar_chart(title, categories, covered_counts, missed_counts, x_label, y_label, filename):
    """
    Produce side-by-side coverage bar chart.
    """
    try:
        plt.figure(figsize=(6,4))
        x = np.arange(len(categories))
        width = 0.4
        plt.bar(x - width/2, covered_counts, width, label='Covered', color='green')
        plt.bar(x + width/2, missed_counts, width, label='Missed', color='red')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return True
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        return False

def color_cell(count):
    """
    Return a RAG color style based on coverage count.
    Suppose we consider:
      0 => Red
      1..3 => Amber
      >3 => Green
    Customize as needed.
    """
    if count == 0:
        return "background-color: #e74c3c; color:white"  # red
    elif 1 <= count <= 3:
        return "background-color: #f39c12; color:white"  # amber
    else:
        return "background-color: #2ecc71; color:white"  # green

#############################
# 4) Load "clean_risks.csv" from Method 1
#############################
csv_file = "clean_risks.csv"  # align with method1 naming
embeddings_file = "embeddings.npy"
index_file = "faiss_index.faiss"

try:
    df_risks = pd.read_csv(csv_file)
    # ensure columns exist
    required_cols = ["risk_description","risk_type","node_name","cluster"]
    for c in required_cols:
        if c not in df_risks.columns:
            st.warning(f"Column '{c}' not found in {csv_file}. Some coverage checks might not work.")
    # fill missing
    df_risks.fillna("", inplace=True)

except FileNotFoundError:
    st.error(f"File {csv_file} not found. Please run Method 1 first.")
    st.stop()

# We'll also optionally set up embeddings if we want to do similarity search
try:
    embed_np = np.load(embeddings_file)
    index = faiss.read_index(index_file)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except:
    st.warning("No embeddings or FAISS index found. Similarity-based coverage might be disabled.")
    index = None
    embedder = None

#############################
# 5) Gather User's Human Analysis
#############################
st.sidebar.subheader("Mural Options")
mural_override = st.sidebar.text_input("Mural ID override", value=MURAL_BOARD_ID)
use_mural = st.sidebar.checkbox("Pull from Mural?", value=False)
if use_mural and st.session_state["mural_access_token"]:
    if st.button("Pull Sticky Notes"):
        notes = pull_mural_stickies(mural_override, st.session_state["mural_access_token"])
        if notes:
            st.session_state["mural_notes"] = notes
            st.success(f"Pulled {len(notes)} sticky notes from Mural!")
        else:
            st.warning("No sticky notes found or Mural pull failed.")
else:
    st.sidebar.write("Authenticate with Mural first or uncheck 'Pull from Mural?'")

# main input
st.subheader("1) Provide Your Final Workshop Risks")
default_text = ""
if "mural_notes" in st.session_state:
    default_text = "\n".join(st.session_state["mural_notes"])
user_risks_text = st.text_area("Paste your final lines (one per line):", value=default_text, height=150)

#############################
# 6) Coverage Analysis
#############################
st.subheader("2) Coverage Analysis")

def analyze_coverage(user_lines, ref_df):
    """
    Compare user lines to reference risks in ref_df to see coverage in categories:
      - risk_type
      - node_name (stakeholder)
      - cluster
    We'll do a simple match: if a user line (or set) includes a phrase similar to reference?
    Or we can do embedding-based nearest neighbors. We'll do a naive approach or embedding if available.
    Return coverage stats as dictionary.
    """
    coverage_results = {
        "risk_types_covered": set(),
        "stakeholders_covered": set(),
        "clusters_covered": set()
    }
    if not user_lines:
        return coverage_results

    # If we have embeddings set up, do a nearest-neighbor approach
    if index and embedder:
        user_embs = np.array(embedder.encode(user_lines), dtype="float32")
        k = 3  # top-3 neighbors
        D,I = index.search(user_embs, k)
        # gather coverage from neighbors
        for row_i in range(len(user_lines)):
            nn_idx = I[row_i]
            subdf = ref_df.iloc[nn_idx]
            coverage_results["risk_types_covered"].update(subdf["risk_type"].unique())
            coverage_results["stakeholders_covered"].update(subdf["node_name"].unique())
            coverage_results["clusters_covered"].update(subdf["cluster"].unique())
    else:
        # fallback: naive substring matching (very crude)
        all_risk_types = ref_df["risk_type"].unique()
        all_stakeholders = ref_df["node_name"].unique()
        all_clusters = ref_df["cluster"].unique()

        # For each user line, just see if it contains substring of risk_type, etc.
        # or skip substring approach
        # We'll do a naive approach to show concept:
        lines_text = " ".join(user_lines).lower()
        for rt in all_risk_types:
            if rt.lower() in lines_text:
                coverage_results["risk_types_covered"].add(rt)
        for sh in all_stakeholders:
            if sh.lower() in lines_text:
                coverage_results["stakeholders_covered"].add(sh)
        for cl in all_clusters:
            if str(cl).lower() in lines_text:
                coverage_results["clusters_covered"].add(cl)

    return coverage_results

def create_rag_table(covered_items, all_items, col_name):
    """
    Create a small DataFrame that has [Item, CoverageCount] and color it with a RAG scale.
    For simplicity, we'll just do a 'covered=1 or 0' approach. Or we can do a numeric count.
    """
    data = []
    for item in sorted(all_items):
        c = (1 if item in covered_items else 0)
        data.append((item, c))
    df_ = pd.DataFrame(data, columns=[col_name,"CoverageCount"])
    return df_

if st.button("Analyze Coverage"):
    user_lines = [ln.strip() for ln in user_risks_text.split("\n") if ln.strip()]
    if not user_lines:
        st.warning("No final lines found. Provide some lines first.")
    else:
        # gather coverage
        coverage = analyze_coverage(user_lines, df_risks)

        # risk_types
        all_types = df_risks["risk_type"].unique()
        missed_types = set(all_types) - coverage["risk_types_covered"]

        # stakeholders
        all_sh = df_risks["node_name"].unique()
        missed_sh = set(all_sh) - coverage["stakeholders_covered"]

        # clusters
        all_cl = df_risks["cluster"].unique()
        missed_cl = set(all_cl) - coverage["clusters_covered"]

        st.write("### Coverage Summary")
        st.write(f"- **Risk Types Covered**: {sorted(list(coverage['risk_types_covered']))}")
        st.write(f"- **Stakeholders Covered**: {sorted(list(coverage['stakeholders_covered']))}")
        st.write(f"- **Clusters Covered**: {sorted(list(coverage['clusters_covered']))}")

        st.write("### Missed Coverage")
        st.write(f"- **Missed Risk Types**: {sorted(list(missed_types))}")
        st.write(f"- **Missed Stakeholders**: {sorted(list(missed_sh))}")
        st.write(f"- **Missed Clusters**: {sorted(list(missed_cl))}")

        # Let's do bar charts for risk_type and stakeholder coverage
        # e.g. "Risk Type Coverage"
        # We'll count how many in coverage, how many missed
        def coverage_counts(all_items, covered_set):
            cov_c = []
            mis_c = []
            for item in all_items:
                # if item is covered => covered=1, missed=0, or if you want counts
                if item in covered_set:
                    cov_c.append(1)
                    mis_c.append(0)
                else:
                    cov_c.append(0)
                    mis_c.append(1)
            return cov_c, mis_c

        # bar chart for risk_type
        cat_types = sorted(all_types)
        cov_t, mis_t = coverage_counts(cat_types, coverage["risk_types_covered"])
        if create_side_by_side_bar_chart(
            title="Risk Type Coverage",
            categories=cat_types,
            covered_counts=cov_t,
            missed_counts=mis_t,
            x_label="Risk Type",
            y_label="Count",
            filename="risk_type_coverage.png"
        ):
            st.image("risk_type_coverage.png", use_column_width=True)

        # bar chart for stakeholder coverage
        cat_sh = sorted(all_sh)
        cov_s, mis_s = coverage_counts(cat_sh, coverage["stakeholders_covered"])
        if create_side_by_side_bar_chart(
            title="Stakeholder Coverage",
            categories=cat_sh,
            covered_counts=cov_s,
            missed_counts=mis_s,
            x_label="Stakeholder",
            y_label="Count",
            filename="stakeholder_coverage.png"
        ):
            st.image("stakeholder_coverage.png", use_column_width=True)

        # Let's do a small RAG table for risk_type coverage
        st.write("### RAG Table - Risk Types")
        df_rag_types = create_rag_table(coverage["risk_types_covered"], all_types, "RiskType")
        def style_rag(df_):
            return [
                color_cell(row["CoverageCount"]) for _, row in df_.iterrows()
            ]
        st.dataframe(
            df_rag_types.style.apply(style_rag, axis=1, subset=["CoverageCount"])
        )

        # If you want an LLM-based short gap feedback, you can do something like:
        if st.checkbox("Generate LLM Gap Feedback?"):
            gap_prompt = f"""
You are an AI assistant reviewing an AI risk workshop's final lines.
We have discovered coverage on these dimensions:

- Risk Types covered: {sorted(list(coverage['risk_types_covered']))}
- Missing risk types: {sorted(list(missed_types))}
- Stakeholders covered: {sorted(list(coverage['stakeholders_covered']))}
- Missing stakeholders: {sorted(list(missed_sh))}
- Clusters covered: {sorted(list(coverage['clusters_covered']))}
- Missing clusters: {sorted(list(missed_cl))}

Please provide a concise explanation of:
1. Why these missing areas might be important.
2. Suggestions on how to add or expand the analysis in these missed or under-covered categories.
3. Positive note on what's adequately covered so far.
"""
            try:
                resp = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role":"system","content":"You are a helpful coverage feedback assistant."},
                        {"role":"user","content": gap_prompt}
                    ],
                    max_tokens=300
                )
                st.markdown("#### LLM Gap Feedback")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {str(e)}")

#############################
# 7) Done / Reset
#############################
if st.button("Reset Session"):
    for k in ["mural_access_token","mural_refresh_token","mural_expires_in","mural_token_ts","mural_notes"]:
        if k in st.session_state:
            st.session_state.pop(k)
    st.experimental_rerun()

st.info("End of Method 2 demonstration script. Customize as desired.")
