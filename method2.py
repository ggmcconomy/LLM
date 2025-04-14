################################################
# method2.py
#
# Purpose:
#  - Single-tab Mural OAuth (no separate window),
#  - Pull final discovered risks from "clean_risks.csv" (Method 1 output).
#  - Optionally pull final workshop lines from Mural or text area.
#  - Analyze coverage vs. risk_type, stakeholders (node_name), clusters.
#  - Generate coverage bar charts, RAG table, and optional LLM feedback.
################################################

import os
import sys
import uuid
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.parse import urlencode
from collections import Counter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# For optional embeddings + similarity
from sentence_transformers import SentenceTransformer
import faiss

# For LLM-based feedback (if desired)
import openai

################################################
# 1) PAGE SETUP
################################################
st.set_page_config(page_title="Method 2 - Single-Tab Mural Auth", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2)")

plt.style.use('ggplot')

################################################
# 2) LOAD SECRETS
################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY

    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    # If you have a default board/workspace
    MURAL_BOARD_ID = st.secrets.get("MURAL_BOARD_ID","")
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID","")
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure .streamlit/secrets.toml properly.")
    st.stop()

################################################
# 3) MURAL OAUTH LOGIC (SINGLE TAB)
################################################
def get_mural_auth_url():
    """Generate link to Mural's OAuth in the same tab."""
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "response_type": "code",
        "state": "some-unique-state-123"
    }
    return "https://app.mural.co/api/public/v1/authorization/oauth2/?" + urlencode(params)

def exchange_code_for_token(auth_code):
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "redirect_uri": MURAL_REDIRECT_URI,
        "code": auth_code,
        "grant_type": "authorization_code"
    }
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Mural auth failed: {resp.status_code}")
            return None
    except Exception as ex:
        st.error(f"Mural auth exception: {str(ex)}")
        return None

def refresh_mural_token(refresh_token):
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
            st.error(f"Mural refresh token error: {r.status_code}")
            return None
    except Exception as e:
        st.error(f"Mural token refresh exception: {str(e)}")
        return None

# Keep tokens in session state
if "mural_access_token" not in st.session_state:
    st.session_state["mural_access_token"] = None
    st.session_state["mural_refresh_token"] = None
    st.session_state["mural_expires_in"] = None
    st.session_state["mural_token_ts"] = None

# Check if user was redirected back with ?code=...
qs = st.experimental_get_query_params()
auth_code = qs.get("code")
if auth_code and not st.session_state["mural_access_token"]:
    # code is typically a list, use first
    code_val = auth_code[0]
    if code_val:
        token_data = exchange_code_for_token(code_val)
        if token_data:
            st.session_state["mural_access_token"] = token_data["access_token"]
            st.session_state["mural_refresh_token"] = token_data.get("refresh_token")
            st.session_state["mural_expires_in"] = token_data.get("expires_in",900)
            st.session_state["mural_token_ts"] = datetime.now().timestamp()
            # Clear code from URL
            st.experimental_set_query_params()
            st.success("Authenticated with Mural in the same tab!")
            st.experimental_rerun()

# If we have a token, check if near expiry
if st.session_state["mural_access_token"]:
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state["mural_token_ts"]) > (st.session_state["mural_expires_in"] - 60):
        # Refresh
        new_data = refresh_mural_token(st.session_state["mural_refresh_token"])
        if new_data:
            st.session_state["mural_access_token"] = new_data["access_token"]
            st.session_state["mural_refresh_token"] = new_data.get("refresh_token", st.session_state["mural_refresh_token"])
            st.session_state["mural_expires_in"] = new_data.get("expires_in",900)
            st.session_state["mural_token_ts"] = datetime.now().timestamp()

################################################
# 4) HELPER: PULL STICKY NOTES FROM MURAL
################################################
def pull_mural_stickies(mural_id, token):
    import json
    from bs4 import BeautifulSoup
    def clean_html_text(html_text):
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text(separator=" ").strip()

    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {"Authorization": f"Bearer {token}", "Accept":"application/json"}
    try:
        sess = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        sess.mount('https://', HTTPAdapter(max_retries=retries))
        resp = sess.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            js = resp.json()
            widgets = js.get("value", []) or js.get("data", [])
            lines = []
            for w in widgets:
                wtype = w.get("type","").lower().replace(" ","_")
                if wtype == "sticky_note":
                    raw = w.get("text") or w.get("htmlText") or ""
                    txt = clean_html_text(raw)
                    if txt:
                        lines.append(txt)
            return lines
        else:
            st.error(f"Pull Mural error: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Mural sticky pull exception: {str(e)}")
        return []

################################################
# 5) SIDEBAR
################################################
st.sidebar.header("Mural OAuth")
if not st.session_state["mural_access_token"]:
    st.sidebar.write("Not authorized with Mural yet.")
    auth_url = get_mural_auth_url()
    # Provide link in same tab
    st.sidebar.markdown(f"[Authorize with Mural (same tab)]({auth_url})")
else:
    st.sidebar.success("Token in session. Ready to pull notes.")

mural_override = st.sidebar.text_input("Mural Board ID", value=MURAL_BOARD_ID)
if st.sidebar.button("Pull Sticky Notes"):
    if not st.session_state["mural_access_token"]:
        st.sidebar.warning("No token. Authorize first.")
    else:
        lines = pull_mural_stickies(mural_override, st.session_state["mural_access_token"])
        if lines:
            st.session_state["mural_notes"] = lines
            st.sidebar.success(f"Pulled {len(lines)} notes!")
        else:
            st.sidebar.warning("No notes found or error.")


################################################
# 6) LOAD CLEAN_RISKS (Method 1 Output)
################################################
csv_file = "clean_risks.csv"
embeddings_file = "embeddings.npy"
faiss_file = "faiss_index.faiss"

try:
    df_risks = pd.read_csv(csv_file)
    st.write(f"Loaded {len(df_risks)} risks from {csv_file}.")
    if "risk_type" not in df_risks.columns or "node_name" not in df_risks.columns or "cluster" not in df_risks.columns:
        st.warning("Column risk_type / node_name / cluster not found in CSV. Some coverage checks may be limited.")
except FileNotFoundError:
    st.error(f"Cannot find {csv_file}. Please run Method 1 first.")
    st.stop()

try:
    embed_np = np.load(embeddings_file)
    index = faiss.read_index(faiss_file)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    st.write("Embeddings & FAISS index loaded. We'll do nearest-neighbor coverage checks.")
except:
    st.warning("No embeddings or FAISS index found; fallback coverage logic will be naive substring matching.")
    embedder = None
    index = None

################################################
# 7) Gather User's Human Workshop Risks
################################################
st.subheader("1) Provide Your Final Workshop Risks")

default_text = ""
if "mural_notes" in st.session_state:
    default_text = "\n".join(st.session_state["mural_notes"])

user_input = st.text_area("Enter your final lines (one per line):", value=default_text, height=160)

################################################
# 8) Coverage Analysis
################################################
st.subheader("2) Coverage Analysis & Gaps")

def do_coverage_analysis(user_lines, df_ref):
    """
    We'll gather coverage across risk_type, node_name, cluster
    using either embeddings or naive approach.
    """
    coverage_dict = {
        "risk_type": set(),
        "node_name": set(),
        "cluster": set()
    }
    if not user_lines:
        return coverage_dict

    if embedder and index:
        # embedding approach
        user_embs = np.array(embedder.encode(user_lines), dtype="float32")
        k = 3  # nearest 3
        D,I = index.search(user_embs, k)
        for i, line in enumerate(user_lines):
            neighbors_idx = I[i]
            subdf = df_ref.iloc[neighbors_idx]
            coverage_dict["risk_type"].update(subdf["risk_type"].unique())
            coverage_dict["node_name"].update(subdf["node_name"].unique())
            coverage_dict["cluster"].update(subdf["cluster"].unique())
    else:
        # naive substring approach
        combined_text = " ".join(line.lower() for line in user_lines)
        # gather all sets
        for rt in df_ref["risk_type"].unique():
            if str(rt).lower() in combined_text:
                coverage_dict["risk_type"].add(rt)
        for nm in df_ref["node_name"].unique():
            if str(nm).lower() in combined_text:
                coverage_dict["node_name"].add(nm)
        for cl in df_ref["cluster"].unique():
            if str(cl).lower() in combined_text:
                coverage_dict["cluster"].add(cl)
    return coverage_dict

def create_side_by_side_chart(title, all_items, covered_set, filename):
    """
    For each item in all_items, covered => 1, missed => 1 in the other bar.
    """
    cat_list = sorted(set(all_items))
    covered_vals = []
    missed_vals = []
    for c in cat_list:
        if c in covered_set:
            covered_vals.append(1)
            missed_vals.append(0)
        else:
            covered_vals.append(0)
            missed_vals.append(1)

    # create bar chart
    try:
        plt.figure(figsize=(6,4))
        x = np.arange(len(cat_list))
        width = 0.4
        plt.bar(x - width/2, covered_vals, width, color='green', label='Covered')
        plt.bar(x + width/2, missed_vals, width, color='red', label='Missed')
        plt.title(title)
        plt.xticks(x, cat_list, rotation=45, ha='right')
        plt.ylabel("Count (1=present,0=absent)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return True
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        return False

def color_cell(val):
    """RAG scale: 0=red, 1..2=amber, >=3=green (example)"""
    if val == 0:
        return "background-color: #e74c3c; color:white"
    elif 1 <= val <=2:
        return "background-color: #f39c12; color:white"
    else:
        return "background-color: #2ecc71; color:white"

if st.button("Analyze Coverage"):
    lines = [ln.strip() for ln in user_input.split("\n") if ln.strip()]
    if not lines:
        st.warning("No lines to analyze. Provide final lines first.")
    else:
        cov = do_coverage_analysis(lines, df_risks)

        # Summaries
        st.write("### Coverage Results")
        # all possible items
        all_types = df_risks["risk_type"].unique()
        all_nodes = df_risks["node_name"].unique()
        all_clusters = df_risks["cluster"].unique()

        missed_types = set(all_types) - cov["risk_type"]
        missed_nodes = set(all_nodes) - cov["node_name"]
        missed_clusters = set(all_clusters) - cov["cluster"]

        st.write(f"- Covered Risk Types: {sorted(list(cov['risk_type']))}")
        st.write(f"- Missed Risk Types: {sorted(list(missed_types))}")
        st.write(f"- Covered Stakeholders (node_name): {sorted(list(cov['node_name']))}")
        st.write(f"- Missed Stakeholders: {sorted(list(missed_nodes))}")
        st.write(f"- Covered Clusters: {sorted(list(cov['cluster']))}")
        st.write(f"- Missed Clusters: {sorted(list(missed_clusters))}")

        # bar charts
        # 1) risk_type coverage chart
        ok1 = create_side_by_side_chart("Risk Type Coverage", all_types, cov["risk_type"], "risk_type.png")
        if ok1:
            st.image("risk_type.png", use_column_width=True)
        # 2) stakeholder coverage
        ok2 = create_side_by_side_chart("Stakeholder Coverage", all_nodes, cov["node_name"], "stakeholder.png")
        if ok2:
            st.image("stakeholder.png", use_column_width=True)
        # 3) cluster coverage
        ok3 = create_side_by_side_chart("Cluster Coverage", all_clusters, cov["cluster"], "cluster.png")
        if ok3:
            st.image("cluster.png", use_column_width=True)

        # optional RAG table for each category
        st.write("### RAG Table: Risk Types")
        rag_data = []
        for t in sorted(all_types):
            # We'll define coverage_count = 1 if t in covered, else 0
            ccount = 1 if t in cov["risk_type"] else 0
            rag_data.append((t, ccount))
        df_rag = pd.DataFrame(rag_data, columns=["RiskType","CoverageCount"])

        def rag_coloring(row):
            cval = row["CoverageCount"]
            return color_cell(cval)

        st.dataframe(
            df_rag.style.apply(lambda r: [rag_coloring(r)], axis=1, subset=["CoverageCount"])
        )

        # If you want an LLM-based gap feedback
        if st.checkbox("Generate LLM Gap Feedback"):
            prompt = f"""
You are an AI coverage assistant. The user provided these lines:
{chr(10).join("- "+l for l in lines)}

We discovered coverage on:
 - risk_types covered: {sorted(list(cov['risk_type']))}
 - missed risk_types: {sorted(list(missed_types))}
 - stakeholders covered: {sorted(list(cov['node_name']))}
 - missed stakeholders: {sorted(list(missed_nodes))}
 - clusters covered: {sorted(list(cov['cluster']))}
 - missed clusters: {sorted(list(missed_clusters))}

Explain why the missing areas might be critical, and how the user might address them, in a short helpful message.
"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"You are a helpful coverage analysis assistant."},
                        {"role":"user","content":prompt}
                    ],
                    max_tokens=300
                )
                st.markdown("#### LLM Gap Feedback:")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {str(e)}")

################################################
# 9) End / Reset
################################################
if st.button("Reset Session"):
    for k in ["mural_access_token","mural_refresh_token","mural_expires_in","mural_token_ts","mural_notes"]:
        if k in st.session_state:
            st.session_state.pop(k)
    st.experimental_rerun()

st.info("Done. This script uses single-tab Mural OAuth and coverage analysis for Method 2.")
