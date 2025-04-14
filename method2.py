#############################################################
# method2.py
#
# Integrates "better" Mural OAuth from your snippet,
# but now uses "clean_risks.csv" from Method 1.
# Provides coverage & gap analysis with bar charts,
# LLM-based feedback, and an optional approach to subtypes/stakeholders.
#############################################################

import os
import sys
import uuid
import requests
import pandas as pd
import numpy as np
import streamlit as st
import re
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.parse import urlencode
from collections import Counter

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# For embeddings + clustering coverage (optional)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

# For LLM if you want textual feedback
try:
    import openai
    from openai import OpenAI
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False

#############################################################
# 1) Page Setup
#############################################################
st.set_page_config(page_title="Method 2 - Mural + Coverage", layout="wide")
st.title("🤖 AI Risk Analysis Dashboard - Method 2 (Improved Mural Connection)")

plt.style.use('ggplot')

#############################################################
# 2) Load Secrets
#############################################################
try:
    # OpenAI
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Mural
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID","someworkspace")
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure .streamlit/secrets.toml.")
    st.stop()

#############################################################
# 3) Utility Functions
#############################################################
def normalize_mural_id(mural_id, workspace_id=MURAL_WORKSPACE_ID):
    """Strip workspace prefix from mural ID if present."""
    prefix = f"{workspace_id}."
    if mural_id.startswith(prefix):
        return mural_id[len(prefix):]
    return mural_id

def clean_html_text(html_text):
    """Strip HTML tags and clean text from Mural notes."""
    if not html_text:
        return ""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text(separator=" ").strip()
        return text if text else ""
    except Exception as e:
        st.error(f"Error cleaning HTML: {str(e)}")
        return ""

def log_feedback(risk_description, user_feedback, disagreement_reason=""):
    """Log user feedback to a local CSV."""
    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_description": risk_description,
        "user_feedback": user_feedback,
        "disagreement_reason": disagreement_reason
    }
    feedback_df = pd.DataFrame([feedback_data])
    feedback_file = "feedback_log.csv"
    try:
        if os.path.exists(feedback_file):
            existing_df = pd.read_csv(feedback_file)
            feedback_df = pd.concat([existing_df, feedback_df], ignore_index=True)
        feedback_df.to_csv(feedback_file, index=False)
    except Exception as e:
        st.error(f"Error logging feedback: {str(e)}")

def create_coverage_chart(title, categories, covered_counts, missed_counts, filename):
    """Side-by-side coverage bar chart."""
    try:
        plt.figure(figsize=(6, 4))
        x = np.arange(len(categories))
        width = 0.4
        plt.bar(x - width/2, covered_counts, width, label='Covered', color='#2ecc71')
        plt.bar(x + width/2, missed_counts, width, label='Missed', color='#e74c3c')
        plt.xlabel(title.split(' ')[-1])
        plt.ylabel('Count')
        plt.title(title)
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return True
    except Exception as e:
        st.error(f"Error creating chart {filename}: {str(e)}")
        return False

def create_coverage_charts(covered_stakeholders, missed_stakeholders,
                           covered_types, missed_types,
                           covered_subtypes, missed_subtypes,
                           top_n_subtypes=5):
    """
    Generate bar charts for coverage (stakeholder, type, subtypes).
    """
    # 1) Stakeholder chart
    stakeholders = sorted(set(covered_stakeholders + missed_stakeholders))
    covered_counts = [covered_stakeholders.count(s) for s in stakeholders]
    missed_counts = [missed_stakeholders.count(s) for s in stakeholders]
    nonzero = [i for i,(c,m) in enumerate(zip(covered_counts, missed_counts)) if c>0 or m>0]
    stakeholders = [stakeholders[i] for i in nonzero]
    covered_counts = [covered_counts[i] for i in nonzero]
    missed_counts = [missed_counts[i] for i in nonzero]
    if stakeholders:
        create_coverage_chart("Stakeholder Coverage Gaps", stakeholders, covered_counts, missed_counts, 'stakeholder_coverage.png')

    # 2) Risk Type chart
    risk_types = sorted(set(covered_types + missed_types))
    covered_counts = [covered_types.count(t) for t in risk_types]
    missed_counts = [missed_types.count(t) for t in risk_types]
    nonzero = [i for i,(c,m) in enumerate(zip(covered_counts, missed_counts)) if c>0 or m>0]
    risk_types = [risk_types[i] for i in nonzero]
    covered_counts = [covered_counts[i] for i in nonzero]
    missed_counts = [missed_counts[i] for i in nonzero]
    if risk_types:
        create_coverage_chart("Risk Type Coverage Gaps", risk_types, covered_counts, missed_counts, 'risk_type_coverage.png')

    # 3) Subtypes chart
    subtype_counts = Counter(missed_subtypes)
    top_missed_subs = [s for s,_ in subtype_counts.most_common(top_n_subtypes)]
    covered_counts = [covered_subtypes.count(s) for s in top_missed_subs]
    missed_counts = [missed_subtypes.count(s) for s in top_missed_subs]
    if top_missed_subs:
        create_coverage_chart(f"Top {top_n_subtypes} Overlooked Risk Subtype Gaps",
                              top_missed_subs, covered_counts, missed_counts,
                              'risk_subtype_coverage.png')

#############################################################
# 4) OAuth Flow
#############################################################
def get_authorization_url():
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    return f"https://app.mural.co/api/public/v1/authorization/oauth2/?{urlencode(params)}"

def exchange_code_for_token(code):
    with st.spinner("Authenticating with Mural..."):
        url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
        data = {
            "client_id": MURAL_CLIENT_ID,
            "client_secret": MURAL_CLIENT_SECRET,
            "redirect_uri": MURAL_REDIRECT_URI,
            "code": code,
            "grant_type": "authorization_code"
        }
        try:
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Authentication failed: {resp.status_code}")
                return None
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return None

def refresh_access_token(refresh_token):
    with st.spinner("Refreshing Mural token..."):
        url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
        data = {
            "client_id": MURAL_CLIENT_ID,
            "client_secret": MURAL_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        try:
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Token refresh failed: {resp.status_code}")
                return None
        except Exception as e:
            st.error(f"Token refresh error: {str(e)}")
            return None

def list_murals(auth_token):
    url = "https://app.mural.co/api/public/v1/murals"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {auth_token}"}
    try:
        s = requests.Session()
        rtry = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        s.mount('https://', HTTPAdapter(max_retries=rtry))
        resp = s.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("value", [])
        else:
            st.error(f"Failed to list murals: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Error listing murals: {str(e)}")
        return []

def verify_mural(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    headers = {"Accept":"application/json","Authorization":f"Bearer {auth_token}"}
    try:
        s = requests.Session()
        rtry = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        s.mount('https://', HTTPAdapter(max_retries=rtry))
        resp = s.get(url, headers=headers, timeout=10)
        return (resp.status_code == 200)
    except Exception as e:
        st.error(f"Error verifying mural: {str(e)}")
        return False

# Keep tokens in session
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
    st.session_state["refresh_token"] = None
    st.session_state["token_expires_in"] = None
    st.session_state["token_timestamp"] = None

# Check if user came back with ?code=...
qs = st.query_params
auth_code = qs.get("code")
if auth_code and not st.session_state["access_token"]:
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state["access_token"] = token_data["access_token"]
        st.session_state["refresh_token"] = token_data.get("refresh_token")
        st.session_state["token_expires_in"] = token_data.get("expires_in",900)
        st.session_state["token_timestamp"] = datetime.now().timestamp()
        st.set_query_params()  # remove code
        st.success("Authenticated with Mural!")
        st.rerun()

# If we have token, possibly refresh
if st.session_state["access_token"]:
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state["token_timestamp"]) > (st.session_state["token_expires_in"] - 60):
        # Refresh
        token_data = refresh_access_token(st.session_state["refresh_token"])
        if token_data:
            st.session_state["access_token"] = token_data["access_token"]
            st.session_state["refresh_token"] = token_data.get("refresh_token", st.session_state["refresh_token"])
            st.session_state["token_expires_in"] = token_data.get("expires_in",900)
            st.session_state["token_timestamp"] = datetime.now().timestamp()

# If still no token, user must authorize
if not st.session_state["access_token"]:
    auth_url = get_authorization_url()
    st.markdown(f"Please [authorize this app]({auth_url}) to access Mural. Then reload the same tab.")
    st.info("Log in, authorize, then you'll be redirected back with ?code=..., the app will store the token, and the dashboard will appear.")
    st.stop()

#############################################################
# 5) Load the "clean_risks.csv" from Method 1
#############################################################
csv_file = "clean_risks.csv"  # The new pipeline file
try:
    df = pd.read_csv(csv_file)
    st.write(f"Loaded {len(df)} risks from {csv_file}.")
    df.fillna("", inplace=True)
except FileNotFoundError:
    st.error(f"{csv_file} not found. Please run Method 1 first.")
    st.stop()

# If we want to do embeddings coverage
embedder = None
index = None
if EMBEDDING_AVAILABLE:
    try:
        embed_np = np.load("embeddings.npy")
        index = faiss.read_index("faiss_index.faiss")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        st.write("Embeddings + FAISS index found - we'll do coverage with nearest neighbors.")
    except:
        st.warning("No embeddings or faiss index found. Fallback coverage approach might be naive substring.")
else:
    st.warning("sentence_transformers or faiss not installed; cannot do embedding-based coverage. Fallback only.")

#############################################################
# 6) Mural Sticky Pull
#############################################################
def pull_sticky_notes(mural_id, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}", "Accept":"application/json"}
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    try:
        s = requests.Session()
        rtry = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        s.mount("https://", HTTPAdapter(max_retries=rtry))
        resp = s.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            items = resp.json().get("value",[]) or resp.json().get("data",[])
            lines=[]
            for w in items:
                wtype = w.get("type","").replace(" ","_").lower()
                if wtype == "sticky_note":
                    raw = w.get("htmlText") or w.get("text") or ""
                    c = clean_html_text(raw)
                    if c: lines.append(c)
            return lines
        else:
            st.error(f"Failed pulling sticky notes: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Error pulling Mural: {str(e)}")
        return []

#############################################################
# 7) Main UI
#############################################################
st.sidebar.header("Mural Actions")
custom_mural_id = st.sidebar.text_input("Mural ID", value=MURAL_BOARD_ID)
if st.sidebar.button("List Murals"):
    with st.spinner("Listing murals..."):
        murals = list_murals(st.session_state["access_token"])
        if murals:
            st.write("Available Murals:", [ (m.get("id"), m.get("title")) for m in murals ])
        else:
            st.warning("No murals found or error.")

if st.sidebar.button("Pull Sticky Notes"):
    with st.spinner("Pulling notes..."):
        if not verify_mural(st.session_state["access_token"], custom_mural_id):
            # maybe remove workspace prefix
            custom_mural_id = normalize_mural_id(custom_mural_id)
            if not verify_mural(st.session_state["access_token"], custom_mural_id):
                st.error(f"Mural ID {custom_mural_id} not found or you lack permission.")
                st.stop()
        lines = pull_sticky_notes(custom_mural_id, st.session_state["access_token"])
        if lines:
            st.session_state["mural_notes"] = lines
            st.success(f"Pulled {len(lines)} sticky notes from {custom_mural_id}!")
        else:
            st.warning("No notes or error pulling data.")

st.subheader("1) Provide Final Lines (From Mural or Manual)")
default_text = "\n".join(st.session_state.get("mural_notes", []))
user_input = st.text_area("Your final lines (one per line):", default_text, height=180)

#############################################################
# 8) Coverage + Gap Analysis
#############################################################
st.subheader("2) Coverage & Gap Analysis")

def coverage_analysis(lines, df_risks):
    coverage_res = {
        "risk_type": set(),
        "node_name": set(),
        "cluster": set()
    }
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return coverage_res

    # If embedder + index => do nearest neighbor
    if embedder and index:
        import numpy as np
        user_embs = embedder.encode(lines)
        user_embs = np.array(user_embs, dtype="float32")
        k=3
        D,I = index.search(user_embs, k)
        for i in range(len(lines)):
            sub = df_risks.iloc[I[i]]
            coverage_res["risk_type"].update(sub["risk_type"].unique())
            coverage_res["node_name"].update(sub["node_name"].unique())
            coverage_res["cluster"].update(sub["cluster"].unique())
    else:
        # naive substring approach
        combined = " ".join(l.lower() for l in lines)
        for rt in df_risks["risk_type"].unique():
            if rt.lower() in combined:
                coverage_res["risk_type"].add(rt)
        for nm in df_risks["node_name"].unique():
            if nm.lower() in combined:
                coverage_res["node_name"].add(nm)
        for cl in df_risks["cluster"].unique():
            if str(cl).lower() in combined:
                coverage_res["cluster"].add(cl)

    return coverage_res

if st.button("Analyze Coverage"):
    lines = [l for l in user_input.split("\n") if l.strip()]
    if not lines:
        st.warning("No lines to analyze.")
    else:
        cov = coverage_analysis(lines, df)
        all_types = df["risk_type"].unique()
        all_nodes = df["node_name"].unique()
        all_clusters = df["cluster"].unique()

        missed_types = set(all_types) - cov["risk_type"]
        missed_nodes = set(all_nodes) - cov["node_name"]
        missed_clusters = set(all_clusters) - cov["cluster"]

        st.write("**Coverage Results**:")
        st.write(f"- Covered Risk Types: {sorted(list(cov['risk_type']))}, Missed: {sorted(list(missed_types))}")
        st.write(f"- Covered Stakeholders (node_name): {sorted(list(cov['node_name']))}, Missed: {sorted(list(missed_nodes))}")
        st.write(f"- Covered Clusters: {sorted(list(cov['cluster']))}, Missed: {sorted(list(missed_clusters))}")

        # Build bar charts for each category
        # We'll do a function for side-by-side coverage
        def side_by_side_coverage_chart(title, all_items, covered_set, filename):
            cat_list = sorted(set(all_items))
            covered_counts, missed_counts = [], []
            for c in cat_list:
                if c in covered_set:
                    covered_counts.append(1)
                    missed_counts.append(0)
                else:
                    covered_counts.append(0)
                    missed_counts.append(1)
            create_coverage_chart(title, cat_list, covered_counts, missed_counts, filename)

        # 1) risk_type
        side_by_side_coverage_chart("Risk Type Coverage", all_types, cov["risk_type"], "risk_type_cov.png")
        st.image("risk_type_cov.png", use_column_width=True)
        # 2) stakeholder
        side_by_side_coverage_chart("Stakeholder Coverage", all_nodes, cov["node_name"], "stakeholder_cov.png")
        st.image("stakeholder_cov.png", use_column_width=True)
        # 3) cluster
        side_by_side_coverage_chart("Cluster Coverage", all_clusters, cov["cluster"], "cluster_cov.png")
        st.image("cluster_cov.png", use_column_width=True)

        # LLM gap feedback
        if st.checkbox("Generate LLM Gap Feedback?"):
            if not LLM_AVAILABLE:
                st.warning("OpenAI is not available. Please check your environment.")
            else:
                gap_prompt = f"""
You are an AI coverage assistant. The user has these final lines:
{chr(10).join("- "+l for l in lines)}

We found coverage on:
- risk_types covered: {sorted(list(cov['risk_type']))}, missed: {sorted(list(missed_types))}
- stakeholders covered: {sorted(list(cov['node_name']))}, missed: {sorted(list(missed_nodes))}
- clusters covered: {sorted(list(cov['cluster']))}, missed: {sorted(list(missed_clusters))}

Explain why missing categories might be critical and how the user can address them.
"""
                try:
                    resp = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role":"system","content":"You are a helpful coverage analysis assistant."},
                            {"role":"user","content":gap_prompt}
                        ]
                    )
                    feedback = resp.choices[0].message.content
                    st.markdown("### Gap Feedback:")
                    st.write(feedback)
                except Exception as e:
                    st.error(f"OpenAI gap feedback error: {str(e)}")

#############################################################
# 9) Brainstorm / Mitigation (Optional)
#############################################################
st.subheader("3) Brainstorm or Mitigation Strategies (Optional)")

# You can replicate the "Brainstorm" approach from your snippet,
# or omit if not needed.

#############################################################
# 10) Reset
#############################################################
if st.button("🗑 Clear Session & Query Params"):
    st.session_state.clear()
    st.set_query_params()  # remove code param
    st.rerun()

st.info("Done. This single script merges your 'better' Mural logic with coverage gap analysis for 'clean_risks.csv'.")
