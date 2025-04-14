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
import matplotlib.pyplot as plt
from collections import Counter
import re
import time

# Temporarily disable torch.classes to avoid Streamlit watcher error
sys.modules['torch.classes'] = None

from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

###############################################################
# Config
###############################################################
st.set_page_config(page_title="Method 2 - Demo", layout="wide")
st.title("Method 2 - AI Risk Coverage & Brainstorming")

###############################################################
# Load secrets
###############################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "myworkspaceid")
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets in .streamlit/secrets.toml.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

###############################################################
# Utility
###############################################################
def normalize_mural_id(mural_id, workspace_id=MURAL_WORKSPACE_ID):
    prefix = f"{workspace_id}."
    if mural_id.startswith(prefix):
        return mural_id[len(prefix):]
    return mural_id

def clean_html_text(html_text):
    if not html_text:
        return ""
    try:
        # Use regex to strip HTML tags and normalize whitespace
        cleaned = re.sub(r'<[^>]+>', ' ', html_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        st.error(f"HTML cleanup error: {e}")
        return ""

def log_feedback(risk_description, user_feedback, disagreement_reason=""):
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_description": risk_description,
        "user_feedback": user_feedback,
        "disagreement_reason": disagreement_reason
    }
    df_log = pd.DataFrame([data])
    filename = "feedback_log.csv"
    if os.path.exists(filename):
        old_df = pd.read_csv(filename)
        df_log = pd.concat([old_df, df_log], ignore_index=True)
    df_log.to_csv(filename, index=False)

def color_cell(val):
    """Example RAG scale."""
    if val == 0:
        return "background-color:#e74c3c; color:white"
    elif 1 <= val <= 2:
        return "background-color:#f39c12; color:white"
    else:
        return "background-color:#2ecc71; color:white"

###############################################################
# Coverage Chart Example
###############################################################
def create_coverage_chart(title, categories, covered_counts, missed_counts, filename):
    """Generate bar chart for coverage."""
    try:
        plt.figure(figsize=(6,4))
        x = np.arange(len(categories))
        plt.bar(x - 0.2, covered_counts, 0.4, label='Covered', color='#2ecc71')
        plt.bar(x + 0.2, missed_counts, 0.4, label='Missed', color='#e74c3c')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        st.error(f"Error creating chart: {e}")

def create_coverage_charts(covered_stakeholders, missed_stakeholders,
                           covered_types, missed_types,
                           covered_subtypes, missed_subtypes,
                           top_n_subtypes=5):
    """Generate bar charts for coverage (stakeholder, type, subtypes)."""
    try:
        plt.style.use('ggplot')
    except:
        pass

    # 1) Stakeholder chart
    all_st = sorted(set(covered_stakeholders + missed_stakeholders))
    cov_ct = [covered_stakeholders.count(s) for s in all_st]
    mis_ct = [missed_stakeholders.count(s) for s in all_st]
    nz_index = [i for i,(c,m) in enumerate(zip(cov_ct, mis_ct)) if c>0 or m>0]
    all_st = [all_st[i] for i in nz_index]
    cov_ct = [cov_ct[i] for i in nz_index]
    mis_ct = [mis_ct[i] for i in nz_index]
    if all_st:
        create_coverage_chart("Stakeholder Coverage Gaps", all_st, cov_ct, mis_ct, 'stakeholder_coverage.png')

    # 2) Risk Type chart
    all_ty = sorted(set(covered_types + missed_types))
    cov_ty = [covered_types.count(t) for t in all_ty]
    mis_ty = [missed_types.count(t) for t in all_ty]
    nz_index = [i for i,(c,m) in enumerate(zip(cov_ty, mis_ty)) if c>0 or m>0]
    all_ty = [all_ty[i] for i in nz_index]
    cov_ty = [cov_ty[i] for i in nz_index]
    mis_ty = [mis_ty[i] for i in nz_index]
    if all_ty:
        create_coverage_chart("Risk Type Coverage Gaps", all_ty, cov_ty, mis_ty, 'risk_type_coverage.png')

    # 3) Risk Subtype chart
    st_counter = Counter(missed_subtypes)
    top_missed = [k for k,_ in st_counter.most_common(top_n_subtypes)]
    cov_sub = [covered_subtypes.count(s) for s in top_missed]
    mis_sub = [missed_subtypes.count(s) for s in top_missed]
    if top_missed:
        create_coverage_chart(f"Top {top_n_subtypes} Overlooked Subtypes", top_missed, cov_sub, mis_sub, 'risk_subtype_coverage.png')

###############################################################
# Mural OAuth
###############################################################
def get_mural_auth_url():
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    return f"https://app.mural.co/api/public/v1/authorization/oauth2/?{urlencode(params)}"

def exchange_code_for_token(auth_code):
    with st.spinner("Authenticating with Mural..."):
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
            st.error(f"Mural auth failed. Status: {resp.status_code}")
            return None
        except Exception as e:
            st.error(f"Mural auth error: {e}")
            return None

def refresh_mural_token(refresh_token):
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
                st.error(f"Token refresh error: {resp.status_code}")
                return None
        except Exception as e:
            st.error(f"Refresh error: {e}")
            return None

def list_murals(token):
    url = "https://app.mural.co/api/public/v1/murals"
    headers = {"Authorization": f"Bearer {token}", "Accept":"application/json"}
    try:
        s = requests.Session()
        retr = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        s.mount("https://", HTTPAdapter(max_retries=retr))
        r = s.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json().get("value", [])
        st.error(f"List murals failed: {r.status_code}")
        return []
    except Exception as e:
        st.error(f"list_murals error: {e}")
        return []

def verify_mural(token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        s = requests.Session()
        retr = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        s.mount("https://", HTTPAdapter(max_retries=retr))
        r = s.get(url, headers=headers, timeout=10)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Error verifying mural: {str(e)}")
        return False

###############################################################
# Session State for Mural
###############################################################
if "mural_access_token" not in st.session_state:
    st.session_state["mural_access_token"] = None
    st.session_state["mural_refresh_token"] = None
    st.session_state["mural_token_expires"] = None
    st.session_state["mural_token_ts"] = None
    st.session_state["mural_notes"] = []

# Get query parameters
qs_params = st.query_params
auth_code = qs_params.get("code", None)

if auth_code and not st.session_state["mural_access_token"]:
    # Exchange code for token
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state["mural_access_token"] = token_data["access_token"]
        st.session_state["mural_refresh_token"] = token_data.get("refresh_token")
        st.session_state["mural_token_expires"] = token_data.get("expires_in", 900)
        st.session_state["mural_token Ministries = time.time()
        
        # Clear the query param "code"
        st.query_params.clear()
        
        st.success("Authenticated with Mural in the same tab!")
        st.rerun()

# If no access token yet, provide user with link
if not st.session_state["mural_access_token"]:
    mural_auth_url = get_mural_auth_url()
    st.markdown(f"[Click here to authorize Mural access]({mural_auth_url})")
    st.stop()

# If we have a token, check expiry
if st.session_state["mural_access_token"]:
    now_ts = time.time()
    if (now_ts - st.session_state["mural_token_ts"]) > (st.session_state["mural_token_expires"] - 60):
        new_data = refresh_mural_token(st.session_state["mural_refresh_token"])
        if new_data:
            st.session_state["mural_access_token"] = new_data["access_token"]
            st.session_state["mural_refresh_token"] = new_data.get("refresh_token", st.session_state["mural_refresh_token"])
            st.session_state["mural_token_expires"] = new_data.get("expires_in", 900)
            st.session_state["mural_token_ts"] = time.time()
            st.info("Refreshed Mural token")

###############################################################
# Load CSV
###############################################################
csv_file = "clean_risks.csv"
embeds_file = "embeddings.npy"
index_file = "faiss_index.faiss"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"Missing CSV: {csv_file}")
    st.stop()

try:
    embeddings = np.load(embeds_file)
except FileNotFoundError:
    st.error(f"Missing embeddings: {embeds_file}")
    st.stop()

try:
    index = faiss.read_index(index_file)
except FileNotFoundError:
    st.error(f"Missing FAISS index: {index_file}")
    st.stop()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

###############################################################
# Sidebar
###############################################################
with st.sidebar:
    st.header("Mural Tools")
    custom_mural_id = st.text_input("Optional Mural ID", value=MURAL_BOARD_ID)
    if st.button("List Murals"):
        ms = list_murals(st.session_state["mural_access_token"])
        st.write("Available murals:", ms if ms else "No murals found.")
    if st.button("Pull Stickies"):
        try:
            token = st.session_state["mural_access_token"]
            real_id = custom_mural_id.strip()
            st.write(f"Checking mural ID: {real_id}")
            if not verify_mural(token, real_id):
                real_id = normalize_mural_id(real_id)
                st.write(f"Normalized mural ID: {real_id}")
                if not verify_mural(token, real_id):
                    st.error("Cannot find mural.")
                    st.stop()
            w_url = f"https://app.mural.co/api/public/v1/murals/{real_id}/widgets"
            h = {"Authorization": f"Bearer {token}"}
            s = requests.Session()
            retr = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
            s.mount("https://", HTTPAdapter(max_retries=retr))
            r = s.get(w_url, headers=h, timeout=10)
            st.write(f"API status code: {r.status_code}")
            if r.status_code == 200:
                raw_response = r.json()
                widgets = raw_response.get("value", [])
                st.write("Full API response:", raw_response)
                st.write("Widgets:", widgets)
                sticky_items = []
                for w_ in widgets:
                    widget_type = w_.get("type", "").lower()
                    st.write(f"Widget: type={widget_type}, id={w_.get('id')}")
                    if widget_type == "sticky note":  # Match API's exact casing
                        raw = w_.get("htmlText") or w_.get("text") or w_.get("content", "")
                        cleaned = clean_html_text(raw)
                        st.write(f"Sticky: raw={raw}, cleaned={cleaned}")
                        if cleaned:
                            sticky_items.append(cleaned)
                        else:
                            st.write("Warning: Empty cleaned text for sticky note")
                st.session_state["mural_notes"] = sticky_items
                st.success(f"Pulled {len(sticky_items)} sticky notes!")
            else:
                st.error(f"Pull stickies failed: {r.status_code}")
        except Exception as e:
            st.error(f"Error pulling stickies: {e}")
    if st.button("Clear Session"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

###############################################################
# Main
###############################################################
st.subheader("1) Enter/Load Finalized Risks")
default_text = ""
if "mural_notes" in st.session_state:
    default_text = "\n".join(st.session_state["mural_notes"])
user_input = st.text_area("Risks from Mural or manual", value=default_text, height=200)

st.subheader("2) Coverage Analysis")
if st.button("Analyze Coverage"):
    with st.spinner("Analyzing coverage..."):
        lines = [ln.strip() for ln in user_input.split("\n") if ln.strip()]
        if not lines:
            st.warning("No input lines found.")
        else:
            # Encode & search
            vecs = embedder.encode(lines)
            vecs = np.array(vecs, dtype="float32")
            distances, idx = index.search(vecs, 3)  # top-3 neighbors
            all_covered_types = set()
            all_covered_subtypes = set()
            all_covered_stakeholders = set()

            for row_i in idx.flatten():
                row_info = df.iloc[row_i].to_dict()
                if "risk_type" in row_info and pd.notna(row_info["risk_type"]):
                    all_covered_types.add(row_info["risk_type"])
                if "risk_subtype" in row_info and pd.notna(row_info["risk_subtype"]):
                    all_covered_subtypes.add(row_info["risk_subtype"])
                if "stakeholder" in row_info and pd.notna(row_info["stakeholder"]):
                    all_covered_stakeholders.add(row_info["stakeholder"])

            all_types = df["risk_type"].dropna().unique().tolist()
            all_subs = df["risk_subtype"].dropna().unique().tolist() if "risk_subtype" in df.columns else []
            all_stkh = df["stakeholder"].dropna().unique().tolist() if "stakeholder" in df.columns else []

            missed_types = sorted(list(set(all_types) - all_covered_types))
            missed_subs = sorted(list(set(all_subs) - all_covered_subtypes))
            missed_stkh = sorted(list(set(all_stkh) - all_covered_stakeholders))

            # Display coverage
            st.write("**Coverage**")
            st.write("Covered risk types:", list(all_covered_types))
            st.write("Missed risk types:", missed_types)
            st.write("Covered subtypes:", list(all_covered_subtypes))
            st.write("Missed subtypes:", missed_subs)
            st.write("Covered stakeholders:", list(all_covered_stakeholders))
            st.write("Missed stakeholders:", missed_stkh)

            create_coverage_charts(
                list(all_covered_stakeholders), missed_stkh,
                list(all_covered_types), missed_types,
                list(all_covered_subtypes), missed_subs,
                top_n_subtypes=5
            )

            # Display charts
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    st.image("stakeholder_coverage.png", caption="Stakeholder Gaps", use_column_width=True)
                except FileNotFoundError:
                    pass
            with col2:
                try:
                    st.image("risk_type_coverage.png", caption="Risk Type Gaps", use_column_width=True)
                except FileNotFoundError:
                    pass
            with col3:
                try:
                    st.image("risk_subtype_coverage.png", caption="Subtype Gaps", use_column_width=True)
                except FileNotFoundError:
                    pass

            # Display neighbor distances
            st.write("### Example neighbor distances")
            for i, line_i in enumerate(lines):
                dists = distances[i]
                idxs = idx[i]
                st.write(f"**Input {i+1}**: {line_i}")
                for rank, (dist_v, idx_v) in enumerate(zip(dists, idxs), start=1):
                    row_ = df.iloc[idx_v]
                    st.write(f"{rank}) {row_['risk_description']} (distance={dist_v:.3f})")
                st.write("---")
