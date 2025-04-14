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
import openai  # If you prefer: from openai import OpenAI

################################################################
# Streamlit app config
################################################################
st.set_page_config(page_title="Method 2 Example", layout="wide")
st.title("Method 2 - GPT-Powered Coverage Dashboard (No experimental_ calls)")

################################################################
# Load secrets
################################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "myworkspaceid")
except KeyError as e:
    st.error(f"Missing secret: {e}. Please add to .streamlit/secrets.toml.")
    st.stop()

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

################################################################
# Utility
################################################################
def normalize_mural_id(mural_id, workspace_id=MURAL_WORKSPACE_ID):
    prefix = f"{workspace_id}."
    if mural_id.startswith(prefix):
        return mural_id[len(prefix):]
    return mural_id

def clean_html_text(html_text):
    if not html_text:
        return ""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text(separator=" ").strip()
    except Exception as e:
        st.error(f"Error cleaning HTML: {str(e)}")
        return ""

def log_feedback(risk_description, user_feedback, disagreement_reason=""):
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_description": risk_description,
        "user_feedback": user_feedback,
        "disagreement_reason": disagreement_reason
    }
    feedback_df = pd.DataFrame([data])
    file_ = "feedback_log.csv"
    if os.path.exists(file_):
        old = pd.read_csv(file_)
        feedback_df = pd.concat([old, feedback_df], ignore_index=True)
    feedback_df.to_csv(file_, index=False)

################################################################
# Coverage chart example
################################################################
def create_coverage_chart(title, categories, covered_counts, missed_counts, filename):
    try:
        plt.figure(figsize=(6, 4))
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
        st.error(f"Chart error for {filename}: {str(e)}")

def create_coverage_charts(
    covered_stakeholders, missed_stakeholders,
    covered_types, missed_types,
    covered_subtypes, missed_subtypes,
    top_n_subtypes=5
):
    try:
        plt.style.use('ggplot')
    except:
        pass

    # Stakeholder chart
    all_stake = sorted(set(covered_stakeholders + missed_stakeholders))
    cov_counts = [covered_stakeholders.count(s) for s in all_stake]
    mis_counts = [missed_stakeholders.count(s) for s in all_stake]
    nz_idx = [i for i,(c,m) in enumerate(zip(cov_counts, mis_counts)) if c>0 or m>0]
    all_stake = [all_stake[i] for i in nz_idx]
    cov_counts = [cov_counts[i] for i in nz_idx]
    mis_counts = [mis_counts[i] for i in nz_idx]
    if all_stake:
        create_coverage_chart("Stakeholder Coverage Gaps", all_stake, cov_counts, mis_counts, 'stakeholder_coverage.png')

    # Risk types
    all_types = sorted(set(covered_types + missed_types))
    c_ty = [covered_types.count(t) for t in all_types]
    m_ty = [missed_types.count(t) for t in all_types]
    nz_idx = [i for i,(cc,mm) in enumerate(zip(c_ty, m_ty)) if cc>0 or mm>0]
    all_types = [all_types[i] for i in nz_idx]
    c_ty = [c_ty[i] for i in nz_idx]
    m_ty = [m_ty[i] for i in nz_idx]
    if all_types:
        create_coverage_chart("Risk Type Coverage Gaps", all_types, c_ty, m_ty, 'risk_type_coverage.png')

    # Subtypes
    sub_counter = Counter(missed_subtypes)
    top_missed = [k for k,_ in sub_counter.most_common(top_n_subtypes)]
    c_sub = [covered_subtypes.count(s) for s in top_missed]
    m_sub = [missed_subtypes.count(s) for s in top_missed]
    if top_missed:
        create_coverage_chart(f"Top {top_n_subtypes} Overlooked Subtypes", top_missed, c_sub, m_sub, 'risk_subtype_coverage.png')


################################################################
# Mural OAuth
################################################################
def get_mural_auth_url():
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    return f"https://app.mural.co/api/public/v1/authorization/oauth2/?{urlencode(params)}"

def exchange_code_for_token(code_):
    with st.spinner("Authenticating with Mural..."):
        url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
        data = {
            "client_id": MURAL_CLIENT_ID,
            "client_secret": MURAL_CLIENT_SECRET,
            "redirect_uri": MURAL_REDIRECT_URI,
            "code": code_,
            "grant_type": "authorization_code"
        }
        try:
            rr = requests.post(url, data=data, timeout=10)
            if rr.status_code == 200:
                return rr.json()
            else:
                st.error(f"Mural auth failed: {rr.status_code}")
                return None
        except Exception as e:
            st.error(f"Auth error: {e}")
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
            r_ = requests.post(url, data=data, timeout=10)
            if r_.status_code == 200:
                return r_.json()
            st.error(f"Token refresh error: {r_.status_code}")
            return None
        except Exception as e:
            st.error(f"Refresh token error: {e}")
            return None

def list_murals(token):
    url = "https://app.mural.co/api/public/v1/murals"
    h = {"Authorization": f"Bearer {token}"}
    try:
        s_ = requests.Session()
        re_ = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        s_.mount("https://", HTTPAdapter(max_retries=re_))
        r_ = s_.get(url, headers=h, timeout=10)
        if r_.status_code == 200:
            return r_.json().get("value", [])
        else:
            st.error(f"List murals: {r_.status_code}")
            return []
    except Exception as e:
        st.error(f"List murals error: {e}")
        return []

def verify_mural(token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    h = {"Authorization": f"Bearer {token}"}
    try:
        s_ = requests.Session()
        re_ = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        s_.mount("https://", HTTPAdapter(max_retries=re_))
        r_ = s_.get(url, headers=h, timeout=10)
        return (r_.status_code == 200)
    except:
        return False

################################################################
# Track Mural session
################################################################
if "mural_access_token" not in st.session_state:
    st.session_state["mural_access_token"] = None
    st.session_state["mural_refresh_token"] = None
    st.session_state["mural_token_expires"] = None
    st.session_state["mural_token_ts"] = None

qs = st.query_params
auth_code = qs.get("code", None)

if auth_code and not st.session_state["mural_access_token"]:
    # We have an auth code in URL
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state["mural_access_token"] = token_data["access_token"]
        st.session_state["mural_refresh_token"] = token_data.get("refresh_token")
        st.session_state["mural_token_expires"] = token_data.get("expires_in", 900)
        st.session_state["mural_token_ts"] = time.time()
        
        # Clear the query param "code"
        st.set_query_params()  # no args => clears all
        st.success("Authenticated with Mural in the same tab!")
        st.rerun()  # replaces st.experimental_rerun()

if not st.session_state["mural_access_token"]:
    # no token? show link
    link_ = get_mural_auth_url()
    st.markdown(f"Please [authorize the app in Mural]({link_})")
    st.stop()

# If we do have token, check expiry
if st.session_state["mural_access_token"]:
    now_ = time.time()
    if (now_ - st.session_state["mural_token_ts"]) > (st.session_state["mural_token_expires"] - 60):
        # Refresh
        new_ = refresh_mural_token(st.session_state["mural_refresh_token"])
        if new_:
            st.session_state["mural_access_token"] = new_["access_token"]
            st.session_state["mural_refresh_token"] = new_.get("refresh_token", st.session_state["mural_refresh_token"])
            st.session_state["mural_token_expires"] = new_.get("expires_in", 900)
            st.session_state["mural_token_ts"] = time.time()
            st.info("Refreshed Mural token")

################################################################
# Load your CSV + FAISS
################################################################
csv_file = "AI-Powered_Valuation_Clustered.csv"
emb_file = "embeddings.npy"
faiss_file = "faiss_index.faiss"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    st.error(f"Missing {csv_file}")
    st.stop()

try:
    embeddings = np.load(emb_file)
except FileNotFoundError:
    st.error(f"Missing {emb_file}")
    st.stop()

try:
    index = faiss.read_index(faiss_file)
except FileNotFoundError:
    st.error(f"Missing {faiss_file}")
    st.stop()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

################################################################
# Sidebar
################################################################
with st.sidebar:
    st.header("Mural Tools")
    my_mural_id = st.text_input("Optional Mural ID", value=MURAL_BOARD_ID)
    
    if st.button("List Murals"):
        mm = list_murals(st.session_state["mural_access_token"])
        st.write(mm if mm else "No murals found.")
    
    if st.button("Pull Sticky Notes"):
        try:
            token = st.session_state["mural_access_token"]
            real_id = my_mural_id.strip()
            if not verify_mural(token, real_id):
                real_id = normalize_mural_id(real_id)
                if not verify_mural(token, real_id):
                    st.error(f"Mural {real_id} not found.")
                    st.stop()
            wurl = f"https://app.mural.co/api/public/v1/murals/{real_id}/widgets"
            h_ = {"Authorization": f"Bearer {token}"}
            s_ = requests.Session()
            re_ = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
            s_.mount("https://", HTTPAdapter(max_retries=re_))
            r_ = s_.get(wurl, headers=h_, timeout=10)
            if r_.status_code == 200:
                widgets = r_.json().get("value", [])
                sticky_notes = []
                for w in widgets:
                    t_ = w.get("type","").lower()
                    if t_ == "sticky_note":
                        raw = w.get("htmlText") or w.get("text","")
                        cleaned = clean_html_text(raw)
                        if cleaned:
                            sticky_notes.append(cleaned)
                st.session_state["mural_notes"] = sticky_notes
                st.success(f"Pulled {len(sticky_notes)} sticky notes!")
            else:
                st.error(f"Pull error: {r_.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.button("Clear Session"):
        st.session_state.clear()
        st.set_query_params()  # clear URL
        st.rerun()

################################################################
# Main UI
################################################################

st.subheader("1) Input Your Finalized Risks")
default_text = ""
if "mural_notes" in st.session_state:
    default_text = "\n".join(st.session_state["mural_notes"])
user_risks = st.text_area("Paste/Load from Mural", value=default_text, height=200)

st.subheader("2) Generate Coverage Feedback + GPT Analysis")

top_n_subtypes = st.slider("Top N Overlooked Subtypes to Display", 3, 10, 5)

if st.button("Analyze Coverage"):
    with st.spinner("Analyzing coverage using GPT..."):
        lines = [r.strip() for r in user_risks.split("\n") if r.strip()]
        if not lines:
            st.warning("No risks found.")
            st.stop()
        
        # 1) Get embeddings
        user_vecs = embedder.encode(lines)
        user_vecs = np.array(user_vecs, dtype="float32")
        
        # 2) Search the FAISS index
        # e.g., top-5 neighbors for each risk
        k = 5
        distances, indices = index.search(user_vecs, k)
        
        covered_types = set()
        covered_subtypes = set()
        covered_stakeholders = set()

        # gather coverage
        for row_idx_list in indices:
            for row_idx in row_idx_list:
                row_ = df.iloc[row_idx]
                if "risk_type" in row_ and pd.notna(row_["risk_type"]):
                    covered_types.add(row_["risk_type"])
                if "risk_subtype" in row_ and pd.notna(row_["risk_subtype"]):
                    covered_subtypes.add(row_["risk_subtype"])
                if "stakeholder" in row_ and pd.notna(row_["stakeholder"]):
                    covered_stakeholders.add(row_["stakeholder"])
        
        all_types = df["risk_type"].dropna().unique().tolist()
        all_subs = df["risk_subtype"].dropna().unique().tolist() if "risk_subtype" in df.columns else []
        all_stk = df["stakeholder"].dropna().unique().tolist() if "stakeholder" in df.columns else []
        
        missed_types = sorted(list(set(all_types) - covered_types))
        missed_subs = sorted(list(set(all_subs) - covered_subtypes))
        missed_stk = sorted(list(set(all_stk) - covered_stakeholders))

        # 3) Create charts
        create_coverage_charts(
            list(covered_stakeholders), missed_stk,
            list(covered_types), missed_types,
            list(covered_subtypes), missed_subs,
            top_n_subtypes=top_n_subtypes
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.image("stakeholder_coverage.png", caption="Stakeholder Gaps", use_column_width=True)
            except:
                pass
        with col2:
            try:
                st.image("risk_type_coverage.png", caption="Risk Type Gaps", use_column_width=True)
            except:
                pass
        with col3:
            try:
                st.image("risk_subtype_coverage.png", caption="Top Overlooked Subtypes", use_column_width=True)
            except:
                pass

        # 4) Ask GPT to produce textual coverage feedback
        domain = df["domain"].iloc[0] if "domain" in df.columns else "AI solution"
        # Use some snippet from the CSV as examples
        # For instance, show examples from missed_types
        example_chunks = []
        for cat_label, cat_values in [
            ("Missed Risk Types", missed_types),
            ("Missed Risk Subtypes", missed_subs),
            ("Missed Stakeholders", missed_stk)
        ]:
            for val in cat_values[:2]:  # up to 2 examples
                if cat_label.endswith("Types"):
                    ex_df = df[df["risk_type"]==val].head(1)
                elif cat_label.endswith("Subtypes"):
                    ex_df = df[df["risk_subtype"]==val].head(1)
                else:
                    ex_df = df[df["stakeholder"]==val].head(1)
                if not ex_df.empty:
                    ex_ = ex_df.iloc[0]
                    example_chunks.append(
                        f"{cat_label} => {val}: Example => {ex_['risk_description']}"
                    )
        example_str = "\n".join(example_chunks)

        # GPT Prompt
        prompt = f"""
You are an AI risk analysis expert in the domain of {domain}.
The user has provided these finalized risks:
{chr(10).join("- "+r for r in lines)}

Below are some categories that appear missed or underrepresented:
{example_str}

Please provide coverage feedback focusing on:
1) Missing or overlooked risk types, subtypes, or stakeholders
2) Why these are important to include in a thorough risk analysis
3) Suggestions to expand or refine the user's Mural items
"""

        try:
            gpt_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":"You are a helpful AI risk advisor."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.3
            )
            coverage_feedback = gpt_response.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            coverage_feedback = None
        
        if coverage_feedback:
            st.markdown("### GPT Coverage Feedback")
            st.write(coverage_feedback)

        # Example: Show neighbor distances (fix incomplete f-string!)
        st.markdown("---")
        st.markdown("### Nearest Neighbors (Debug Info)")
        for i, risk_line in enumerate(lines):
            st.write(f"**User Risk {i+1}:** {risk_line}")
            row_dists = distances[i]
            row_idx = indices[i]
            for rank, (dist_val, idx_val) in enumerate(zip(row_dists, row_idx), start=1):
                row_ = df.iloc[idx_val]
                # Properly close the f-string:
                st.write(f"{rank}) {row_['risk_description']} (distance={dist_val:.3f})")
            st.write("---")

st.subheader("3) Next Steps")
st.write("You can incorporate GPT feedback by adding new sticky notes in Mural, then re-pulling them here.")

