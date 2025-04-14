###############################################################################
# method2.py - Author Gareth McConomy 
#
# STREAMLIT APP (Method 2) for:
#   - Pulling data from Mural (if desired),
#   - Loading your final "clean_risks.csv" from Method 1,
#   - Checking coverage/gaps in stakeholder or risk types,
#   - Generating coverage feedback & charts,
#   - Optionally brainstorming additional risks.
#
# NOTE: 
#   - This script expects you have "clean_risks.csv" with columns like:
#       [risk_id, risk_description, stakeholder, risk_type, severity, probability, combined_score, cluster, ...]
#     or you can tweak the column references if your CSV is named/structured differently.
#   - The Mural integration uses secrets in Streamlit, or environment variables for OAuth.
#   - If you don't need Mural pulling, you can skip or remove those sections.
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
import matplotlib.pyplot as plt
from collections import Counter
import re

# Temporarily disable torch.classes to avoid Streamlit watcher error
# (Only needed if you get "torch.classes" import errors in Streamlit)
sys.modules['torch.classes'] = None

from sentence_transformers import SentenceTransformer
import faiss
# If you used openai: 
import openai

###############################################################################
# Streamlit Page Config
###############################################################################
st.set_page_config(page_title="AI Risk Coverage & Analysis (Method 2)", layout="wide")
st.title("🤖 AI Risk Coverage & Brainstorming Dashboard")

###############################################################################
# Load your secrets or environment variables
###############################################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "myworkspace")
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets in .streamlit/secrets.toml.")
    st.stop()

openai.api_key = OPENAI_API_KEY

###############################################################################
# Utility Functions
###############################################################################
def clean_html_text(html_text):
    """Strip HTML tags and return plain text."""
    if not html_text:
        return ""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text(separator=" ").strip()
    except Exception as e:
        st.error(f"HTML cleaning error: {str(e)}")
        return ""

def log_feedback(risk_description, user_feedback, disagreement_reason=""):
    """Log user feedback to CSV (or any store)."""
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_description": risk_description,
        "user_feedback": user_feedback,
        "disagreement_reason": disagreement_reason
    }
    df_feedback = pd.DataFrame([data])
    feedback_file = "feedback_log.csv"
    if os.path.exists(feedback_file):
        existing = pd.read_csv(feedback_file)
        df_feedback = pd.concat([existing, df_feedback], ignore_index=True)
    df_feedback.to_csv(feedback_file, index=False)

def create_coverage_chart(title, categories, covered_counts, missed_counts, filename):
    """Create a single bar chart for coverage (stakeholders, risk types, etc.)."""
    try:
        plt.figure(figsize=(6, 4))
        x = np.arange(len(categories))
        plt.bar(x - 0.2, covered_counts, 0.4, label='Covered', color='#2ecc71')
        plt.bar(x + 0.2, missed_counts, 0.4, label='Missed', color='#e74c3c')
        plt.xlabel(title.split(' ')[-1])
        plt.ylabel('Number of Risks')
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
    """Generate bar charts for coverage visualization."""
    # Basic approach: show coverage vs. missed for stakeholder, risk type, plus subtypes
    try:
        plt.style.use('ggplot')
    except Exception:
        plt.style.use('default')

    # 1) Stakeholders
    stakeholders = sorted(set(covered_stakeholders + missed_stakeholders))
    covered_counts = [covered_stakeholders.count(s) for s in stakeholders]
    missed_counts = [missed_stakeholders.count(s) for s in stakeholders]
    non_zero_indices = [i for i,(c,m) in enumerate(zip(covered_counts, missed_counts)) if c>0 or m>0]
    stakeholders = [stakeholders[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    if stakeholders:
        create_coverage_chart("Stakeholder Coverage Gaps", stakeholders, covered_counts, missed_counts, 'stakeholder_coverage.png')

    # 2) Risk Types
    risk_types = sorted(set(covered_types + missed_types))
    covered_counts = [covered_types.count(t) for t in risk_types]
    missed_counts = [missed_types.count(t) for t in risk_types]
    non_zero_indices = [i for i,(c,m) in enumerate(zip(covered_counts, missed_counts)) if c>0 or m>0]
    risk_types = [risk_types[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    if risk_types:
        create_coverage_chart("Risk Type Coverage Gaps", risk_types, covered_counts, missed_counts, 'risk_type_coverage.png')

    # 3) Risk Subtypes (example only if you have subtypes)
    # We'll just show the top-n missed subtypes
    subtype_counts = Counter(missed_subtypes)
    top_missed_subtypes = [s for s,_ in subtype_counts.most_common(top_n_subtypes)]
    covered_counts = [covered_subtypes.count(s) for s in top_missed_subtypes]
    missed_counts = [missed_subtypes.count(s) for s in top_missed_subtypes]
    if top_missed_subtypes:
        create_coverage_chart(f"Top {top_n_subtypes} Overlooked Subtypes", top_missed_subtypes,
                              covered_counts, missed_counts, 'risk_subtype_coverage.png')

###############################################################################
# MURAL OAuth Functions (optional)
###############################################################################
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
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "redirect_uri": MURAL_REDIRECT_URI,
        "code": code,
        "grant_type": "authorization_code"
    }
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Mural Auth failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Mural Auth error: {str(e)}")
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
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Mural refresh failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Mural refresh error: {str(e)}")
        return None

def list_murals(auth_token):
    url = "https://app.mural.co/api/public/v1/murals"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    response = session.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        return response.json().get("value", [])
    else:
        st.error(f"Failed to list murals: {response.status_code}")
        return []

def verify_mural(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    response = session.get(url, headers=headers, timeout=10)
    return (response.status_code == 200)

###############################################################################
# Initialize session state for Mural OAuth
###############################################################################
if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.token_expires_in = None
    st.session_state.token_timestamp = None

query_params = st.query_params
auth_code = query_params.get("code")
if auth_code and not st.session_state.access_token:
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state.access_token = token_data["access_token"]
        st.session_state.refresh_token = token_data.get("refresh_token")
        st.session_state.token_expires_in = token_data.get("expires_in", 900)
        st.session_state.token_timestamp = pd.Timestamp.now().timestamp()
        st.query_params.clear()
        st.success("Authenticated with Mural!")
        st.experimental_rerun()

if not st.session_state.access_token:
    auth_url = get_authorization_url()
    st.markdown(f"Please [authorize this app with Mural]({auth_url}).")
    st.stop()
else:
    current_time = pd.Timestamp.now().timestamp()
    if (current_time - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        token_data = refresh_access_token(st.session_state.refresh_token)
        if token_data:
            st.session_state.access_token = token_data["access_token"]
            st.session_state.refresh_token = token_data.get("refresh_token", st.session_state.refresh_token)
            st.session_state.token_expires_in = token_data.get("expires_in", 900)
            st.session_state.token_timestamp = pd.Timestamp.now().timestamp()

###############################################################################
# Sidebar Mural Options
###############################################################################
st.sidebar.subheader("Mural Actions")
mural_id_input = st.sidebar.text_input("Mural ID (defaults to your board)", value=MURAL_BOARD_ID)

if st.sidebar.button("List Murals"):
    with st.spinner("Listing murals..."):
        murals = list_murals(st.session_state.access_token)
        if murals:
            st.write("Available Murals:", murals)
        else:
            st.info("No murals found or request failed.")

if st.sidebar.button("Pull Sticky Notes"):
    with st.spinner("Pulling sticky notes from Mural..."):
        try:
            headers = {
                "Authorization": f"Bearer {st.session_state.access_token}",
                "Accept": "application/json"
            }
            if not verify_mural(st.session_state.access_token, mural_id_input):
                st.warning("Mural ID invalid or not accessible. Trying normalized ID...")
                # normalize if needed
            url = f"https://app.mural.co/api/public/v1/murals/{mural_id_input}/widgets"
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            mural_data = session.get(url, headers=headers, timeout=10)
            if mural_data.status_code == 200:
                widgets = mural_data.json().get("value", mural_data.json().get("data", []))
                sticky_widgets = [w for w in widgets if w.get('type','').replace(' ','_').lower() == 'sticky_note']
                notes = []
                for w in sticky_widgets:
                    raw_text = w.get('htmlText') or w.get('text') or ''
                    cleaned = clean_html_text(raw_text)
                    if cleaned:
                        notes.append(cleaned)
                st.session_state['mural_notes'] = notes
                st.success(f"Pulled {len(notes)} sticky notes from Mural.")
            else:
                st.error(f"Mural pull error: {mural_data.status_code}")
        except Exception as e:
            st.error(f"Mural connection error: {str(e)}")

if st.sidebar.button("Logout Mural"):
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.experimental_rerun()

###############################################################################
# MAIN Coverage & Analysis
###############################################################################
st.subheader("1️⃣ Load or Paste Human-Finalized Risks")
notes_default = "\n".join(st.session_state.get('mural_notes', []))
user_input = st.text_area("Paste your finalized risks from Mural or local input:", value=notes_default, height=200)

st.subheader("2️⃣ Load the Output from Method 1 (clean_risks.csv)")
csv_file = st.text_input("CSV File from Method 1", value="clean_risks.csv")

if st.button("Load CSV & Analyze Coverage"):
    try:
        df = pd.read_csv(csv_file)
        if 'risk_type' not in df.columns:
            st.error("Expected 'risk_type' column missing. Check your CSV from Method 1.")
            st.stop()
        st.success(f"Loaded {df.shape[0]} risk lines from {csv_file}.")

        # Simple coverage approach: if user_input is non-empty, embed or do coverage checks
        if user_input.strip():
            # Suppose we have N lines from the user
            user_lines = [l.strip() for l in user_input.split('\n') if l.strip()]

            # For coverage: we can do a naive approach: check if user lines match or are similar to the cluster lines
            # or we do some advanced approach. For demonstration, let's just do a conceptual coverage check:
            # We'll embed the user lines, do a nearest neighbor search on df's embeddings if we want (df had "embeddings" originally).
            # But let's do something simpler, like we ask GPT to generate coverage feedback for "missed" risk types/stakeholders.

            all_risk_types = df['risk_type'].unique().tolist()
            # Suppose we define "stakeholder" column or "node_name" for coverage
            # We'll guess you have 'stakeholder' or 'node_name' in df:
            possible_stakeholders = []
            if 'stakeholder' in df.columns:
                possible_stakeholders = df['stakeholder'].dropna().unique().tolist()
            else:
                possible_stakeholders = ["UnknownStakeholder"]

            # This is a simplistic approach: we see which risk types or stakeholders appear in user lines
            # versus which appear in the overall CSV. Then we do coverage charts.
            # For demonstration, let's assume the user lines are simpler or we won't parse them thoroughly.
            covered_types = []
            missed_types = []
            covered_stakeholders = []
            missed_stakeholders = []

            # We'll do a naive approach: if a user line mentions "Financial" or "Ethical" etc. we consider it "covered".
            # More robust approach is to embed & nearest-neighbor, but let's keep it simple:
            for rt in all_risk_types:
                # if user lines mention 'rt' as substring? (a hack, but just for demonstration)
                mention = any(rt.lower() in l.lower() for l in user_lines)
                if mention:
                    covered_types.append(rt)
                else:
                    missed_types.append(rt)

            for stkh in possible_stakeholders:
                mention = any(stkh.lower() in l.lower() for l in user_lines)
                if mention:
                    covered_stakeholders.append(stkh)
                else:
                    missed_stakeholders.append(stkh)

            # For subtypes, if you have 'risk_subtype' column:
            if 'risk_subtype' in df.columns:
                all_subtypes = df['risk_subtype'].dropna().unique().tolist()
                covered_subtypes = []
                missed_subtypes = []
                for sb in all_subtypes:
                    mention = any(sb.lower() in l.lower() for l in user_lines)
                    if mention:
                        covered_subtypes.append(sb)
                    else:
                        missed_subtypes.append(sb)
            else:
                covered_subtypes = []
                missed_subtypes = []

            # Create coverage charts:
            create_coverage_charts(covered_stakeholders, missed_stakeholders,
                                   covered_types, missed_types,
                                   covered_subtypes, missed_subtypes,
                                   top_n_subtypes=5)

            st.info("Coverage analysis done. Check charts below (if generated).")

            try:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image("stakeholder_coverage.png", caption="Stakeholder Gaps", use_container_width=True)
                with col2:
                    st.image("risk_type_coverage.png", caption="Risk Type Gaps", use_container_width=True)
                with col3:
                    st.image("risk_subtype_coverage.png", caption="Overlooked Subtype Gaps", use_container_width=True)
            except FileNotFoundError:
                st.warning("No coverage charts to display or chart generation failed.")

            # We can also use GPT to generate textual coverage feedback:
            domain = df['domain'].iloc[0] if 'domain' in df.columns else "AI deployment"
            joined_user_lines = "\n".join(f"- {u}" for u in user_lines)
            example_missed = missed_types[:3] if missed_types else []
            missed_str = ", ".join(example_missed) if example_missed else "None"

            feedback_prompt = f"""
You are an AI risk analysis expert for {domain}.
The user has these finalized risks:
{joined_user_lines}

Observed coverage gaps:
- Missed risk types: {missed_str}
(And possibly missed stakeholders or subtypes from the charts above.)

Provide a short textual feedback explaining why these gaps matter and how the user can address them. 
Emphasize the importance of thorough coverage for a comprehensive AI harms analysis.
"""
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"system","content":"You are a helpful coverage analysis assistant."},
                              {"role":"user","content": feedback_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                coverage_feedback = resp.choices[0].message.content.strip()
                st.subheader("Coverage Feedback (LLM-Generated):")
                st.write(coverage_feedback)
            except Exception as e:
                st.error(f"OpenAI coverage feedback error: {str(e)}")

        else:
            st.warning("No user-input risks found. Please paste or pull Mural data first, then re-run coverage analysis.")

    except FileNotFoundError:
        st.error(f"CSV file not found: {csv_file}")
    except Exception as e:
        st.error(f"Error loading or analyzing CSV: {str(e)}")

st.subheader("3️⃣ Brainstorm Additional Risks")
num_suggestions = st.slider("Number of Suggestions", 1, 7, 5)
risk_type_focus = st.selectbox("Focus Risk Type (optional)", ["Any","Technical","Financial","Ethical","Operational","Regulatory","Social","Legal","Unknown"])
user_stakeholder = st.text_input("Focus Stakeholder (optional)", value="Any")

if st.button("Generate Additional Risk Suggestions"):
    # We do a quick GPT call to brainstorm new risk lines:
    domain = "AI-based property valuations"  # Or read from your CSV
    prompt_b = f"""
You are an AI risk brainstorming assistant for {domain}.
We want about {num_suggestions} new risk ideas focusing on '{risk_type_focus}' type (if relevant)
and stakeholder '{user_stakeholder}' (if relevant).

Format each suggestion as a single bullet line that includes:
- A short risk description
- Why it matters
- (Optional) who might be impacted

Aim for new or overlooked angles not in the user's current list.
"""
    try:
        resp_b = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"You are a creative risk brainstorming assistant."},
                      {"role":"user","content":prompt_b}],
            max_tokens=800,
            temperature=0.9
        )
        suggestions_text = resp_b.choices[0].message.content.strip()
        st.subheader("New Brainstormed Risk Suggestions:")
        st.write(suggestions_text)
        # Optionally parse them into lines, etc.
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")

st.subheader("4️⃣ Provide Feedback on Brainstormed Risks (Optional)")
# If we wanted to let user vote on them or log feedback:
# (similar to earlier approach with log_feedback)
st.markdown("Use the text area or a button to provide feedback on new suggestions above, then log it if desired.")

# Done. 
st.info("End of Method 2. You can refine coverage, charts, or brainstorming logic as needed.")
