#############################################################
# method2.py
#
# Re-uses your existing pipeline code for Mural connection,
# but changes the CSV to "clean_risks.csv" and adds a RAG table
# for improved gap visuals. Otherwise the Mural OAuth logic is unchanged.
#############################################################

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
sys.modules['torch.classes'] = None

# For embeddings + coverage checks
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

#############################################################
# --- Configuration & Page ---
#############################################################
st.set_page_config(page_title="Method 2 - AI Risk Coverage & Brainstorming", layout="wide")
st.title("🤖 AI Risk Analysis Dashboard - Method 2 (Using Mural + clean_risks.csv)")

#############################################################
# 1) Load secrets
#############################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "aiimpacttesting2642")
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets in .streamlit/secrets.toml.")
    st.stop()

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

#############################################################
# 2) Utility Functions
#############################################################
def normalize_mural_id(mural_id, workspace_id=MURAL_WORKSPACE_ID):
    """Strip workspace prefix from mural ID if present."""
    prefix = f"{workspace_id}."
    if mural_id.startswith(prefix):
        return mural_id[len(prefix):]
    return mural_id

def clean_html_text(html_text):
    """Strip HTML tags and clean text."""
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
    """Log user feedback to CSV."""
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

def color_cell(val):
    """
    Simple RAG scale for coverage:
     0 => Red
     1..2 => Amber
     >=3 => Green
    """
    if val == 0:
        return "background-color:#e74c3c; color:white"
    elif 1 <= val <=2:
        return "background-color:#f39c12; color:white"
    else:
        return "background-color:#2ecc71; color:white"

def create_coverage_chart(title, categories, covered_counts, missed_counts, filename):
    """Create a single bar chart for coverage."""
    try:
        plt.figure(figsize=(6, 4))
        x = np.arange(len(categories))
        plt.bar(x - 0.2, covered_counts, 0.4, label='Covered', color='#2ecc71')
        plt.bar(x + 0.2, missed_counts, 0.4, label='Missed', color='#e74c3c')
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
    """Create bar charts for coverage visualization."""
    try:
        plt.style.use('ggplot')
    except Exception as e:
        st.warning(f"ggplot style failed: {str(e)}. Using default style.")
        plt.style.use('default')

    # Stakeholder Chart
    stakeholders = sorted(set(covered_stakeholders + missed_stakeholders))
    covered_counts = [covered_stakeholders.count(s) for s in stakeholders]
    missed_counts = [missed_stakeholders.count(s) for s in stakeholders]
    non_zero_indices = [i for i, (c, m) in enumerate(zip(covered_counts, missed_counts)) if c > 0 or m > 0]
    stakeholders = [stakeholders[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    
    if stakeholders:
        create_coverage_chart("Stakeholder Coverage Gaps", stakeholders, covered_counts, missed_counts, 'stakeholder_coverage.png')
    else:
        st.warning("No stakeholder data to display.")

    # Risk Type Chart
    risk_types = sorted(set(covered_types + missed_types))
    covered_counts = [covered_types.count(t) for t in risk_types]
    missed_counts = [missed_types.count(t) for t in risk_types]
    non_zero_indices = [i for i, (c, m) in enumerate(zip(covered_counts, missed_counts)) if c > 0 or m > 0]
    risk_types = [risk_types[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    
    if risk_types:
        create_coverage_chart("Risk Type Coverage Gaps", risk_types, covered_counts, missed_counts, 'risk_type_coverage.png')
    else:
        st.warning("No risk type data to display.")

    # Risk Subtype Chart (Top N Missed Only)
    subtype_counts = Counter(missed_subtypes)
    top_missed_subtypes = [subtype for subtype, _ in subtype_counts.most_common(top_n_subtypes)]
    covered_counts = [covered_subtypes.count(s) for s in top_missed_subtypes]
    missed_counts = [missed_subtypes.count(s) for s in top_missed_subtypes]
    
    if top_missed_subtypes:
        create_coverage_chart(f"Top {top_n_subtypes} Overlooked Risk Subtype Gaps", top_missed_subtypes, covered_counts, missed_counts, 'risk_subtype_coverage.png')
    else:
        st.warning("No risk subtype data to display.")

#############################################################
# --- OAuth Functions ---
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
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Authentication failed: {response.status_code}")
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
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Token refresh failed: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Token refresh error: {str(e)}")
            return None

#############################################################
# 3) Mural API
#############################################################
def list_murals(auth_token):
    url = "https://app.mural.co/api/public/v1/murals"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("value", [])
        else:
            st.error(f"Failed to list murals: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error listing murals: {str(e)}")
        return []

def verify_mural(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error verifying mural: {str(e)}")
        return False

#############################################################
# --- Handle OAuth Flow ---
#############################################################
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
        st.set_query_params()  # remove code
        st.success("Authenticated with Mural!")
        st.rerun()

if not st.session_state.access_token:
    auth_url = get_authorization_url()
    st.markdown(f"Please [authorize the app]({auth_url}) to access Mural.")
    st.info("Click the link above, log into Mural, and authorize.")
    st.stop()

if st.session_state.access_token:
    current_time = pd.Timestamp.now().timestamp()
    if (current_time - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        token_data = refresh_access_token(st.session_state.refresh_token)
        if token_data:
            st.session_state.access_token = token_data["access_token"]
            st.session_state.refresh_token = token_data.get("refresh_token", st.session_state.refresh_token)
            st.session_state.token_expires_in = token_data.get("expires_in", 900)
            st.session_state.token_timestamp = pd.Timestamp.now().timestamp()

#############################################################
# --- Load "clean_risks.csv" ---
#############################################################
csv_file = 'clean_risks.csv'  # <--- CHANGED to "clean_risks.csv"
embeddings_file = 'embeddings.npy'
index_file = 'faiss_index.faiss'

try:
    df = pd.read_csv(csv_file)
    if 'overlooked_label' in df.columns:
        df = df.drop(columns=['overlooked_label'])
    numeric_columns = ['severity', 'probability', 'combined_score']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                non_numeric_rows = df[df[col].isna()].index.tolist()
                st.warning(f"Non-numeric values found in {col} at rows: {non_numeric_rows}. Converted to NaN.")
except FileNotFoundError:
    st.error(f"Clean CSV {csv_file} not found. Please run Method 1 first.")
    st.stop()

try:
    csv_embeddings = np.load(embeddings_file)
except FileNotFoundError:
    st.error(f"Embeddings file {embeddings_file} not found. Please run the embedding step first.")
    st.stop()

try:
    index = faiss.read_index(index_file)
except FileNotFoundError:
    st.error(f"Index file {index_file} not found. Please run the embedding step first.")
    st.stop()

# Initialize embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

#############################################################
# --- Domain coverage logic (unchanged, or minimal changes) ---
#############################################################

st.sidebar.header("🔧 Settings")
num_clusters = st.sidebar.slider("Number of Clusters (Themes)", 5, 20, 10)
severity_threshold = st.sidebar.slider("Severity Threshold", 0.0, 5.0, 4.0, 0.5)

st.markdown("---")
st.subheader("1️⃣ Input Risks")
st.write("Pull finalized risks from Mural or edit below to analyze coverage.")

default_notes = st.session_state.get('mural_notes', [])
default_text = "\n".join(default_notes) if default_notes else ""
user_input = st.text_area("", value=default_text, height=200, placeholder="Enter risks, one per line.")

#############################################################
# --- Generate Coverage Feedback (with new CSV) ---
#############################################################
st.subheader("2️⃣ Coverage Feedback")
st.write("Analyze gaps in your risk coverage with examples from 'clean_risks.csv'.")

top_n_subtypes = st.slider("Top N Overlooked Subtypes to Display", 3, 10, 5)
if st.button("🔍 Generate Coverage Feedback"):
    with st.spinner("Analyzing coverage..."):
        if user_input.strip():
            try:
                human_risks = [r.strip() for r in user_input.split('\n') if r.strip()]
                human_embeddings = np.array(embedder.encode(human_risks))
                distances, indices = index.search(human_embeddings, 5)
                similar_risks = [df.iloc[idx].to_dict() for idx in indices.flatten()]

                # Analyze coverage
                covered_types = {r['risk_type'] for r in similar_risks if 'risk_type' in r}
                covered_subtypes = {r['risk_subtype'] for r in similar_risks if 'risk_subtype' in r}
                covered_stakeholders = {r['stakeholder'] for r in similar_risks if 'stakeholder' in r}

                # Missed
                all_types = df['risk_type'].dropna().unique()
                all_subtypes = df['risk_subtype'].dropna().unique() if 'risk_subtype' in df.columns else []
                all_stakeholders = df['stakeholder'].dropna().unique() if 'stakeholder' in df.columns else []

                missed_types = sorted(list(set(all_types) - covered_types))
                missed_subtypes = sorted(list(set(all_subtypes) - covered_subtypes))
                missed_stakeholders = sorted(list(set(all_stakeholders) - covered_stakeholders))

                # Prepare coverage visuals
                st.write(f"**Covered Risk Types**: {sorted(list(covered_types))}, Missed: {missed_types}")
                st.write(f"**Covered Subtypes**: {sorted(list(covered_subtypes))}, Missed: {missed_subtypes}")
                st.write(f"**Covered Stakeholders**: {sorted(list(covered_stakeholders))}, Missed: {missed_stakeholders}")

                # Create coverage charts
                create_coverage_charts(
                    list(covered_stakeholders), missed_stakeholders,
                    list(covered_types), missed_types,
                    list(covered_subtypes), missed_subtypes,
                    top_n_subtypes=top_n_subtypes
                )

                # Show newly generated images
                col1, col2, col3 = st.columns(3)
                try:
                    with col1:
                        st.image("stakeholder_coverage.png", caption="Stakeholder Gaps", use_column_width=True)
                    with col2:
                        st.image("risk_type_coverage.png", caption="Risk Type Gaps", use_container_width=True)
                    with col3:
                        st.image("risk_subtype_coverage.png", caption="Top Overlooked Subtype Gaps", use_container_width=True)
                except FileNotFoundError:
                    st.warning("No coverage images. Try again?")

                # Example LLM feedback
                domain = df['domain'].iloc[0] if 'domain' in df.columns else "AI deployment"
                prompt = f"""
                You are an AI risk analysis expert for {domain}. The user has identified these final lines:
                {chr(10).join("- "+r for r in human_risks)}

                We see coverage on:
                - risk_types covered: {sorted(list(covered_types))}
                - missed: {missed_types}
                - subtypes covered: {sorted(list(covered_subtypes))}
                - missed: {missed_subtypes}
                - stakeholders covered: {sorted(list(covered_stakeholders))}
                - missed: {missed_stakeholders}

                Provide a short gap analysis, focusing on why the missed areas matter and how to improve coverage.
                """

                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful coverage feedback assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300
                    )
                    st.markdown("### LLM Gap Feedback:")
                    st.write(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"OpenAI error: {str(e)}")

            except Exception as e:
                st.error(f"Error processing coverage: {str(e)}")
        else:
            st.warning("Please enter or pull some risks first.")

#############################################################
# 4) Coverage Visualization (If stored in session?)
#############################################################
# Similar to your snippet's code.

# --- Additional Brainstorm / Mitigation if needed... ---
# (Omitted or add if you want)

#############################################################
# 5) Clean Up
#############################################################
st.subheader("Done.")
if st.button("🗑 Clear Session"):
    st.session_state.clear()
    st.set_query_params()  # remove code param
    st.rerun()
