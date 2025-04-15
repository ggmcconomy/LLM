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
import seaborn as sns
import plotly.express as px

# Temporarily disable torch.classes to avoid Streamlit watcher error
sys.modules['torch.classes'] = None
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# --- Configuration ---
st.set_page_config(page_title="AI Risk Feedback & Brainstorming", layout="wide")
st.title("🤖 AI Risk Analysis Dashboard")

# Load secrets
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

# --- Utility Functions ---
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

def get_cluster_labels(df):
    """Generate descriptive labels for clusters based on risk descriptions."""
    cluster_labels = {}
    for cluster_id in df['cluster'].unique():
        cluster_risks = df[df['cluster'] == cluster_id]['risk_description'].dropna()
        if not cluster_risks.empty:
            text = ' '.join(cluster_risks).lower()
            words = re.findall(r'\b\w+\b', text)
            stopwords = {'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'on', 'with', 'by', 'at', 'this', 'but', 'from', 'or', 'an', 'are'}
            words = [w for w in words if w not in stopwords and len(w) > 3]
            word_counts = Counter(words)
            top_words = [word for word, _ in word_counts.most_common(2)]
            label = ' '.join(top_words).title() if top_words else f"Cluster {cluster_id}"
            cluster_labels[cluster_id] = label
        else:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
    return cluster_labels

def create_coverage_chart(title, categories, covered_counts, missed_counts, filename):
    """Create a single bar chart for coverage."""
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

def create_heatmap(df):
    """Create a heatmap for risk severity across risk types and drivers."""
    try:
        pivot_table = df.pivot_table(values='severity', index='risk_type', columns='driver', aggfunc='mean')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.1f')
        plt.title('Risk Severity Heatmap by Type and Driver')
        plt.tight_layout()
        plt.savefig('severity_heatmap.png')
        plt.close()
        return True
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return False

def create_time_series_chart(df):
    """Create a time series chart for average risk severity over time horizons."""
    try:
        time_horizons = ['Short-term', 'Medium-term', 'Long-term']
        avg_severity = df.groupby('time_horizon')['severity'].mean().reindex(time_horizons)
        plt.figure(figsize=(8, 6))
        plt.plot(time_horizons, avg_severity, marker='o')
        plt.xlabel('Time Horizon')
        plt.ylabel('Average Severity')
        plt.title('Average Risk Severity Over Time Horizons')
        plt.tight_layout()
        plt.savefig('time_series_chart.png')
        plt.close()
        return True
    except Exception as e:
        st.error(f"Error creating time series chart: {str(e)}")
        return False

def create_interactive_dashboard(df):
    """Create an interactive dashboard for risk exploration using Plotly."""
    try:
        fig = px.scatter(df, x='risk_description', y='severity', color='risk_type',
                         hover_data=['stakeholder', 'driver', 'time_horizon'],
                         title='Interactive Risk Dashboard')
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label='All Stakeholders',
                             method='update',
                             args=[{'visible': [True] * len(df)}]),
                        *[
                            dict(label=s,
                                 method='update',
                                 args=[{'visible': df['stakeholder'] == s}])
                            for s in df['stakeholder'].unique()
                        ]
                    ]),
                    direction='down',
                    showactive=True,
                ),
            ]
        )
        fig.write_html('interactive_dashboard.html')
        return True
    except Exception as e:
        st.error(f"Error creating interactive dashboard: {str(e)}")
        return False

def create_coverage_charts(covered_stakeholders, missed_stakeholders, covered_types, missed_types, covered_subtypes, missed_subtypes, covered_horizons, missed_horizons, covered_drivers, missed_drivers, top_n_subtypes=5):
    """Create bar charts for coverage visualization including time horizons and drivers."""
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

    # Time Horizon Chart
    horizons = sorted(set(covered_horizons + missed_horizons))
    covered_counts = [covered_horizons.count(h) for h in horizons]
    missed_counts = [missed_horizons.count(h) for h in horizons]
    non_zero_indices = [i for i, (c, m) in enumerate(zip(covered_counts, missed_counts)) if c > 0 or m > 0]
    horizons = [horizons[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    
    if horizons:
        create_coverage_chart("Time Horizon Coverage Gaps", horizons, covered_counts, missed_counts, 'time_horizon_coverage.png')
    else:
        st.warning("No time horizon data to display.")

    # Driver Chart
    drivers = sorted(set(covered_drivers + missed_drivers))
    covered_counts = [covered_drivers.count(d) for d in drivers]
    missed_counts = [missed_drivers.count(d) for d in drivers]
    non_zero_indices = [i for i, (c, m) in enumerate(zip(covered_counts, missed_counts)) if c > 0 or m > 0]
    drivers = [drivers[i] for i in non_zero_indices]
    covered_counts = [covered_counts[i] for i in non_zero_indices]
    missed_counts = [missed_counts[i] for i in non_zero_indices]
    
    if drivers:
        create_coverage_chart("Driver Coverage Gaps", drivers, covered_counts, missed_counts, 'driver_coverage.png')
    else:
        st.warning("No driver data to display.")

# --- OAuth Functions ---
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

# --- Mural API Functions ---
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

# --- Handle OAuth Flow ---
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

# --- Load Pre-Clustered Data ---
csv_file = 'clean_risks.csv'
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
    st.error(f"Clustered CSV {csv_file} not found. Please run generate_clustered_files.py first.")
    st.stop()

try:
    csv_embeddings = np.load(embeddings_file)
except FileNotFoundError:
    st.error(f"Embeddings file {embeddings_file} not found. Please run generate_clustered_files.py first.")
    st.stop()

try:
    index = faiss.read_index(index_file)
except FileNotFoundError:
    st.error(f"Index file {index_file} not found. Please run generate_clustered_files.py first.")
    st.stop()

# Initialize embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate cluster labels
cluster_labels = get_cluster_labels(df)

# --- Sidebar Settings ---
with st.sidebar:
    st.header("🔧 Settings")
    num_clusters = st.slider("Number of Clusters (Themes)", 5, 20, 10)
    severity_threshold = st.slider("Severity Threshold", 0.0, 5.0, 4.0, 0.5)
    st.markdown("---")
    st.subheader("📥 Mural Actions")
    custom_mural_id = st.text_input("Custom Mural ID (optional)", value=MURAL_BOARD_ID)
    if st.button("🔍 List Murals"):
        with st.spinner("Listing murals..."):
            murals = list_murals(st.session_state.access_token)
            if murals:
                st.write("Available Murals:", [{"id": m["id"], "title": m.get("title", "Untitled")} for m in murals])
            else:
                st.warning("No murals found.")
    if st.button("🔄 Pull Sticky Notes"):
        with st.spinner("Pulling sticky notes..."):
            try:
                headers = {'Authorization': f'Bearer {st.session_state.access_token}'}
                mural_id = custom_mural_id or MURAL_BOARD_ID
                if not verify_mural(st.session_state.access_token, mural_id):
                    mural_id = normalize_mural_id(mural_id)
                    if not verify_mural(st.session_state.access_token, mural_id):
                        st.error(f"Mural {mural_id} not found.")
                        st.stop()
                url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
                session = requests.Session()
                retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
                session.mount('https://', HTTPAdapter(max_retries=retries))
                mural_data = session.get(url, headers=headers, timeout=10)
                if mural_data.status_code == 200:
                    widgets = mural_data.json().get("value", mural_data.json().get("data", []))
                    sticky_widgets = [w for w in widgets if w.get('type', '').replace(' ', '_').lower() == 'sticky_note']
                    stickies = []
                    for w in sticky_widgets:
                        raw_text = w.get('htmlText') or w.get('text') or ''
                        if raw_text:
                            cleaned_text = clean_html_text(raw_text)
                            if cleaned_text:
                                stickies.append(cleaned_text)
                    st.session_state['mural_notes'] = stickies
                    st.success(f"Pulled {len(stickies)} sticky notes.")
                else:
                    st.error(f"Failed to pull sticky notes: {mural_data.status_code}")
                    if mural_data.status_code == 401:
                        st.warning("OAuth token invalid. Please re-authenticate.")
                        st.session_state.access_token = None
                        auth_url = get_authorization_url()
                        st.markdown(f"[Re-authorize the app]({auth_url}).")
                    elif mural_data.status_code == 403:
                        st.warning("Access denied. Ensure collaborator access.")
                    elif mural_data.status_code == 404:
                        st.warning(f"Mural ID {mural_id} not found.")
            except Exception as e:
                st.error(f"Error connecting to Mural: {str(e)}")
    if st.button("🗑️ Clear Session"):
        st.session_state.clear()
        st.rerun()

# --- Section 1: Input Risks ---
st.subheader("1️⃣ Input Risks")
st.write("Pull finalized risks from Mural or edit below to analyze coverage.")
default_notes = st.session_state.get('mural_notes', [])
default_text = "\n".join(default_notes) if default_notes else ""
user_input = st.text_area("", value=default_text, height=200, placeholder="Enter risks, one per line.")

# --- Section 2: Generate Coverage Feedback ---
st.subheader("2️⃣ Coverage Feedback")
st.write("Analyze gaps in your risk coverage with examples.")
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
                covered_types = {r['risk_type'] for r in similar_risks}
                covered_subtypes = {r['risk_subtype'] for r in similar_risks}
                covered_stakeholders = {r['stakeholder'] for r in similar_risks}
                covered_horizons = {r['time_horizon'] for r in similar_risks}
                covered_drivers = {r['driver'] for r in similar_risks}
                covered_clusters = {r['cluster'] for r in similar_risks}

                # Find missed and underrepresented areas
                missed_types = sorted(list(set(df['risk_type']) - covered_types))
                missed_subtypes = sorted(list(set(df['risk_subtype']) - covered_subtypes))
                missed_stakeholders = sorted(list(set(df['stakeholder']) - covered_stakeholders))
                missed_horizons = sorted(list(set(df['time_horizon']) - covered_horizons))
                missed_drivers = sorted(list(set(df['driver']) - covered_drivers))
                missed_clusters = sorted(list(set(df['cluster']) - covered_clusters))

                # Identify underrepresented areas
                human_risk_types = [r['risk_type'] for r in similar_risks]
                human_subtypes = [r['risk_subtype'] for r in similar_risks]
                human_stakeholders = [r['stakeholder'] for r in similar_risks]
                human_horizons = [r['time_horizon'] for r in similar_risks]
                human_drivers = [r['driver'] for r in similar_risks]
                human_clusters = [r['cluster'] for r in similar_risks]

                underrepresented_types = [t for t in df['risk_type'].unique() if human_risk_types.count(t) < df['risk_type'].value_counts()[t] * 0.1]
                underrepresented_subtypes = [s for s in df['risk_subtype'].unique() if human_subtypes.count(s) < df['risk_subtype'].value_counts()[s] * 0.1]
                underrepresented_stakeholders = [s for s in df['stakeholder'].unique() if human_stakeholders.count(s) < df['stakeholder'].value_counts()[s] * 0.1]
                underrepresented_horizons = [h for h in df['time_horizon'].unique() if human_horizons.count(h) < df['time_horizon'].value_counts()[h] * 0.1]
                underrepresented_drivers = [d for d in df['driver'].unique() if human_drivers.count(d) < df['driver'].value_counts()[d] * 0.1]
                underrepresented_clusters = [c for c in df['cluster'].unique() if human_clusters.count(c) < df['cluster'].value_counts()[c] * 0.1]

                # Prepare context examples
                context_examples = []
                for category, items in [
                    ("Missed Risk Types", missed_types),
                    ("Missed Risk Subtypes", missed_subtypes),
                    ("Missed Stakeholders", missed_stakeholders),
                    ("Missed Time Horizons", missed_horizons),
                    ("Missed Drivers", missed_drivers),
                    ("Missed Clusters", missed_clusters),
                    ("Underrepresented Risk Types", underrepresented_types),
                    ("Underrepresented Risk Subtypes", underrepresented_subtypes),
                    ("Underrepresented Stakeholders", underrepresented_stakeholders),
                    ("Underrepresented Time Horizons", underrepresented_horizons),
                    ("Underrepresented Drivers", underrepresented_drivers),
                    ("Underrepresented Clusters", underrepresented_clusters)
                ]:
                    if items:
                        for item in items[:3]:
                            if "Types" in category:
                                example_rows = df[df['risk_type'] == item].head(1)
                            elif "Subtypes" in category:
                                example_rows = df[df['risk_subtype'] == item].head(1)
                            elif "Stakeholders" in category:
                                example_rows = df[df['stakeholder'] == item].head(1)
                            elif "Horizons" in category:
                                example_rows = df[df['time_horizon'] == item].head(1)
                            elif "Drivers" in category:
                                example_rows = df[df['driver'] == item].head(1)
                            elif "Clusters" in category:
                                example_rows = df[df['cluster'] == item].head(1)
                            if not example_rows.empty:
                                example = example_rows.iloc[0]
                                context_examples.append(f"{category}: {item} - Example: {example['risk_description']} (Type: {example['risk_type']}, Subtype: {example['risk_subtype']}, Stakeholder: {example['stakeholder']}, Time Horizon: {example['time_horizon']}, Driver: {example['driver']}, Cluster: {cluster_labels[example['cluster']]})")

                context_str = "\n".join(context_examples)

                # Prepare coverage feedback prompt
                domain = df['domain'].iloc[0] if 'domain' in df.columns else "AI deployment"
                prompt = f"""
                You are an AI risk analysis expert for {domain}, focusing solely on harms identification. The user has identified these finalized risks from Mural:
                {chr(10).join(f'- {r}' for r in human_risks)}

                Using the following examples from the risk database:
                {context_str}

                Provide feedback on the gaps in the user's harms analysis, focusing on risk types, subtypes, stakeholders, time horizons, drivers, and clusters that are overlooked or insufficiently developed:

                1. **Missing Risk Types, Subtypes, Stakeholders, Time Horizons, Drivers, or Clusters**:
                   - Identify categories completely missing from the user's risks and explain why they are critical.
                   - Use the provided examples to illustrate comprehensive coverage.

                2. **Underrepresented Risk Types, Subtypes, Stakeholders, Time Horizons, Drivers, or Clusters**:
                   - Highlight categories that lack depth or breadth.
                   - Explain how this limits the analysis and use examples to show what could be added.

                3. **Suggestions for Improvement**:
                   - Offer actionable advice on addressing these gaps by adding or expanding risks in Mural.

                Ensure feedback is constructive and tied to the examples, emphasizing comprehensive harms analysis.
                """

                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful AI risk advisor specializing in harms identification."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    feedback = response.choices[0].message.content
                except Exception as e:
                    st.error(f"OpenAI API error: {str(e)}")
                    feedback = None

                if feedback:
                    # Prepare data for coverage charts
                    covered_stakeholders_list = list(covered_stakeholders)
                    covered_types_list = list(covered_types)
                    covered_subtypes_list = list(covered_subtypes)
                    covered_horizons_list = list(covered_horizons)
                    covered_drivers_list = list(covered_drivers)
                    missed_stakeholders_list = list(missed_stakeholders)
                    missed_types_list = list(missed_types)
                    missed_subtypes_list = list(missed_subtypes)
                    missed_horizons_list = list(missed_horizons)
                    missed_drivers_list = list(missed_drivers)

                    # Generate coverage charts
                    create_coverage_charts(
                        covered_stakeholders_list, missed_stakeholders_list,
                        covered_types_list, missed_types_list,
                        covered_subtypes_list, missed_subtypes_list,
                        covered_horizons_list, missed_horizons_list,
                        covered_drivers_list, missed_drivers_list,
                        top_n_subtypes=top_n_subtypes
                    )

                    # Generate new visuals
                    create_heatmap(df)
                    create_time_series_chart(df)
                    create_interactive_dashboard(df)

                    st.session_state['feedback'] = feedback
                    st.session_state['coverage_data'] = {
                        'covered_stakeholders': covered_stakeholders_list,
                        'missed_stakeholders': missed_stakeholders_list,
                        'covered_types': covered_types_list,
                        'missed_types': missed_types_list,
                        'covered_subtypes': covered_subtypes_list,
                        'missed_subtypes': missed_subtypes_list,
                        'covered_horizons': covered_horizons_list,
                        'missed_horizons': missed_horizons_list,
                        'covered_drivers': covered_drivers_list,
                        'missed_drivers': missed_drivers_list
                    }
            except Exception as e:
                st.error(f"Error processing risks: {str(e)}")
        else:
            st.warning("Please enter or pull some risks first.")

# --- Section 3: Coverage Visualization ---
if 'coverage_data' in st.session_state:
    st.subheader("3️⃣ Coverage Visualization")
    st.write("View gaps in risk coverage to identify weaknesses.")
    col1, col2, col3 = st.columns(3)
    try:
        with col1:
            st.image("stakeholder_coverage.png", caption="Stakeholder Gaps", use_column_width=True)
            st.image("severity_heatmap.png", caption="Risk Severity Heatmap: Darker colors indicate higher severity.", use_column_width=True)
        with col2:
            st.image("risk_type_coverage.png", caption="Risk Type Gaps", use_column_width=True)
            st.image("time_series_chart.png", caption="Risk Severity Over Time", use_column_width=True)
        with col3:
            st.image("risk_subtype_coverage.png", caption=f"Top {top_n_subtypes} Overlooked Subtype Gaps", use_column_width=True)
            st.markdown("[Interactive Dashboard](interactive_dashboard.html)")
    except FileNotFoundError:
        st.error("Coverage charts failed to generate. Please try generating feedback again.")

# --- Section 4: Coverage Feedback (Textual) ---
if 'feedback' in st.session_state:
    st.subheader("4️⃣ Coverage Feedback (Textual)")
    st.write("Review gaps in your risk analysis with examples to inspire additions to Mural.")
    st.markdown("### Coverage Feedback:")
    st.markdown(st.session_state['feedback'])
    st.info("Add inspired risks to Mural based on these examples, then re-pull Mural data for further analysis.")

# --- Section 5: Brainstorm Risks ---
st.subheader("5️⃣ Brainstorm Risks")
st.write("Generate creative risk suggestions to broaden your analysis.")
num_brainstorm_risks = st.slider("Number of Suggestions", 1, 5, 5, key="num_brainstorm_risks")
stakeholder_options = sorted(df['stakeholder'].dropna().unique())
risk_type_options = sorted(df['risk_type'].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    stakeholder = st.selectbox("Target Stakeholder (optional):", ["Any"] + stakeholder_options, key="brainstorm_stakeholder")
with col2:
    risk_type = st.selectbox("Target Risk Type (optional):", ["Any"] + risk_type_options, key="brainstorm_risk_type")

if st.button("💡 Generate Risk Suggestions", key="generate_risk_suggestions"):
    with st.spinner("Generating ideas..."):
        try:
            filt = df.copy()
            if stakeholder != "Any":
                filt = filt[filt['stakeholder'] == stakeholder]
            if risk_type != "Any":
                filt = filt[filt['risk_type'] == risk_type]
            top_suggestions = filt.sort_values(by='combined_score', ascending=False).head(num_brainstorm_risks)

            suggestions = "\n".join(f"- {r['risk_description']} (Type: {r['risk_type']}, Subtype: {r['risk_subtype']}, Stakeholder: {r['stakeholder']}, Time Horizon: {r['time_horizon']}, Driver: {r['driver']})" for r in top_suggestions.to_dict('records'))

            domain = df['domain'].iloc[0] if 'domain' in df.columns else "AI deployment"

            prompt = f"""
            You are a creative AI risk analysis expert for {domain}. Based on these high-priority risks:
            {suggestions}

            Generate {num_brainstorm_risks} new risk suggestions to broaden the risk analysis. Focus on diverse, overlooked risks that complement the existing ones. For each suggestion, include:
            - A concise risk description
            - Risk Type
            - Risk Subtype
            - Stakeholder
            - Time Horizon
            - Driver
            - Why it matters

            Format each suggestion as:
            - Risk: [description] (Type: [type], Subtype: [subtype], Stakeholder: [stakeholder], Time Horizon: [horizon], Driver: [driver], Why it matters: [reason])
            """

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a creative AI risk brainstorming assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            brainstorm_output = response.choices[0].message.content

            brainstorm_suggestions = [s.strip() for s in brainstorm_output.split('\n') if s.strip().startswith('- Risk:')]
            brainstorm_suggestions = [s[2:].strip() for s in brainstorm_suggestions]

            if not brainstorm_suggestions:
                st.warning("No suggestions were generated.")
            else:
                st.session_state['brainstorm_suggestions'] = brainstorm_suggestions[:num_brainstorm_risks]

        except Exception as e:
            st.error(f"Error generating risk suggestions: {str(e)}")

# Display Brainstorming Suggestions with Feedback
if 'brainstorm_suggestions' in st.session_state and st.session_state['brainstorm_suggestions']:
    st.markdown("### Brainstormed Risk Suggestions:")
    st.write("Vote on creative risk ideas to add to Mural, or disagree with a reason.")
    
    for idx, suggestion in enumerate(st.session_state['brainstorm_suggestions']):
        suggestion_key = f"brainstorm_{idx}"
        short_text = suggestion[:200] + ("..." if len(suggestion) > 200 else "")
        
        st.markdown(f"**Suggestion {idx + 1}:** {short_text}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Agree", key=f"agree_{suggestion_key}"):
                log_feedback(suggestion, "agree")
                st.success("Thanks! Copy this suggestion to add it to Mural manually, then re-pull Mural data.")
        with col2:
            if st.button("👎 Disagree", key=f"disagree_{suggestion_key}"):
                st.session_state[f"show_disagree_{suggestion_key}"] = True
        
        if st.session_state.get(f"show_disagree_{suggestion_key}", False):
            with st.form(key=f"disagree_form_{suggestion_key}"):
                disagreement_reason = st.text_area("Why do you disagree?", key=f"reason_{suggestion_key}", height=100)
                if st.form_submit_button("Submit"):
                    if disagreement_reason.strip():
                        log_feedback(suggestion, "disagree", disagreement_reason)
                        st.success("Disagreement noted. Thanks for your input!")
                        st.session_state[f"show_disagree_{suggestion_key}"] = False
                    else:
                        st.error("Please provide a reason.")
else:
    st.info("No suggestions available. Click 'Generate Risk Suggestions' to create new ideas.")

# --- Section 6: Mitigation Strategies ---
st.subheader("6️⃣ Mitigation Strategies")
st.write("Review human-centric mitigation strategies for each finalized Mural risk.")
if st.button("🔧 Generate Mitigation Strategies"):
    with st.spinner("Generating mitigation strategies..."):
        if user_input.strip():
            try:
                human_risks = [r.strip() for r in user_input.split('\n') if r.strip()]
                domain = df['domain'].iloc[0] if 'domain' in df.columns else "AI deployment"
                
                mitigation_strategies = []
                for risk in human_risks:
                    risk_embedding = embedder.encode([risk])
                    distances, indices = index.search(risk_embedding, 1)
                    similar_risk = df.iloc[indices[0][0]].to_dict()
                    
                    prompt = f"""
                    You are an AI risk mitigation expert for {domain}, specializing in human-centric design. For the following finalized risk:
                    - {risk}

                    Consider the following similar risk from the database:
                    - Description: {similar_risk['risk_description']}
                    - Type: {similar_risk['risk_type']}
                    - Subtype: {similar_risk['risk_subtype']}
                    - Stakeholder: {similar_risk['stakeholder']}
                    - Time Horizon: {similar_risk['time_horizon']}
                    - Driver: {similar_risk['driver']}

                    Provide 1-2 human-centric mitigation strategies that prioritize user needs, ethical considerations, and practical implementation. Include a brief example for each strategy.

                    Format the response as:
                    - Strategy 1: [description] (Example: [example])
                    - Strategy 2: [description] (Example: [example])
                    """
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a helpful AI risk mitigation advisor specializing in human-centric design."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        mitigation = response.choices[0].message.content
                        mitigation_strategies.append({"risk": risk, "mitigation": mitigation})
                    except Exception as e:
                        st.error(f"OpenAI API error for risk '{risk}': {str(e)}")
                        mitigation_strategies.append({"risk": risk, "mitigation": "Error generating mitigation strategies."})

                st.session_state['mitigation_strategies'] = mitigation_strategies
            except Exception as e:
                st.error(f"Error processing mitigation strategies: {str(e)}")
        else:
            st.warning("Please enter or pull some risks first.")

# Display Mitigation Strategies
if 'mitigation_strategies' in st.session_state:
    st.markdown("### Mitigation Strategies for Finalized Risks:")
    for idx, item in enumerate(st.session_state['mitigation_strategies']):
        st.markdown(f"**Risk {idx + 1}:** {item['risk']}")
        st.markdown(f"**Mitigation Strategies:**")
        st.markdown(item['mitigation'])
        st.markdown("---")
