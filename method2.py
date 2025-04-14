import os
import json
import uuid
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import openai
import faiss
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse, urlunparse

###############################################################################
# 1) Page Config & Title
###############################################################################
st.set_page_config(page_title="Method 2 - Comprehensive Tool", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2) - No rerun Error")
st.write("DEBUG: Page config and title rendered")

###############################################################################
# 2) Load Secrets (Mural, OpenAI)
###############################################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "myworkspace")
    st.write("DEBUG: Secrets loaded successfully")
except KeyError as e:
    st.error(f"Missing secret: {e}. Please set Mural and OpenAI secrets properly.")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MURAL_CLIENT_ID = os.getenv("MURAL_CLIENT_ID", "")
    MURAL_CLIENT_SECRET = os.getenv("MURAL_CLIENT_SECRET", "")
    MURAL_REDIRECT_URI = os.getenv("MURAL_REDIRECT_URI", "")
    MURAL_BOARD_ID = os.getenv("MURAL_BOARD_ID", "")
    MURAL_WORKSPACE_ID = os.getenv("MURAL_WORKSPACE_ID", "myworkspace")
    st.write("DEBUG: Fallback to environment variables")

openai.api_key = OPENAI_API_KEY

# Normalize redirect URI to remove trailing slashes
def normalize_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', '', ''))

MURAL_REDIRECT_URI = normalize_url(MURAL_REDIRECT_URI)
st.write(f"DEBUG: Using redirect_uri: {MURAL_REDIRECT_URI}")

# Validate secrets
if not all([MURAL_CLIENT_ID, MURAL_CLIENT_SECRET, MURAL_REDIRECT_URI, MURAL_BOARD_ID]):
    st.error("Critical Mural secrets missing. Please check MURAL_CLIENT_ID, MURAL_CLIENT_SECRET, MURAL_REDIRECT_URI, MURAL_BOARD_ID.")
    st.stop()

###############################################################################
# 3) Mural OAuth & Auth
###############################################################################
if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.token_expires_in = None
    st.session_state.token_timestamp = None
st.write("DEBUG: Session state initialized")

def get_authorization_url():
    from urllib.parse import urlencode
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    auth_url = "https://app.mural.co/api/public/v1/authorization/oauth2/?" + urlencode(params)
    st.write("DEBUG: Authorization URL generated")
    return auth_url

def exchange_code_for_token(code, max_retries=2):
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "redirect_uri": MURAL_REDIRECT_URI,
        "code": code,
        "grant_type": "authorization_code"
    }
    for attempt in range(max_retries):
        try:
            st.write(f"DEBUG: Attempting token exchange (try {attempt + 1}/{max_retries})")
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
                st.write("DEBUG: Token exchange successful")
                return resp.json()
            else:
                try:
                    error_detail = resp.json().get("error_description", "No error description provided")
                    error_code = resp.json().get("error", "Unknown error")
                except ValueError:
                    error_detail = resp.text
                    error_code = "Parsing error"
                st.error(f"Mural Auth failed: {resp.status_code} - {error_code}: {error_detail}")
                if resp.status_code == 400:
                    if "redirect" in error_detail.lower():
                        st.error(
                            f"Redirect URI mismatch detected. Sent: '{MURAL_REDIRECT_URI}'. "
                            "Please ensure this EXACTLY matches the redirect URI in Mural's app settings at "
                            "https://app.mural.co/developers. Common issues: trailing slashes, http vs https, or wrong domain."
                        )
                    else:
                        st.warning(
                            "Other 400 error causes:\n"
                            "- Invalid or expired authorization code.\n"
                            "- Incorrect client_id or client_secret.\n"
                            "Please reauthorize and verify settings."
                        )
                if attempt < max_retries - 1:
                    st.info("Retrying token exchange...")
                    continue
                return None
        except requests.RequestException as e:
            st.error(f"Mural Auth network error: {str(e)}")
            if attempt < max_retries - 1:
                st.info("Retrying token exchange...")
                continue
            return None
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
        st.write("DEBUG: Attempting token refresh")
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            st.write("DEBUG: Token refresh successful")
            return resp.json()
        else:
            error_detail = resp.json().get("error_description", "No error description provided")
            st.error(f"Mural refresh failed: {resp.status_code} - {error_detail}")
            return None
    except requests.RequestException as e:
        st.error(f"Mural refresh network error: {str(e)}")
        return None
    except ValueError as e:
        st.error(f"Mural refresh response parsing error: {str(e)}")
        return None

def verify_mural(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {auth_token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        st.write("DEBUG: Mural verification status: " + str(resp.status_code))
        return resp.status_code == 200
    except requests.RequestException as e:
        st.error(f"Mural verification error: {str(e)}")
        return False

def pull_mural_stickies(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {auth_token}"}
    try:
        st.write("DEBUG: Pulling Mural sticky notes")
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            widgets = data.get("value", data.get("data", []))
            note_widgets = [w for w in widgets if w.get('type', '').replace(' ', '_').lower() == 'sticky_note']
            lines = []
            from bs4 import BeautifulSoup
            for w in note_widgets:
                raw = w.get('htmlText') or w.get('text') or ''
                cleaned = BeautifulSoup(raw, "html.parser").get_text(separator=" ").strip()
                if cleaned:
                    lines.append(cleaned)
            st.write(f"DEBUG: Pulled {len(lines)} sticky notes")
            return lines
        else:
            st.error(f"Failed to pull Mural sticky notes: {resp.status_code}")
            return []
    except requests.RequestException as e:
        st.error(f"Mural sticky notes error: {str(e)}")
        return []

# Check for auth code
try:
    auth_code = st.query_params.get("code", [None])[0] if "code" in st.query_params else None
    st.write(f"DEBUG: Checked query params, auth_code: {'present' if auth_code else 'none'}")
except AttributeError:
    qs = st.experimental_get_query_params()
    auth_code_list = qs.get("code", [])
    auth_code = auth_code_list[0] if isinstance(auth_code_list, list) and auth_code_list else None
    st.write("DEBUG: Fallback to experimental_get_query_params")

if auth_code and not st.session_state.access_token:
    st.write(f"DEBUG: Processing auth_code: {auth_code[:10]}... (truncated)")
    if not all([MURAL_CLIENT_ID, MURAL_CLIENT_SECRET, MURAL_REDIRECT_URI]):
        st.error("Missing Mural credentials. Please check secrets.")
        st.stop()
    tok_data = exchange_code_for_token(auth_code)
    if tok_data:
        st.session_state.access_token = tok_data["access_token"]
        st.session_state.refresh_token = tok_data.get("refresh_token")
        st.session_state.token_expires_in = tok_data.get("expires_in", 900)
        st.session_state.token_timestamp = datetime.now().timestamp()
        try:
            st.query_params.clear()
            st.write("DEBUG: Query params cleared")
        except AttributeError:
            st.experimental_set_query_params()
            st.write("DEBUG: Fallback to experimental_set_query_params")
        st.success("Authenticated with Mural!")
        st.stop()
    else:
        st.write("DEBUG: Token exchange failed, continuing to render UI")

# Refresh token if expired
if st.session_state.access_token:
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        st.write("DEBUG: Access token expired, attempting refresh")
        if not st.session_state.refresh_token:
            st.error("No refresh token available. Please reauthorize with Mural.")
            st.session_state.access_token = None
            st.stop()
        refreshed = refresh_access_token(st.session_state.refresh_token)
        if refreshed:
            st.session_state.access_token = refreshed["access_token"]
            st.session_state.refresh_token = refreshed.get("refresh_token", st.session_state.refresh_token)
            st.session_state.token_expires_in = refreshed.get("expires_in", 900)
            st.session_state.token_timestamp = datetime.now().timestamp()
            st.success("Refreshed Mural token!")
        else:
            st.error("Failed to refresh Mural token. Please reauthorize.")
            st.session_state.access_token = None
st.write("DEBUG: OAuth logic completed")

# Debug button to clear session state
if st.sidebar.button("Clear Session State (Debug)"):
    for key in st.session_state:
        del st.session_state[key]
    st.success("Session state cleared. Please reauthorize.")
    st.write("DEBUG: Session state cleared")

###############################################################################
# 4) Sidebar Mural Actions
###############################################################################
st.sidebar.header("Mural Integration")
st.write("DEBUG: Rendering sidebar")
mural_id_input = st.sidebar.text_input("Mural Board ID", value=MURAL_BOARD_ID)

if st.sidebar.button("Pull Sticky Notes"):
    st.write("DEBUG: Pull Sticky Notes button clicked")
    if st.session_state.access_token:
        lines = pull_mural_stickies(st.session_state.access_token, mural_id_input)
        if lines:
            st.success(f"Pulled {len(lines)} lines from Mural.")
            st.session_state["mural_lines"] = lines
    else:
        st.warning("No Mural token found. Please authorize first.")

auth_url = get_authorization_url()
st.sidebar.markdown(f"[Authorize Mural]({auth_url}) if needed.")
st.write("DEBUG: Sidebar rendered")

###############################################################################
# 5) Collect Human-Finalized Risks
###############################################################################
st.subheader("Paste or Use Mural-Fetched Risks")
st.write("DEBUG: Rendering risks input")
default_text = ""
if "mural_lines" in st.session_state:
    default_text = "\n".join(st.session_state["mural_lines"])
user_input = st.text_area("Human-provided (final) Risks from Mural or manual input:", value=default_text, height=150)
st.write("DEBUG: Risks input rendered")

###############################################################################
# RAG & Synergy Coverage
###############################################################################
def assign_rag_score(cscore):
    if cscore >= 13: return "Red"
    elif cscore >= 9: return "Amber"
    else: return "Green"

def color_rag(val):
    if val == "Red": return "background-color: #f9cccc"
    elif val == "Amber": return "background-color: #fcebcf"
    elif val == "Green": return "background-color: #ccf2d0"
    return ""

def style_rag_col(df):
    if "rag" in df.columns:
        return df.style.apply(
            lambda col: [color_rag(v) for v in col] if col.name == "rag" else ["" for _ in col],
            axis=0
        )
    return df

###############################################################################
# 6) Main Synergy Coverage + RAG + Brainstorming + Mitigation
###############################################################################
st.subheader("Load CSV from Method 1 & Perform Synergy Coverage")
st.write("DEBUG: Rendering CSV input")
csv_path = st.text_input("Path to CSV from Method 1", value="clean_risks.csv")
if st.button("Analyze CSV"):
    st.write("DEBUG: Analyze CSV button clicked")
    try:
        df = pd.read_csv(csv_path)
        st.success(f"Loaded {df.shape[0]} lines from {csv_path}.")

        # Basic columns check
        needed_cols = ["risk_id", "risk_description", "risk_type", "stakeholder", "severity", "probability", "combined_score"]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
            st.stop()

        # Assign RAG
        df["rag"] = df["combined_score"].apply(assign_rag_score)

        st.markdown("### RAG-Enhanced Risk Table")
        rag_styled = style_rag_col(df)
        st.dataframe(rag_styled, use_container_width=True)
        csv_with_rag = df.to_csv(index=False)
        st.download_button("Download CSV w/ RAG", data=csv_with_rag, file_name="clean_risks_with_rag.csv")

        # Synergy coverage: stakeholder × risk_type
        synergy_cov = df.groupby(["stakeholder", "risk_type"]).size().reset_index(name="count")
        st.markdown("### Synergy Coverage: (stakeholder × risk_type) Count")
        st.dataframe(synergy_cov.head(30))

        st.markdown("#### Heatmap: coverage count")
        chart_cov = alt.Chart(synergy_cov).mark_rect().encode(
            x=alt.X("risk_type:N"),
            y=alt.Y("stakeholder:N"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["stakeholder", "risk_type", "count"]
        ).properties(width=500, height=300)
        st.altair_chart(chart_cov, use_container_width=True)

        # Synergy gap: combos with <2 lines
        synergy_gaps = synergy_cov[synergy_cov["count"] < 2]
        if synergy_gaps.empty:
            st.info("No synergy combos with <2 lines. Good coverage!")
        else:
            st.warning("Some synergy combos appear under-covered (fewer than 2 lines).")
            st.dataframe(synergy_gaps)

        # BRAINSTORM
        st.subheader("Brainstorm Additional Risks for Coverage Gaps")
        focus_stkh = st.text_input("Focus stakeholder (optional)", value="")
        focus_type = st.text_input("Focus risk_type (optional)", value="")
        num_sugg = st.slider("Number of suggestions", 1, 10, 5)

        if st.button("Brainstorm Missing Risks"):
            synergy_str = "\n".join(
                f"- {row['stakeholder']} + {row['risk_type']}, coverage={row['count']}"
                for _, row in synergy_gaps.iterrows()
            )
            domain = df['domain'].iloc[0] if 'domain' in df.columns else "the AI system"
            prompt_b = f"""
You are an AI risk brainstorming assistant for {domain}.
We found synergy combos with low coverage:
{synergy_str}

Focus (if relevant) on stakeholder='{focus_stkh}' and risk_type='{focus_type}'.
Propose {num_sugg} new or overlooked AI risks. 
Each suggestion: bullet line with short rationale.
"""
            try:
                resp = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a synergy coverage brainstorming assistant."},
                        {"role": "user", "content": prompt_b}
                    ],
                    max_tokens=600,
                    temperature=0.8
                )
                st.markdown("#### Brainstormed Risks:")
                st.write(resp.choices[0].message.content.strip())
            except Exception as e:
                st.error(f"GPT error (brainstorm): {str(e)}")

        # MITIGATION
        st.subheader("Suggest Mitigation Strategies")
        pick_risk = st.selectbox("Pick a risk from the CSV", ["(none)"] + df["risk_description"].head(30).tolist())
        if st.button("Generate Mitigation"):
            if pick_risk and pick_risk != "(none)":
                domain = df["domain"].iloc[0] if "domain" in df.columns else "the AI domain"
                prompt_m = f"""
You are an AI mitigation expert for {domain}.
We have this risk:

'{pick_risk}'

Propose 2-3 human-centric mitigation strategies (each as bullet line) with short rationale.
"""
                try:
                    resp = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful AI risk mitigation advisor."},
                            {"role": "user", "content": prompt_m}
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    st.markdown("#### Mitigation Strategies:")
                    st.write(resp.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"GPT error (mitigation): {str(e)}")
            else:
                st.warning("No risk selected for mitigation.")

    except FileNotFoundError:
        st.error(f"File not found: {csv_path}")
    except Exception as e:
        st.error(f"Error loading CSV or synergy coverage: {str(e)}")
st.write("DEBUG: CSV section rendered")

###############################################################################
# 7) Optional: Semantic Coverage w/ Embeddings + FAISS
###############################################################################
st.subheader("Optional: Semantic Coverage with Embeddings + FAISS")
st.write("DEBUG: Rendering semantic coverage")
embed_path = st.text_input("Embeddings .npy", "embeddings.npy")
faiss_path = st.text_input("FAISS index .faiss", "faiss_index.faiss")
lines_for_sem = st.text_area("Lines to check coverage semantically", height=100)

if st.button("Check Semantic Coverage"):
    st.write("DEBUG: Check Semantic Coverage button clicked")
    user_lines = [l.strip() for l in lines_for_sem.split("\n") if l.strip()]
    if not user_lines:
        st.warning("No lines to check.")
    else:
        try:
            df2 = pd.read_csv(csv_path)
            main_embeds = np.load(embed_path)
            idx = faiss.read_index(faiss_path)

            # Embed user lines
            model_name = "all-MiniLM-L6-v2"
            embedder = SentenceTransformer(model_name)
            user_vecs = embedder.encode(user_lines, show_progress_bar=False).astype("float32")

            k = 3
            D, I = idx.search(user_vecs, k)
            st.markdown("### Semantic Coverage Matches:")
            for i, uln in enumerate(user_lines):
                st.write(f"**Your line**: {uln}")
                for rank, (nid, dist_) in enumerate(zip(I[i], D[i]), start=1):
                    row_ = df2.iloc[nid]
                    st.write(f"{rank}) {row_['risk_description']} (dist={dist_:.3f}, stkh={row_.get('stakeholder', '')}, type={row_.get('risk_type', '')})")
                st.write("---")
        except Exception as e:
            st.error(f"Semantic coverage error: {str(e)}")
st.write("DEBUG: Semantic coverage section rendered")

st.info("All done! This script uses st.query_params + st.stop for Mural auth.")
st.write("DEBUG: Script completed")
