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
except KeyError as e:
    st.error(f"Missing secret: {e}. Please set Mural and OpenAI secrets properly.")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MURAL_CLIENT_ID = os.getenv("MURAL_CLIENT_ID", "")
    MURAL_CLIENT_SECRET = os.getenv("MURAL_CLIENT_SECRET", "")
    MURAL_REDIRECT_URI = os.getenv("MURAL_REDIRECT_URI", "")
    MURAL_BOARD_ID = os.getenv("MURAL_BOARD_ID", "")
    MURAL_WORKSPACE_ID = os.getenv("MURAL_WORKSPACE_ID", "myworkspace")

openai.api_key = OPENAI_API_KEY

# Normalize redirect URI to remove trailing slashes and ensure consistency
def normalize_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', '', ''))

MURAL_REDIRECT_URI = normalize_url(MURAL_REDIRECT_URI)
st.write(f"Using redirect_uri: {MURAL_REDIRECT_URI}")

###############################################################################
# 3) Mural OAuth & Auth
###############################################################################
if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.token_expires_in = None
    st.session_state.token_timestamp = None

def get_authorization_url():
    from urllib.parse import urlencode
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    return "https://app.mural.co/api/public/v1/authorization/oauth2/?" + urlencode(params)

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
            st.write(f"Attempting token exchange (try {attempt + 1}/{max_retries}) with redirect_uri: {MURAL_REDIRECT_URI}")
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
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
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
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
        return resp.status_code == 200
    except requests.RequestException as e:
        st.error(f"Mural verification error: {str(e)}")
        return False

def pull_mural_stickies(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {auth_token}"}
    try:
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
            return lines
        else:
            st.error(f"Failed to pull Mural sticky notes: {resp.status_code}")
            return []
    except requests.RequestException as e:
        st.error(f"Mural sticky notes error: {str(e)}")
        return []

# Check if we got ?code=... from Mural
try:
    auth_code = st.query_params.get("code", [None])[0] if "code" in st.query_params else None
except AttributeError:
    # Fallback for older Streamlit versions
    qs = st.experimental_get_query_params()
    auth_code_list = qs.get("code", [])
    auth_code = auth_code_list[0] if isinstance(auth_code_list, list) and auth_code_list else None

if auth_code and not st.session_state.access_token:
    # Validate inputs
    if not all([MURAL_CLIENT_ID, MURAL_CLIENT_SECRET, MURAL_REDIRECT_URI, auth_code]):
        st.error("Missing Mural credentials or authorization code. Please check secrets and reauthorize.")
        st.stop()
    # Log the code for debugging
    st.write(f"Received authorization code: {auth_code[:10]}... (truncated for security)")
    # Exchange code for token
    tok_data = exchange_code_for_token(auth_code)
    if tok_data:
        st.session_state.access_token = tok_data["access_token"]
        st.session_state.refresh_token = tok_data.get("refresh_token")
        st.session_state.token_expires_in = tok_data.get("expires_in", 900)
        st.session_state.token_timestamp = datetime.now().timestamp()
        # Clear query params
        try:
            st.query_params.clear()
        except AttributeError:
            st.experimental_set_query_params()
        st.success("Authenticated with Mural!")
        st.stop()

# Refresh token if expired
if st.session_state.access_token:
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        if not st.session_state.refresh_token:
            st.error("No refresh token available. Please reauthorize with Mural.")
            st.session_state.access
