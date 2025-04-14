# method2.py - Fixed version using st.query_params (2024+ safe)

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
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

st.set_page_config(page_title="Method 2 - AI Risk Coverage", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2) - Fixed for 2024")

# Load secrets or environment
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
MURAL_CLIENT_ID = st.secrets.get("MURAL_CLIENT_ID", "")
MURAL_CLIENT_SECRET = st.secrets.get("MURAL_CLIENT_SECRET", "")
MURAL_BOARD_ID = st.secrets.get("MURAL_BOARD_ID", "")
MURAL_REDIRECT_URI = st.secrets.get("MURAL_REDIRECT_URI", "")

openai.api_key = OPENAI_API_KEY

# Session state
if "access_token" not in st.session_state:
    st.session_state.access_token = None

# OAuth helpers
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

def exchange_code_for_token(code):
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "redirect_uri": MURAL_REDIRECT_URI,
        "code": code,
        "grant_type": "authorization_code"
    }
    r = requests.post(url, data=data)
    return r.json() if r.status_code == 200 else None

def pull_mural_stickies(token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return []
    widgets = r.json().get("value", [])
    notes = []
    for w in widgets:
        if w.get("type", "").lower() == "sticky_note":
            raw = w.get("htmlText") or w.get("text") or ""
            cleaned = BeautifulSoup(raw, "html.parser").get_text(separator=" ").strip()
            if cleaned:
                notes.append(cleaned)
    return notes

# Handle Mural OAuth callback
qs = st.query_params
auth_code = qs.get("code")
if isinstance(auth_code, list):
    auth_code = auth_code[0]

if auth_code and not st.session_state.access_token:
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state.access_token = token_data.get("access_token")
        st.success("Authenticated with Mural!")
        st.query_params.clear()
        st.stop()

# Sidebar: Mural controls
auth_url = get_authorization_url()
st.sidebar.markdown(f"[Authorize with Mural]({auth_url})")
mural_id = st.sidebar.text_input("Mural Board ID", value=MURAL_BOARD_ID)
if st.sidebar.button("Pull Sticky Notes") and st.session_state.access_token:
    notes = pull_mural_stickies(st.session_state.access_token, mural_id)
    if notes:
        st.session_state["mural_notes"] = notes
        st.success(f"Pulled {len(notes)} notes.")

# Show final risks
st.subheader("1. Human-Finalized Risks")
def_text = "\n".join(st.session_state.get("mural_notes", []))
user_input = st.text_area("Finalized risk lines:", value=def_text, height=180)

# Synergy analysis
st.subheader("2. Load CSV from Method 1 for Synergy Coverage")
csv_path = st.text_input("CSV File", "clean_risks.csv")
if st.button("Analyze CSV"):
    try:
        df = pd.read_csv(csv_path)
        if "combined_score" in df:
            df["rag"] = df["combined_score"].apply(lambda x: "Red" if x >= 13 else ("Amber" if x >= 9 else "Green"))
            st.dataframe(df.head(50))
            synergy = df.groupby(["stakeholder", "risk_type"]).size().reset_index(name="count")
            st.markdown("### Synergy Coverage")
            st.dataframe(synergy)
            st.altair_chart(
                alt.Chart(synergy).mark_rect().encode(
                    x="risk_type:N",
                    y="stakeholder:N",
                    color="count:Q",
                    tooltip=["stakeholder", "risk_type", "count"]
                ).properties(width=600, height=400),
                use_container_width=True
            )
        else:
            st.error("CSV must contain 'combined_score' column")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

# Done
st.info("Method 2 interface fully updated for 2024 Streamlit changes. Use 'st.query_params' instead of deprecated experimental methods.")
