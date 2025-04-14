###############################################################################
# method2_enhanced.py
#
# A full “Method 2” pipeline:
#   1) Mural OAuth (pull sticky notes)
#   2) CSV synergy coverage (with color-coded heatmap)
#   3) GPT-based brainstorming for synergy gaps
#   4) GPT-based mitigation for selected risk
#   5) (Optional) FAISS semantic coverage
#   6) RAG labeling in synergy coverage
#   7) Download enriched CSV
###############################################################################

import os
import sys
import re
import json
import uuid
import time
import requests
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime
from sentence_transformers import SentenceTransformer
from collections import Counter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlencode

###############################################################################
# 1) Streamlit Page Config
###############################################################################
st.set_page_config(page_title="Method 2 Enhanced", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2) – Enhanced")

###############################################################################
# 2) Load/OpenAI Setup
###############################################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

import openai
openai.api_key = OPENAI_API_KEY

###############################################################################
# 3) Mural OAuth Setup
###############################################################################
try:
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
except KeyError:
    MURAL_CLIENT_ID = os.getenv("MURAL_CLIENT_ID", "")
    MURAL_CLIENT_SECRET = os.getenv("MURAL_CLIENT_SECRET", "")
    MURAL_BOARD_ID = os.getenv("MURAL_BOARD_ID", "")
    MURAL_REDIRECT_URI = os.getenv("MURAL_REDIRECT_URI", "")

if "mural_access_token" not in st.session_state:
    st.session_state.mural_access_token = None
if "mural_auth_complete" not in st.session_state:
    st.session_state.mural_auth_complete = False

def get_mural_auth_url():
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    return "https://app.mural.co/api/public/v1/authorization/oauth2/?" + urlencode(params)

def exchange_code_for_token_mural(code):
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

def pull_mural_stickies(token, board_id):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"https://app.mural.co/api/public/v1/murals/{board_id}/widgets", headers=headers)
    if r.status_code!=200:
        st.error(f"Failed to pull mural data: {r.status_code}")
        return []
    data = r.json()
    widgets = data.get("value", [])
    lines = []
    for w in widgets:
        if w.get("type", "").lower()=="sticky_note":
            raw = w.get("htmlText") or w.get("text") or ""
            cleaned = BeautifulSoup(raw, "html.parser").get_text(separator=" ").strip()
            if cleaned:
                lines.append(cleaned)
    return lines

qs = st.query_params
auth_code = qs.get("code")
if isinstance(auth_code, list):
    auth_code = auth_code[0]

if auth_code and not st.session_state.mural_access_token and not st.session_state.mural_auth_complete:
    token_data = exchange_code_for_token_mural(auth_code)
    if token_data:
        st.session_state.mural_access_token = token_data.get("access_token")
        st.session_state.mural_auth_complete = True
        st.success("Authenticated with Mural!")
        st.query_params.clear()
        st.experimental_rerun()

st.sidebar.markdown(f"[Authorize with Mural]({get_mural_auth_url()})")
mural_board = st.sidebar.text_input("Mural Board ID", MURAL_BOARD_ID)

if st.sidebar.button("Pull Sticky Notes") and st.session_state.mural_access_token:
    lines_ = pull_mural_stickies(st.session_state.mural_access_token, mural_board)
    if lines_:
        st.session_state["mural_notes"] = lines_
        st.success(f"Pulled {len(lines_)} notes from Mural board={mural_board}.")

###############################################################################
# 4) Section 1 - Human Final Risks
###############################################################################
st.subheader("1️⃣ Human-Finalized Risks")
default_text = "\n".join(st.session_state.get("mural_notes", []))
user_risks_input = st.text_area("Finalized Mural risk lines:", default_text, height=180)

###############################################################################
# 5) Load CSV from Method 1 & Show RAG + synergy coverage
###############################################################################
st.subheader("2️⃣ Analyze Risk Landscape from Method 1")
csv_file = st.text_input("Method 1 CSV File", "clean_risks.csv")

def synergy_color_scale(count):
    """
    Example synergy threshold-based scale:
    0 => #FFAAAA (red)
    1 => #FFC080 (orange)
    2 => #FFE380 (yellow)
    3 => #CCFFB2 (light green)
    4+ => #70DB70 (green)
    """
    if count<=0: return "#FFAAAA"
    elif count==1: return "#FFC080"
    elif count==2: return "#FFE380"
    elif count==3: return "#CCFFB2"
    else: return "#70DB70"

if st.button("Analyze CSV"):
    try:
        df = pd.read_csv(csv_file)
        if "combined_score" not in df.columns:
            st.error("CSV missing 'combined_score' column!")
            st.stop()

        # RAG
        df["rag"] = df["combined_score"].apply(lambda s: "Red" if s>=13 else ("Amber" if s>=9 else "Green"))
        st.markdown("### First 30 lines from CSV")
        st.dataframe(df.head(30))

        # Synergy coverage
        synergy = df.groupby(["stakeholder","risk_type"]).size().reset_index(name="count")
        st.markdown("### Synergy Coverage - color-coded altair heatmap")

        # Build color scale using a transform
        synergy["color"] = synergy["count"].apply(synergy_color_scale)

        # We'll produce an altair chart with a color encoding from the synergy table
        chart = alt.Chart(synergy).mark_rect().encode(
            x=alt.X("risk_type:N", sort=alt.EncodingSortField("risk_type", order="ascending")),
            y=alt.Y("stakeholder:N", sort=alt.EncodingSortField("stakeholder", order="ascending")),
            tooltip=[alt.Tooltip("stakeholder"), alt.Tooltip("risk_type"), alt.Tooltip("count")],
            color=alt.Color("count:Q", scale=None)  # We'll override with color
        ).properties(width=600, height=400).configure_mark(color="color")
        # This approach doesn't easily let us pass synergy['color'] directly to Altair. We'll do a workaround:
        # We'll pass the color in a separate encoding channel or use a trick with 'detail'.
        # Simpler approach: We'll build a derived column for the numeric scale and let altair do it. Let's do altair scale:
        # We'll define altair scale domain and range manually:

        # We'll define a domain [0,1,2,3,4,10] => range [red, orange, yellow, lightgreen, green, green]
        coverage_scale = alt.Scale(
            domain=[0,1,2,3,4,100],
            range=["#FFAAAA","#FFC080","#FFE380","#CCFFB2","#70DB70","#70DB70"]
        )

        synergy_chart = alt.Chart(synergy).mark_rect().encode(
            x=alt.X("risk_type:N", sort=alt.SortField("risk_type", order="ascending")),
            y=alt.Y("stakeholder:N", sort=alt.SortField("stakeholder", order="ascending")),
            color=alt.Color("count:Q", scale=coverage_scale),
            tooltip=["stakeholder","risk_type","count"]
        ).properties(width=600, height=400)

        st.altair_chart(synergy_chart, use_container_width=True)

        st.markdown("#### Combos with coverage <2 lines:")
        synergy_gaps = synergy[synergy["count"]<2]
        st.dataframe(synergy_gaps)

        st.subheader("3️⃣ GPT-Based Brainstorming for Gaps")
        sugg_num = st.slider("Number of suggestions", 3, 10, 5)
        if st.button("Brainstorm Additional Risks"):
            prompt_brain = f\"\"\"\nYou are an AI risk analysis expert. We have synergy coverage combos under 2 lines:\n{synergy_gaps.to_string(index=False)}\nProvide {sugg_num} new or overlooked AI deployment risks, referencing these combos.\nFormat each risk as a bullet.\n\"\"\"  # can refine
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role":"system","content":"You propose synergy-based AI risks."},
                        {"role":"user","content":prompt_brain}
                    ],
                    max_tokens=700
                )
                st.markdown("**Brainstormed Risks**:")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {str(e)}")

        st.subheader("4️⃣ GPT-Based Mitigation for a Selected Risk")
        pick_risk = st.selectbox("Select a CSV risk to mitigate", df["risk_description"].head(40))
        if st.button("Suggest Mitigation"):
            prompt_m = f\"\"\"\nYou are an AI risk mitigation specialist. Provide 2-3 mitigations for this risk:\n\"{pick_risk}\"\nInclude short rationale each.\n\"\"\" 
            try:
                resp = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role":"system","content":"You provide actionable mitigation steps."},
                        {"role":"user","content":prompt_m}
                    ],
                    max_tokens=700
                )
                st.markdown("**Mitigation Strategies:**")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {str(e)}")

        st.subheader("5️⃣ Optional: Semantic Similarity Search")
        embed_file = st.text_input("Embeddings file path", "embeddings.npy")
        faiss_file = st.text_input("FAISS index path", "faiss_index.faiss")
        lines_to_check = st.text_area("Paste lines to check semantically", height=100)
        if st.button("Check Semantic Coverage"):
            lines_parsed = [l.strip() for l in lines_to_check.split("\\n") if l.strip()]
            try:
                index = faiss.read_index(faiss_file)
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                user_emb = embedder.encode(lines_parsed).astype("float32")
                D,I = index.search(user_emb, 3)
                for i, line in enumerate(lines_parsed):
                    st.markdown(f"**Line**: {line}")
                    for rank, idx_ in enumerate(I[i]):
                        row_ = df.iloc[idx_]
                        st.write(f"{rank+1}. {row_['risk_description']} (dist={D[i][rank]:.3f})")
                    st.write("---")
            except Exception as e:
                st.error(f"Semantic search error: {str(e)}")

        st.download_button(
            label="Download Enriched CSV",
            data=df.to_csv(index=False),
            file_name="method2_enriched.csv"
        )

    except Exception as e:
        st.error(f"Error analyzing CSV: {str(e)}")

###############################################################################
# 6) Done
###############################################################################
st.info("Method 2 Enhanced loaded. Use the steps above to see synergy coverage, brainstorm new risks, and generate mitigations!")
