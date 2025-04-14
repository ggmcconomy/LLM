# method2.py - Fully Restored and Updated Method 2 (GPT + Synergy + Mitigation + Semantic)

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

st.set_page_config(page_title="Method 2 - Full Risk Coverage Toolkit", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2) ✅ Full Functionality")

# Load secrets or env
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
MURAL_CLIENT_ID = st.secrets.get("MURAL_CLIENT_ID", "")
MURAL_CLIENT_SECRET = st.secrets.get("MURAL_CLIENT_SECRET", "")
MURAL_BOARD_ID = st.secrets.get("MURAL_BOARD_ID", "")
MURAL_REDIRECT_URI = st.secrets.get("MURAL_REDIRECT_URI", "")
openai.api_key = OPENAI_API_KEY

# Session
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "auth_complete" not in st.session_state:
    st.session_state.auth_complete = False

# OAuth

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
    r = requests.post("https://app.mural.co/api/public/v1/authorization/oauth2/token", data={
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "redirect_uri": MURAL_REDIRECT_URI,
        "code": code,
        "grant_type": "authorization_code"
    })
    return r.json() if r.status_code == 200 else None

def pull_mural_stickies(token, mural_id):
    r = requests.get(f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets", headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 200: return []
    widgets = r.json().get("value", [])
    return [BeautifulSoup(w.get("htmlText") or w.get("text") or "", "html.parser").get_text(separator=" ").strip() for w in widgets if w.get("type", "").lower() == "sticky_note"]

qs = st.query_params
auth_code = qs.get("code")
if isinstance(auth_code, list): auth_code = auth_code[0]
if auth_code and not st.session_state.access_token and not st.session_state.auth_complete:
    tok = exchange_code_for_token(auth_code)
    if tok:
        st.session_state.access_token = tok.get("access_token")
        st.session_state.auth_complete = True
        st.success("✅ Authenticated with Mural!")
        st.query_params.clear()
        st.experimental_rerun()

# Sidebar - Mural
st.sidebar.markdown(f"[Authorize with Mural]({get_authorization_url()})")
mural_id = st.sidebar.text_input("Mural Board ID", value=MURAL_BOARD_ID)
if st.sidebar.button("Pull Sticky Notes") and st.session_state.access_token:
    notes = pull_mural_stickies(st.session_state.access_token, mural_id)
    if notes:
        st.session_state["mural_notes"] = notes
        st.success(f"Pulled {len(notes)} notes from Mural.")

# Step 1: Human Final Risks
st.subheader("1️⃣ Human-Finalized Risks")
def_text = "\n".join(st.session_state.get("mural_notes", []))
user_input = st.text_area("Finalized risks from Mural or manual: (one per line)", value=def_text, height=200)

# Step 2: Load CSV
st.subheader("2️⃣ Analyze Risk Landscape from Method 1")
csv_file = st.text_input("Method 1 CSV File", "clean_risks.csv")
if st.button("Analyze CSV"):
    try:
        df = pd.read_csv(csv_file)
        if "combined_score" not in df:
            st.error("Missing 'combined_score' column in CSV.")
            st.stop()

        df["rag"] = df["combined_score"].apply(lambda x: "Red" if x >= 13 else ("Amber" if x >= 9 else "Green"))
        st.dataframe(df.head(50))

        st.markdown("### 🔍 Synergy Coverage")
        synergy = df.groupby(["stakeholder", "risk_type"]).size().reset_index(name="count")
        st.dataframe(synergy)

        chart = alt.Chart(synergy).mark_rect().encode(
            x="risk_type:N", y="stakeholder:N", color="count:Q",
            tooltip=["stakeholder", "risk_type", "count"]
        ).properties(width=600, height=400)
        st.altair_chart(chart, use_container_width=True)

        st.subheader("3️⃣ GPT-Based Brainstorming for Gaps")
        st.write("Synergy combos with low coverage (less than 2 lines):")
        gaps = synergy[synergy["count"] < 2]
        st.dataframe(gaps)

        n_sugg = st.slider("Number of suggestions", 3, 10, 5)
        if st.button("💡 Brainstorm Suggestions"):
            prompt = f"""
Suggest {n_sugg} overlooked or underrepresented AI deployment risks.
Each risk should reflect gaps in stakeholder × risk_type synergy:
{gaps.to_string(index=False)}
Output 1 risk per bullet.
"""
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an AI risk discovery expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800
                )
                st.markdown("**Suggested New Risks:**")
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {str(e)}")

        st.subheader("4️⃣ Mitigation Strategies")
        pick = st.selectbox("Pick a risk to mitigate", df["risk_description"].head(30))
        if st.button("🛡️ Generate Mitigation"):
            prompt_m = f"""
You are an AI risk mitigation expert. Provide 2-3 strategies to mitigate:
"{pick}"
Include a short rationale per strategy.
"""
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You provide actionable AI mitigation advice."},
                        {"role": "user", "content": prompt_m}
                    ],
                    max_tokens=600
                )
                st.markdown("**Mitigation Suggestions:**")
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {str(e)}")

        st.subheader("5️⃣ Optional: Semantic Similarity Check")
        embed_file = st.text_input("Embeddings file", "embeddings.npy")
        faiss_file = st.text_input("FAISS index file", "faiss_index.faiss")
        lines_to_check = st.text_area("Paste risks to check", height=120)
        if st.button("🔍 Check Semantic Matches"):
            lines = [l.strip() for l in lines_to_check.split("\n") if l.strip()]
            try:
                index = faiss.read_index(faiss_file)
                embed = SentenceTransformer("all-MiniLM-L6-v2")
                vecs = embed.encode(lines).astype("float32")
                D, I = index.search(vecs, 3)
                for i, line in enumerate(lines):
                    st.markdown(f"**{line}**")
                    for j, idx in enumerate(I[i]):
                        match = df.iloc[idx]
                        st.write(f"{j+1}. {match['risk_description']} (dist={D[i][j]:.3f})")
            except Exception as e:
                st.error(f"Error running semantic similarity: {e}")

        st.download_button("📥 Download Enriched CSV", data=df.to_csv(index=False), file_name="method2_enriched.csv")

    except Exception as e:
        st.error(f"Error loading CSV: {e}")
