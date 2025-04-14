###############################################################################
# method2.py - Comprehensive: Mural Pull, Advanced Synergy Coverage, Brainstorming, Mitigation
###############################################################################
import os
import json
import uuid
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import openai
from collections import defaultdict
from datetime import datetime
import re
import faiss

from sentence_transformers import SentenceTransformer

###############################################################################
# Streamlit Page Config
###############################################################################
st.set_page_config(page_title="Method 2 - Comprehensive Tool", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2)")

###############################################################################
# 1) Load Secrets (Mural, OpenAI)
###############################################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "myworkspace")
except KeyError as e:
    st.warning(f"Missing secret: {e}. If you're not using Mural, ignore. Otherwise set secrets.")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")

openai.api_key = OPENAI_API_KEY

###############################################################################
# 2) Mural OAuth & Pull
###############################################################################
if "access_token" not in st.session_state:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.token_expires_in = None
    st.session_state.token_timestamp = None

def get_authorization_url():
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "state": str(uuid.uuid4()),
        "response_type": "code"
    }
    from urllib.parse import urlencode
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
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Mural Auth failed: {resp.status_code}")
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
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Mural refresh failed: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Mural refresh error: {str(e)}")
        return None

def verify_mural(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    return (resp.status_code == 200)

def pull_mural_stickies(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {"Accept":"application/json", "Authorization":f"Bearer {auth_token}"}
    session = requests.Session()
    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
    resp = session.get(url, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        widgets = data.get("value", data.get("data", []))
        note_widgets = [w for w in widgets if w.get('type','').replace(' ','_').lower() == 'sticky_note']
        lines = []
        for w in note_widgets:
            raw = w.get('htmlText') or w.get('text') or ''
            from bs4 import BeautifulSoup
            cleaned = BeautifulSoup(raw, "html.parser").get_text(separator=" ").strip()
            if cleaned:
                lines.append(cleaned)
        return lines
    else:
        st.error(f"Failed to pull Mural stickies: {resp.status_code}")
        return []

query_params = st.query_params
auth_code = query_params.get("code", None)
if auth_code and not st.session_state.access_token:
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state.access_token = token_data["access_token"]
        st.session_state.refresh_token = token_data.get("refresh_token")
        st.session_state.token_expires_in = token_data.get("expires_in", 900)
        from datetime import datetime
        st.session_state.token_timestamp = datetime.now().timestamp()
        st.success("Authenticated with Mural!")
        st.experimental_rerun()

if st.session_state.access_token:
    # Check if token needs refreshing
    import time
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        # refresh
        new_tk = refresh_access_token(st.session_state.refresh_token)
        if new_tk:
            st.session_state.access_token = new_tk["access_token"]
            st.session_state.refresh_token = new_tk.get("refresh_token", st.session_state.refresh_token)
            st.session_state.token_expires_in = new_tk.get("expires_in",900)
            st.session_state.token_timestamp = datetime.now().timestamp()
else:
    st.markdown("No Mural token found. Please [authorize]({}) if needed.".format(get_authorization_url()))

###############################################################################
# 3) Sidebar Mural Actions
###############################################################################
st.sidebar.header("Mural Integration")
mural_id_input = st.sidebar.text_input("Mural Board ID", MURAL_BOARD_ID)
if st.sidebar.button("Pull Sticky Notes"):
    if st.session_state.access_token:
        lines = pull_mural_stickies(st.session_state.access_token, mural_id_input)
        if lines:
            st.success(f"Pulled {len(lines)} lines from Mural.")
            st.session_state["mural_lines"] = lines
    else:
        st.warning("No Mural access token. Please authorize first.")

###############################################################################
# 4) Collect Human-Finalised Risks
###############################################################################
st.subheader("Paste or Use Mural-Fetched Risks")
default_text = ""
if "mural_lines" in st.session_state:
    default_text = "\n".join(st.session_state["mural_lines"])

user_input = st.text_area("Human-provided Risks", value=default_text, height=150)

###############################################################################
# RAG & synergy coverage
###############################################################################
def assign_rag(comb_score):
    if comb_score >= 13:
        return "Red"
    elif comb_score >= 9:
        return "Amber"
    else:
        return "Green"

def color_rag(r):
    if r=="Red":
        return "background-color: #f5cccc"
    elif r=="Amber":
        return "background-color: #fcebcf"
    elif r=="Green":
        return "background-color: #ccf2d0"
    return ""

def style_rag_df(df):
    if "rag" in df.columns:
        return df.style.apply(lambda col: [color_rag(v) for v in col] if col.name=="rag" else ["" for _ in col], axis=0)
    return df

###############################################################################
# 5) Main synergy coverage & method
###############################################################################
st.subheader("Load CSV from Method 1")

csv_path = st.text_input("CSV Path from Method 1", "clean_risks.csv")
if st.button("Run Analysis"):
    try:
        df = pd.read_csv(csv_path)
        st.success(f"Loaded {df.shape[0]} lines from {csv_path}.")

        # Check columns
        needed = ["risk_id","risk_description","risk_type","stakeholder","severity","probability","combined_score"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}. Please check your CSV structure.")
            st.stop()

        # Assign RAG
        df["rag"] = df["combined_score"].apply(assign_rag)

        st.markdown("### RAG-Enhanced Risk Table")
        styled = style_rag_df(df)
        st.dataframe(styled, use_container_width=True)
        csv_updated = df.to_csv(index=False)
        st.download_button("Download CSV (RAG)", csv_updated, file_name="clean_risks_with_rag.csv")

        # Pivot synergy coverage
        synergy_cov = df.groupby(["stakeholder","risk_type"]).size().reset_index(name="count")
        st.markdown("### Synergy Coverage (stakeholder × risk_type)")
        st.dataframe(synergy_cov.head(30))

        st.markdown("#### Heatmap: coverage count")
        cov_chart = alt.Chart(synergy_cov).mark_rect().encode(
            x=alt.X("risk_type:N"),
            y=alt.Y("stakeholder:N"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["stakeholder","risk_type","count"]
        ).properties(width=500, height=300)
        st.altair_chart(cov_chart, use_container_width=True)

        # Gap detection: combos with <2 lines
        synergy_gaps = synergy_cov[synergy_cov["count"]<2].copy()
        if synergy_gaps.empty:
            st.success("No synergy combos with <2 coverage lines. Good coverage!")
        else:
            st.warning("Some synergy combos appear under-covered (<2 lines).")
            st.dataframe(synergy_gaps)

        # Brainstorm function
        st.subheader("Brainstorm Additional Risks for Coverage Gaps")
        focus_stkh = st.text_input("Focus stakeholder (optional)", value="")
        focus_rtype = st.text_input("Focus risk_type (optional)", value="")
        n_suggestions = st.slider("Number of Suggestions", 1, 10, 5)

        if st.button("Brainstorm Risks"):
            domain = df["domain"].iloc[0] if "domain" in df.columns else "the AI system"
            synergy_info = synergy_gaps if not synergy_gaps.empty else synergy_cov.head(3)  # fallback
            synergy_str = "\n".join(
                f"- {row['stakeholder']} & {row['risk_type']} have coverage={row['count']}"
                for _,row in synergy_info.iterrows()
            )
            # Build prompt
            prompt_b = f"""
You are an AI risk brainstorming assistant for {domain}.
We identified synergy combos that may need more coverage:
{synergy_str}

Please propose {n_suggestions} new or overlooked AI risks relevant to these combos.
Focus on stakeholder='{focus_stkh}' and risk_type='{focus_rtype}' if specified (otherwise be general).
Return each suggestion as a bullet line with a short rationale.
"""
            try:
                resp = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"You are a creative synergy coverage assistant."},
                              {"role":"user","content":prompt_b}],
                    max_tokens=600,
                    temperature=0.8
                )
                suggestions = resp.choices[0].message.content
                st.markdown("#### Brainstormed Risk Suggestions:")
                st.write(suggestions)
            except Exception as e:
                st.error(f"Error calling GPT for brainstorming: {str(e)}")

        # Mitigation function
        st.subheader("Suggest Mitigation Strategies")
        st.write("Enter a risk line or pick from the CSV to see mitigation strategies from GPT.")
        risk_for_mitig = st.selectbox("Pick a risk from CSV", ["(none)"] + df["risk_description"].head(50).tolist())
        if st.button("Generate Mitigation"):
            if risk_for_mitig and risk_for_mitig!="(none)":
                domain = df["domain"].iloc[0] if "domain" in df.columns else "the AI system"
                prompt_m = f"""
You are an AI risk mitigation advisor for {domain}.
The following risk is identified:

'{risk_for_mitig}'

Suggest 2-3 human-centric mitigation strategies for this risk. 
Format them as bullet lines with a short rationale each.
"""
                try:
                    resp = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":"You are a helpful AI mitigation expert."},
                                  {"role":"user","content":prompt_m}],
                        max_tokens=500,
                        temperature=0.7
                    )
                    mit_text = resp.choices[0].message.content
                    st.markdown("#### Mitigation Strategies:")
                    st.write(mit_text)
                except Exception as e:
                    st.error(f"GPT call error (mitigation): {str(e)}")
            else:
                st.warning("No risk selected for mitigation.")


    except FileNotFoundError:
        st.error(f"File not found: {csv_path}")
    except Exception as e:
        st.error(f"Error processing CSV or synergy coverage: {str(e)}")

###############################################################################
# 6) Optional: Semantic Coverage (Embeddings)
###############################################################################
st.subheader("Optional: Semantic Coverage Check with Embeddings + FAISS")
emb_file = st.text_input("Embeddings .npy from Method 1", "embeddings.npy")
faiss_file = st.text_input("FAISS index from Method 1", "faiss_index.faiss")
semantic_input = st.text_area("Risk lines to check coverage:", height=120)

if st.button("Check Coverage (Semantic)"):
    lines = [l.strip() for l in semantic_input.split("\n") if l.strip()]
    if not lines:
        st.warning("No lines to check.")
    else:
        try:
            df2 = pd.read_csv(csv_path)
            main_embeds = np.load(emb_file)
            index = faiss.read_index(faiss_file)

            embed_model = "all-MiniLM-L6-v2"
            embedder = SentenceTransformer(embed_model)
            user_vecs = embedder.encode(lines, show_progress_bar=False).astype("float32")

            k=3
            D, I = index.search(user_vecs, k)
            st.markdown("### Semantic Coverage Matches")
            for i, userln in enumerate(lines):
                st.markdown(f"**Your line**: {userln}")
                for rank, (idxN, distN) in enumerate(zip(I[i], D[i]), start=1):
                    row_ = df2.iloc[idxN]
                    st.write(f"Match {rank}: {row_['risk_description']} (distance={distN:.3f}, type={row_.get('risk_type','N/A')}, stakeholder={row_.get('stakeholder','N/A')})")
                st.write("---")

        except Exception as e:
            st.error(f"Error in semantic coverage check: {str(e)}")

st.info("Method 2.")
