###############################################################################
# method2.py - Comprehensive: Mural Pull, Advanced Synergy Coverage, Brainstorming, Mitigation
# No st.experimental_rerun() (uses st.experimental_set_query_params + st.stop)
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
import faiss
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer

###############################################################################
# 1) Page Config & Title
###############################################################################
st.set_page_config(page_title="Method 2 - Comprehensive Tool", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2) - No rerun Error")

###############################################################################
# 2) Load Secrets (Mural, OpenAI)
###############################################################################
# If you're not using secrets, fallback to environment variables.
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "myworkspace")
except KeyError as e:
    st.warning(f"Missing secret: {e}. If not using Mural, ignore. Otherwise set secrets properly.")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")

openai.api_key = OPENAI_API_KEY

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
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Mural refresh failed: {r.status_code}")
            return None
    except Exception as e:
        st.error(f"Mural refresh error: {str(e)}")
        return None

def verify_mural(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}"
    headers = {"Accept":"application/json", "Authorization":f"Bearer {auth_token}"}
    r = requests.get(url, headers=headers, timeout=10)
    return (r.status_code == 200)

def pull_mural_stickies(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {"Accept":"application/json", "Authorization":f"Bearer {auth_token}"}
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        widgets = data.get("value", data.get("data", []))
        note_widgets = [w for w in widgets if w.get('type','').replace(' ','_').lower() == 'sticky_note']
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

# Check if we got ?code=... from Mural
qs = st.experimental_get_query_params()
auth_code_list = qs.get("code", [])
auth_code = auth_code_list[0] if isinstance(auth_code_list, list) and auth_code_list else None

if auth_code and not st.session_state.access_token:
    # Exchange code for token
    tok_data = exchange_code_for_token(auth_code)
    if tok_data:
        st.session_state.access_token = tok_data["access_token"]
        st.session_state.refresh_token = tok_data.get("refresh_token")
        st.session_state.token_expires_in = tok_data.get("expires_in", 900)
        st.session_state.token_timestamp = datetime.now().timestamp()

        # Clear ?code=... from the URL
        st.experimental_set_query_params()
        st.success("Authenticated with Mural!")
        # Halt now so the user sees success. Next run, we have tokens in session.
        st.stop()

# If we do have a token, maybe refresh it if expired
if st.session_state.access_token:
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        refreshed = refresh_access_token(st.session_state.refresh_token)
        if refreshed:
            st.session_state.access_token = refreshed["access_token"]
            st.session_state.refresh_token = refreshed.get("refresh_token","")
            st.session_state.token_expires_in = refreshed.get("expires_in",900)
            st.session_state.token_timestamp = datetime.now().timestamp()
            st.success("Refreshed Mural token!")

###############################################################################
# 4) Sidebar Mural Actions
###############################################################################
st.sidebar.header("Mural Integration")
mural_id_input = st.sidebar.text_input("Mural Board ID", value=MURAL_BOARD_ID)

if st.sidebar.button("Pull Sticky Notes"):
    if st.session_state.access_token:
        lines = pull_mural_stickies(st.session_state.access_token, mural_id_input)
        if lines:
            st.success(f"Pulled {len(lines)} lines from Mural.")
            st.session_state["mural_lines"] = lines
    else:
        st.warning("No Mural token found. Please authorize first (use the link above).")

auth_url = get_authorization_url()
st.sidebar.markdown(f"[Authorize Mural]({auth_url}) if needed.")

###############################################################################
# 5) Collect Human-Finalized Risks
###############################################################################
st.subheader("Paste or Use Mural-Fetched Risks")
default_text = ""
if "mural_lines" in st.session_state:
    default_text = "\n".join(st.session_state["mural_lines"])
user_input = st.text_area("Human-provided (final) Risks from Mural or manual input:", value=default_text, height=150)

###############################################################################
# RAG & synergy coverage
###############################################################################
def assign_rag_score(cscore):
    if cscore >= 13: return "Red"
    elif cscore >= 9: return "Amber"
    else: return "Green"

def color_rag(val):
    if val=="Red": return "background-color: #f9cccc"
    elif val=="Amber": return "background-color: #fcebcf"
    elif val=="Green": return "background-color: #ccf2d0"
    return ""

def style_rag_col(df):
    if "rag" in df.columns:
        return df.style.apply(
            lambda col: [color_rag(v) for v in col] if col.name=="rag" else ["" for _ in col],
            axis=0
        )
    return df

###############################################################################
# 6) Main synergy coverage + RAG + brainstorming + mitigation
###############################################################################
st.subheader("Load CSV from Method 1 & Perform Synergy Coverage")

csv_path = st.text_input("Path to CSV from Method 1", value="clean_risks.csv")
if st.button("Analyze CSV"):
    try:
        df = pd.read_csv(csv_path)
        st.success(f"Loaded {df.shape[0]} lines from {csv_path}.")

        # Basic columns check
        needed_cols = ["risk_id","risk_description","risk_type","stakeholder","severity","probability","combined_score"]
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

        # synergy coverage: stakeholder × risk_type
        synergy_cov = df.groupby(["stakeholder","risk_type"]).size().reset_index(name="count")
        st.markdown("### Synergy Coverage: (stakeholder × risk_type) Count")
        st.dataframe(synergy_cov.head(30))

        st.markdown("#### Heatmap: coverage count")
        chart_cov = alt.Chart(synergy_cov).mark_rect().encode(
            x=alt.X("risk_type:N"),
            y=alt.Y("stakeholder:N"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["stakeholder","risk_type","count"]
        ).properties(width=500, height=300)
        st.altair_chart(chart_cov, use_container_width=True)

        # synergy gap: combos with <2 lines
        synergy_gaps = synergy_cov[synergy_cov["count"]<2]
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
                for _,row in synergy_gaps.iterrows()
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
                        {"role":"system","content":"You are a synergy coverage brainstorming assistant."},
                        {"role":"user","content":prompt_b}
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
            if pick_risk and pick_risk!="(none)":
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
                            {"role":"system","content":"You are a helpful AI risk mitigation advisor."},
                            {"role":"user","content":prompt_m}
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

###############################################################################
# 7) Optional: Semantic Coverage w/ Embeddings + FAISS
###############################################################################
st.subheader("Optional: Semantic Coverage with Embeddings + FAISS")
embed_path = st.text_input("Embeddings .npy", "embeddings.npy")
faiss_path = st.text_input("FAISS index .faiss", "faiss_index.faiss")
lines_for_sem = st.text_area("Lines to check coverage semantically", height=100)

if st.button("Check Semantic Coverage"):
    user_lines = [l.strip() for l in lines_for_sem.split("\n") if l.strip()]
    if not user_lines:
        st.warning("No lines to check.")
    else:
        try:
            df2 = pd.read_csv(csv_path)
            main_embeds = np.load(embed_path)
            idx = faiss.read_index(faiss_path)

            # embed user lines
            model_name = "all-MiniLM-L6-v2"
            embedder = SentenceTransformer(model_name)
            user_vecs = embedder.encode(user_lines, show_progress_bar=False).astype("float32")

            k=3
            D,I = idx.search(user_vecs, k)
            st.markdown("### Semantic Coverage Matches:")
            for i,uln in enumerate(user_lines):
                st.write(f"**Your line**: {uln}")
                for rank,(nid,dist_) in enumerate(zip(I[i],D[i]), start=1):
                    row_ = df2.iloc[nid]
                    st.write(f"{rank}) {row_['risk_description']} (dist={dist_:.3f}, stkh={row_.get('stakeholder','')}, type={row_.get('risk_type','')})")
                st.write("---")
        except Exception as e:
            st.error(f"Semantic coverage error: {str(e)}")

st.info("All done! This script uses st.experimental_set_query_params + st.stop instead of st.experimental_rerun.")
