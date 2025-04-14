###############################################################################
# method2.py
#
# A single Streamlit script with:
#   - Mural OAuth (pull sticky notes),
#   - Advanced synergy coverage & RAG,
#   - Brainstorming,
#   - Mitigation suggestions,
#   - Optional semantic coverage with FAISS embeddings,
#   - No calls to st.experimental_rerun() or st.experimental_set_query_params().
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
# 1) Page Config
###############################################################################
st.set_page_config(page_title="Method 2 Comprehensive", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2) – No set_query_params")

###############################################################################
# 2) Load Secrets (if using)
###############################################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID", "myworkspace")
except KeyError as e:
    st.warning(f"Missing secret: {e} - if not using Mural, ignore.")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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

def pull_mural_stickies(auth_token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        widgets = data.get("value", data.get("data", []))
        lines = []
        from bs4 import BeautifulSoup
        for w in widgets:
            if w.get('type','').replace(' ','_').lower() == 'sticky_note':
                raw_text = w.get('htmlText') or w.get('text') or ''
                cleaned = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ").strip()
                if cleaned:
                    lines.append(cleaned)
        return lines
    else:
        st.error(f"Failed to pull Mural sticky notes: {resp.status_code}")
        return []

# Checking query param for ?code=...
qs = st.experimental_get_query_params()
auth_code_list = qs.get("code", [])
auth_code = auth_code_list[0] if isinstance(auth_code_list, list) and auth_code_list else None

if auth_code and not st.session_state.access_token:
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state.access_token = token_data["access_token"]
        st.session_state.refresh_token = token_data.get("refresh_token", "")
        st.session_state.token_expires_in = token_data.get("expires_in", 900)
        st.session_state.token_timestamp = datetime.now().timestamp()
        st.success("Authenticated with Mural!")
        st.stop()  # Halt script so user sees success. We'll do no set_query_params.

# If we do have an access token, maybe refresh if it expired
if st.session_state.access_token:
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state.token_timestamp) > (st.session_state.token_expires_in - 60):
        new_data = refresh_access_token(st.session_state.refresh_token)
        if new_data:
            st.session_state.access_token = new_data["access_token"]
            st.session_state.refresh_token = new_data.get("refresh_token","")
            st.session_state.token_expires_in = new_data.get("expires_in",900)
            st.session_state.token_timestamp = datetime.now().timestamp()
            st.success("Refreshed Mural token successfully!")

# Mural authorize link
auth_url = get_authorization_url()
st.sidebar.markdown(f"[Authorize with Mural]({auth_url})")

mural_board_id = st.sidebar.text_input("Mural Board ID", MURAL_BOARD_ID)
if st.sidebar.button("Pull Mural Sticky Notes"):
    if st.session_state.access_token:
        notes = pull_mural_stickies(st.session_state.access_token, mural_board_id)
        if notes:
            st.success(f"Pulled {len(notes)} sticky notes!")
            st.session_state["mural_lines"] = notes
    else:
        st.warning("No Mural access token. Please authenticate first.")

###############################################################################
# 4) Collect Human-Finalized Risks
###############################################################################
st.subheader("Paste or Use Mural-Fetched Risks")
default_text = "\n".join(st.session_state.get("mural_lines", []))
user_input = st.text_area("Final Risk Lines from Mural or manually:", default_text, height=150)

###############################################################################
# RAG color function
###############################################################################
def assign_rag(comb):
    if comb>=13: return "Red"
    elif comb>=9: return "Amber"
    else: return "Green"

def color_rag(r):
    if r=="Red": return "background-color: #f6bbbb"
    elif r=="Amber": return "background-color: #fcebcf"
    elif r=="Green": return "background-color: #ccf2d0"
    return ""

def style_rag_col(df):
    if "rag" in df.columns:
        return df.style.apply(
            lambda col: [color_rag(v) for v in col] if col.name=="rag" else ["" for _ in col],
            axis=0
        )
    return df

###############################################################################
# 5) Main synergy coverage
###############################################################################
st.subheader("Load CSV from Method 1 & Analyze Coverage")

csv_file = st.text_input("CSV File (Method 1)", "clean_risks.csv")
if st.button("Perform Coverage Analysis"):
    try:
        df = pd.read_csv(csv_file)
        st.success(f"Loaded {df.shape[0]} lines from {csv_file}.")

        needed_cols = ["risk_id","risk_description","risk_type","stakeholder","severity","probability","combined_score"]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        df["rag"] = df["combined_score"].apply(assign_rag)

        st.markdown("### RAG-Enhanced Risk Table")
        styled_df = style_rag_col(df)
        st.dataframe(styled_df, use_container_width=True)

        # synergy coverage
        synergy = df.groupby(["stakeholder","risk_type"]).size().reset_index(name="count")
        st.markdown("### Synergy Coverage (stakeholder × risk_type)")
        st.dataframe(synergy.head(30))

        st.markdown("#### Heatmap: coverage count")
        chart_cov = alt.Chart(synergy).mark_rect().encode(
            x=alt.X("risk_type:N"),
            y=alt.Y("stakeholder:N"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["stakeholder","risk_type","count"]
        ).properties(width=500, height=300)
        st.altair_chart(chart_cov, use_container_width=True)

        synergy_gaps = synergy[synergy["count"]<2]
        if synergy_gaps.empty:
            st.info("No synergy combos with <2 coverage. Good job!")
        else:
            st.warning("Some synergy combos <2 coverage lines.")
            st.dataframe(synergy_gaps)

        st.markdown("#### Scatter: Probability vs. Severity")
        scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0,5])),
            y=alt.Y("severity:Q", scale=alt.Scale(domain=[0,5])),
            color=alt.Color("rag:N", scale=alt.Scale(domain=["Green","Amber","Red"], range=["green","orange","red"])),
            tooltip=["risk_description","stakeholder","risk_type","severity","probability","combined_score","rag"]
        ).interactive()
        st.altair_chart(scatter_plot, use_container_width=True)

        # Brainstorm
        st.subheader("Brainstorm Additional Risks for Coverage Gaps")
        focus_stakeholder = st.text_input("Focus stakeholder", "")
        focus_rtype = st.text_input("Focus risk_type", "")
        n_sugg = st.slider("Number of suggestions", 1, 10, 5)

        if st.button("Brainstorm Missing Risks"):
            synergy_str = "\n".join(
                f"- {row['stakeholder']} & {row['risk_type']}, coverage={row['count']}"
                for _,row in synergy_gaps.iterrows()
            )
            domain = df["domain"].iloc[0] if "domain" in df.columns else "the AI system"
            prompt_b = f"""
You are an AI risk brainstorming assistant for {domain}.
We found synergy combos with low coverage:
{synergy_str}

Focus on stakeholder='{focus_stakeholder}' and risk_type='{focus_rtype}' if relevant.
Propose {n_sugg} new or overlooked AI risks, each with a brief rationale.
"""
            try:
                resp = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"You are a synergy coverage brainstorming assistant."},
                              {"role":"user","content":prompt_b}],
                    max_tokens=500,
                    temperature=0.8
                )
                st.markdown("#### Brainstormed Additional Risks:")
                st.write(resp.choices[0].message.content.strip())
            except Exception as e:
                st.error(f"GPT error (brainstorm): {str(e)}")

        # Mitigation
        st.subheader("Suggest Mitigation Strategies")
        pick_risk = st.selectbox("Pick a risk from CSV", ["(none)"] + df["risk_description"].head(40).tolist())
        if st.button("Generate Mitigation"):
            if pick_risk!="(none)":
                dom = df["domain"].iloc[0] if "domain" in df.columns else "the AI domain"
                prompt_m = f"""
You are an AI risk mitigation expert for {dom}.
We have the following risk:

'{pick_risk}'

Propose 2-3 human-centric mitigation strategies as bullet points with short rationale.
"""
                try:
                    resp = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":"You are a helpful mitigation advisor."},
                                  {"role":"user","content":prompt_m}],
                        max_tokens=500,
                        temperature=0.7
                    )
                    st.markdown("#### Mitigation Strategies:")
                    st.write(resp.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"GPT error (mitigation): {str(e)}")
            else:
                st.warning("No risk chosen for mitigation.")

    except FileNotFoundError:
        st.error(f"CSV file not found: {csv_file}")
    except Exception as e:
        st.error(f"Error analyzing coverage: {str(e)}")

###############################################################################
# 6) Optional: Semantic Coverage
###############################################################################
st.subheader("Optional: Semantic Coverage with Embeddings + FAISS")
embed_file = st.text_input("Embeddings .npy", "embeddings.npy")
faiss_file = st.text_input("FAISS index", "faiss_index.faiss")
sem_input = st.text_area("Lines to check coverage semantically", height=120)

if st.button("Check Semantic Coverage"):
    lines_ = [l.strip() for l in sem_input.split("\n") if l.strip()]
    if not lines_:
        st.warning("No lines to check.")
    else:
        try:
            df2 = pd.read_csv(csv_file)
            main_embeds = np.load(embed_file)
            idx = faiss.read_index(faiss_file)

            model_embed = "all-MiniLM-L6-v2"
            embedder = SentenceTransformer(model_embed)
            user_vecs = embedder.encode(lines_, show_progress_bar=False).astype("float32")

            k=3
            D,I = idx.search(user_vecs, k)
            st.markdown("### Semantic Coverage Results")
            for i, userln in enumerate(lines_):
                st.markdown(f"**User line**: {userln}")
                for rank,(nid,dist_) in enumerate(zip(I[i],D[i]), start=1):
                    row_ = df2.iloc[nid]
                    st.write(f"Match {rank}: {row_['risk_description']} (dist={dist_:.3f}, stkh={row_.get('stakeholder','')}, type={row_.get('risk_type','')})")
                st.write("---")

        except Exception as e:
            st.error(f"Semantic coverage error: {str(e)}")

st.info("All done! No st.experimental_rerun() or st.experimental_set_query_params() used.")
