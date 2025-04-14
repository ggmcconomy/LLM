###############################################################################
# method2.py - Full Method 2 script with Mural OAuth, synergy coverage, 
# brainstorming, mitigation, semantic coverage, no deprecated calls.
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
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

###############################################################################
# 1) Page Config
###############################################################################
st.set_page_config(page_title="Method 2 - AI Risk Coverage", layout="wide")
st.title("AI Risk Coverage & Mitigation Dashboard (Method 2) - Comprehensive & 2024 Safe")

###############################################################################
# 2) Load Secrets or Fallback Env
###############################################################################
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY",""))
MURAL_CLIENT_ID = st.secrets.get("MURAL_CLIENT_ID", "")
MURAL_CLIENT_SECRET = st.secrets.get("MURAL_CLIENT_SECRET", "")
MURAL_BOARD_ID = st.secrets.get("MURAL_BOARD_ID", "")
MURAL_REDIRECT_URI = st.secrets.get("MURAL_REDIRECT_URI", "")

openai.api_key = OPENAI_API_KEY

###############################################################################
# 3) Session State (Mural OAuth)
###############################################################################
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "auth_complete" not in st.session_state:
    st.session_state.auth_complete = False

###############################################################################
# 4) Mural OAuth Helpers
###############################################################################
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
    resp = requests.post(url, data=data)
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Mural Auth failed: {resp.status_code}")
        return None

def pull_mural_stickies(token, mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        st.error(f"Failed to pull Mural stickies: {resp.status_code}")
        return []
    data = resp.json()
    widgets = data.get("value", [])
    lines = []
    for w in widgets:
        if w.get("type","").replace(" ","_").lower() == "sticky_note":
            raw_text = w.get("htmlText") or w.get("text") or ""
            cleaned = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ").strip()
            if cleaned:
                lines.append(cleaned)
    return lines

###############################################################################
# 5) Handle Mural OAuth Callback
###############################################################################
qs = st.query_params
auth_code = qs.get("code")
if isinstance(auth_code, list):
    auth_code = auth_code[0]

# If we have code but no token or not completed auth
if auth_code and not st.session_state.access_token and not st.session_state.auth_complete:
    token_data = exchange_code_for_token(auth_code)
    if token_data:
        st.session_state.access_token = token_data.get("access_token")
        st.session_state.auth_complete = True
        st.success("Authenticated with Mural!")
        st.query_params.clear()
        st.experimental_rerun()

###############################################################################
# 6) Mural Sidebar
###############################################################################
auth_url = get_authorization_url()
st.sidebar.markdown(f"[Authorize with Mural]({auth_url})")
mural_id_input = st.sidebar.text_input("Mural Board ID", value=MURAL_BOARD_ID)
if st.sidebar.button("Pull Sticky Notes") and st.session_state.access_token:
    notes = pull_mural_stickies(st.session_state.access_token, mural_id_input)
    if notes:
        st.session_state["mural_notes"] = notes
        st.success(f"Pulled {len(notes)} notes from Mural board '{mural_id_input}'.")

###############################################################################
# 7) Display / Collect Finalized Human Risks
###############################################################################
st.subheader("1. Human-Finalized Risks (From Mural or Manual)")
default_text = "\n".join(st.session_state.get("mural_notes", []))
user_input = st.text_area(
    "Final Risk Lines from Mural or manually combined:",
    value=default_text,
    height=180
)

###############################################################################
# 8) The RAG function
###############################################################################
def rag_from_score(sc):
    if sc >= 13:
        return "Red"
    elif sc >= 9:
        return "Amber"
    else:
        return "Green"

###############################################################################
# 9) Synergy Analysis & CSV from Method 1
###############################################################################
st.subheader("2. Load CSV from Method 1, Perform Synergy Coverage, Brainstorm, Mitigation")

csv_path = st.text_input("CSV from Method 1", "clean_risks.csv")

def color_rag_bg(val):
    if val=="Red": return "background-color: #f8cccc"
    elif val=="Amber": return "background-color: #fcebcf"
    elif val=="Green": return "background-color: #ccf2d0"
    return ""

def style_rag(df):
    if "rag" in df.columns:
        return df.style.apply(
            lambda col: [color_rag_bg(v) for v in col] if col.name=="rag" else ["" for _ in col],
            axis=0
        )
    return df

if st.button("Run Coverage Analysis"):
    try:
        df = pd.read_csv(csv_path)
        st.success(f"Loaded {df.shape[0]} lines from {csv_path}")
        needed_cols = ["risk_id","risk_description","risk_type","stakeholder","severity","probability","combined_score"]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            st.error(f"CSV missing these columns: {missing}")
            st.stop()

        # RAG
        df["rag"] = df["combined_score"].apply(rag_from_score)

        st.markdown("### RAG-Enhanced DataFrame (first 30 rows)")
        styled = style_rag(df.head(30))
        st.dataframe(styled, use_container_width=True)

        # synergy coverage
        synergy_cov = df.groupby(["stakeholder","risk_type"]).size().reset_index(name="count")
        st.markdown("### Synergy Coverage (stakeholder×risk_type count)")
        st.dataframe(synergy_cov.head(50))

        # synergy heatmap
        st.markdown("#### Heatmap: coverage count")
        chart_cov = alt.Chart(synergy_cov).mark_rect().encode(
            x=alt.X("risk_type:N"),
            y=alt.Y("stakeholder:N"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["stakeholder","risk_type","count"]
        ).properties(width=600, height=400)
        st.altair_chart(chart_cov, use_container_width=True)

        # synergy gap
        synergy_gaps = synergy_cov[synergy_cov["count"]<2]
        if synergy_gaps.empty:
            st.info("No synergy combos with <2 coverage lines. Good coverage!")
        else:
            st.warning("Some synergy combos have under 2 lines coverage.")
            st.dataframe(synergy_gaps)

        ############################################################################
        # BRAINSTORM: GPT suggestions
        ############################################################################
        st.subheader("Brainstorm Additional Risks for Synergy Gaps")

        # Let user pick a stakeholder or type to focus
        focus_stk = st.text_input("Focus Stakeholder (optional)", "")
        focus_rtyp = st.text_input("Focus Risk Type (optional)", "")
        n_sugg = st.slider("Number of Brainstorm Suggestions", 1, 10, 5)

        if st.button("Brainstorm Missing Risks"):
            synergy_str = "\n".join(
                f"- {row['stakeholder']} & {row['risk_type']}, coverage={row['count']}"
                for _, row in synergy_gaps.iterrows()
            )
            domain = df['domain'].iloc[0] if 'domain' in df.columns else "the AI system"
            prompt_b = f"""
You are an AI risk brainstorming assistant for {domain}.
We found synergy combos with low coverage:
{synergy_str}

Focus on stakeholder='{focus_stk}' and risk_type='{focus_rtyp}' if relevant.
Propose {n_sugg} new or overlooked AI deployment risks, each with a short rationale.
"""
            try:
                resp = openai.chat.completions.create(
                    model="gpt-40-mini",
                    messages=[
                        {"role":"system","content":"You are a synergy coverage brainstorming assistant."},
                        {"role":"user","content": prompt_b}
                    ],
                    max_tokens=600,
                    temperature=0.8
                )
                st.markdown("#### Brainstormed Additional Risks:")
                st.write(resp.choices[0].message.content.strip())
            except Exception as e:
                st.error(f"Brainstorm GPT error: {str(e)}")

        ############################################################################
        # MITIGATION: GPT suggestions
        ############################################################################
        st.subheader("Suggest Mitigation Strategies for a Risk")
        pick_risk = st.selectbox("Pick a risk from CSV", ["(none)"] + df["risk_description"].head(30).tolist())

        if st.button("Generate Mitigation"):
            if pick_risk and pick_risk!="(none)":
                domain = df["domain"].iloc[0] if "domain" in df.columns else "the AI domain"
                prompt_m = f"""
You are an AI risk mitigation expert for {domain}.
We have the following risk:

'{pick_risk}'

Propose 2-3 human-centric mitigation strategies as bullet points with short rationale.
"""
                try:
                    resp = openai.chat.completions.create(
                        model="gpt-40-mini",
                        messages=[
                            {"role":"system","content":"You are a helpful AI risk mitigation advisor."},
                            {"role":"user","content":prompt_m}
                        ],
                        max_tokens=600,
                        temperature=0.7
                    )
                    st.markdown("#### Mitigation Strategies:")
                    st.write(resp.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"Mitigation GPT error: {str(e)}")
            else:
                st.warning("No risk chosen for mitigation.")

    except FileNotFoundError:
        st.error(f"File not found: {csv_path}")
    except Exception as e:
        st.error(f"Coverage analysis error: {str(e)}")

###############################################################################
# 10) Optional: Semantic Coverage w/ Embeddings + FAISS
###############################################################################
st.subheader("3. Optional Semantic Coverage with Embeddings + FAISS")

emb_file = st.text_input("Embeddings file (.npy)", "embeddings.npy")
faiss_file = st.text_input("FAISS index (.faiss)", "faiss_index.faiss")
semantic_input = st.text_area("Risk lines to check coverage semantically", height=120)

if st.button("Check Semantic Coverage"):
    lines_ = [l.strip() for l in semantic_input.split("\n") if l.strip()]
    if not lines_:
        st.warning("No lines to check.")
    else:
        try:
            df_main = pd.read_csv(csv_path)
            main_embeds = np.load(emb_file)
            index = faiss.read_index(faiss_file)

            model_name = "all-MiniLM-L6-v2"
            embedder = SentenceTransformer(model_name)
            user_vecs = embedder.encode(lines_, show_progress_bar=False).astype("float32")

            k=3
            D,I = index.search(user_vecs, k)
            st.markdown("### Semantic Coverage Matches:")
            for i, userln in enumerate(lines_):
                st.markdown(f"**Your line**: {userln}")
                for rank, (nid,dist_) in enumerate(zip(I[i],D[i]), start=1):
                    row_ = df_main.iloc[nid]
                    st.write(f"Match {rank}: {row_['risk_description']} (distance={dist_:.3f}, stkh={row_.get('stakeholder','')}, type={row_.get('risk_type','')})")
                st.write("---")

        except FileNotFoundError:
            st.error("Embeddings or CSV file not found.")
        except Exception as e:
            st.error(f"Semantic coverage error: {str(e)}")

###############################################################################
# Done
###############################################################################
st.info("Method 2 script: Mural OAuth, synergy coverage, brainstorming, mitigation, and semantic coverage. No deprecated calls. All in one!")
