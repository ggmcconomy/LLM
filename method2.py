############################################################
# method2.py
#
# Single-tab Mural OAuth flow that:
#  - If not authorized => user sees "Authorize" button
#  - On button click => redirect to Mural in the same tab
#  - Mural returns ?code=... => exchange for token => remove code => show dashboard
#  - Then we do coverage analysis vs. "clean_risks.csv"
############################################################

import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
from urllib.parse import urlencode
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
import openai  # optional, for LLM feedback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# optional for embeddings coverage
from sentence_transformers import SentenceTransformer
import faiss

# --- 1) Basic Page Setup ---
st.set_page_config(page_title="Method 2 - Single-Tab Mural OAuth", layout="wide")
st.title("Method 2 - AI Risk Coverage & Mitigation (Single Tab Mural OAuth)")

plt.style.use("ggplot")

############################################################
# 2) Load Secrets
############################################################
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY

    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]  # must match your site EXACTLY
    MURAL_BOARD_ID = st.secrets.get("MURAL_BOARD_ID", "")
except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()

############################################################
# 3) Mural OAuth
############################################################
def get_mural_auth_url():
    # Build standard URL to Mural
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "response_type": "code",
        "state": "unique-state-123"
    }
    return "https://app.mural.co/api/public/v1/authorization/oauth2/?" + urlencode(params)

def exchange_code_for_token(auth_code):
    """Exchange the ?code= from Mural for an access token."""
    url = "https://app.mural.co/api/public/v1/authorization/oauth2/token"
    data = {
        "client_id": MURAL_CLIENT_ID,
        "client_secret": MURAL_CLIENT_SECRET,
        "redirect_uri": MURAL_REDIRECT_URI,
        "code": auth_code,
        "grant_type": "authorization_code"
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Mural auth error: {r.status_code}")
            return None
    except Exception as ex:
        st.error(f"Mural auth exception: {str(ex)}")
        return None

def refresh_mural_token(refresh_token):
    """Optional: Refresh the Mural token if near expiry."""
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
            st.error(f"Refresh token error: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Refresh exception: {str(e)}")
        return None

# Keep tokens in session
if "mural_access_token" not in st.session_state:
    st.session_state["mural_access_token"] = None
    st.session_state["mural_refresh_token"] = None
    st.session_state["mural_expires_in"] = None
    st.session_state["mural_token_ts"] = None

############################################################
# 4) Single-Tab Handling of ?code=...
############################################################
query = st.query_params
if "code" in query and not st.session_state["mural_access_token"]:
    code_val = query["code"]
    if code_val:
        # Exchange
        token_data = exchange_code_for_token(code_val)
        if token_data:
            st.session_state["mural_access_token"] = token_data["access_token"]
            st.session_state["mural_refresh_token"] = token_data.get("refresh_token")
            st.session_state["mural_expires_in"] = token_data.get("expires_in", 900)
            st.session_state["mural_token_ts"] = datetime.now().timestamp()

            # remove code from URL
            st.set_query_params()
            st.success("Mural authorized in same tab!")
            st.rerun()

# if we have a token, check expiry
if st.session_state["mural_access_token"]:
    now_ts = datetime.now().timestamp()
    if (now_ts - st.session_state["mural_token_ts"]) > (st.session_state["mural_expires_in"] - 60):
        new_data = refresh_mural_token(st.session_state["mural_refresh_token"])
        if new_data:
            st.session_state["mural_access_token"] = new_data["access_token"]
            st.session_state["mural_refresh_token"] = new_data.get("refresh_token", st.session_state["mural_refresh_token"])
            st.session_state["mural_expires_in"] = new_data.get("expires_in", 900)
            st.session_state["mural_token_ts"] = datetime.now().timestamp()

############################################################
# 5) UI: If no token => "Authorize" button that JavaScript-redirects in same tab
############################################################
if not st.session_state["mural_access_token"]:
    st.warning("No Mural token in session. Must authorize first to proceed.")
    if st.button("Authorize Mural (Same Tab)"):
        auth_url = get_mural_auth_url()
        # JavaScript redirect in same tab
        js_code = f"""
        <script>
        window.location.href = "{auth_url}";
        </script>
        """
        st.write(js_code, unsafe_allow_html=True)
    # Return here to avoid showing the rest of the dashboard
    st.stop()

else:
    st.sidebar.success("Mural token active in session.")

############################################################
# 6) Mural Sticky Pull (Optional)
############################################################
def pull_mural_stickies(board_id, token):
    url = f"https://app.mural.co/api/public/v1/murals/{board_id}/widgets"
    headers = {"Authorization": f"Bearer {token}", "Accept":"application/json"}
    try:
        sess = requests.Session()
        rtry = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        sess.mount("https://", HTTPAdapter(max_retries=rtry))
        resp = sess.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            js = resp.json()
            widgets = js.get("value",[]) or js.get("data",[])
            lines = []
            for w in widgets:
                wtype = w.get("type","").lower().replace(" ","_")
                if wtype == "sticky_note":
                    raw = w.get("htmlText") or w.get("text") or ""
                    soup = BeautifulSoup(raw, "html.parser")
                    txt = soup.get_text(separator=" ").strip()
                    if txt:
                        lines.append(txt)
            return lines
        else:
            st.error(f"Pull mural error: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Exception pulling Mural stickies: {str(e)}")
        return []

st.sidebar.header("Mural Pull")
board_id_input = st.sidebar.text_input("Mural Board ID", MURAL_BOARD_ID)
if st.sidebar.button("Pull Sticky Notes"):
    lines = pull_mural_stickies(board_id_input, st.session_state["mural_access_token"])
    if lines:
        st.session_state["mural_notes"] = lines
        st.sidebar.success(f"Pulled {len(lines)} sticky notes.")
    else:
        st.sidebar.info("No notes or error pulling data.")

############################################################
# 7) Load clean_risks.csv from Method 1
############################################################
csv_file = "clean_risks.csv"
try:
    df_risks = pd.read_csv(csv_file)
    st.write(f"Loaded {len(df_risks)} risks from {csv_file}.")
    df_risks.fillna("", inplace=True)
except FileNotFoundError:
    st.error(f"No {csv_file} found. Run Method 1 first.")
    st.stop()

# Embeddings + index optional
embedder = None
index = None
try:
    embed_np = np.load("embeddings.npy")
    index = faiss.read_index("faiss_index.faiss")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    st.write("Embeddings & FAISS index loaded - we'll do similarity coverage check.")
except:
    st.warning("No embeddings or FAISS index found - fallback coverage logic.")

############################################################
# 8) Gather final lines
############################################################
st.subheader("1) Provide Final Workshop Lines")
default_text = "\n".join(st.session_state.get("mural_notes", []))
user_input = st.text_area("Paste your lines (one per line):", value=default_text, height=160)

############################################################
# 9) Coverage Analysis
############################################################
st.subheader("2) Coverage & Gaps")

def coverage_analysis(user_lines, df_ref):
    coverage = {"risk_type":set(), "node_name":set(), "cluster":set()}
    if not user_lines:
        return coverage
    # If embedder + index => similarity approach
    if embedder and index:
        import numpy as np
        user_embs = embedder.encode(user_lines)
        user_embs = np.array(user_embs, dtype="float32")
        k=3
        D,I = index.search(user_embs,k)
        for i, line in enumerate(user_lines):
            neighbors = df_ref.iloc[I[i]]
            coverage["risk_type"].update(neighbors["risk_type"].unique())
            coverage["node_name"].update(neighbors["node_name"].unique())
            coverage["cluster"].update(neighbors["cluster"].unique())
    else:
        # naive substring
        combined = " ".join(x.lower() for x in user_lines)
        for rt in df_ref["risk_type"].unique():
            if rt.lower() in combined:
                coverage["risk_type"].add(rt)
        for nm in df_ref["node_name"].unique():
            if nm.lower() in combined:
                coverage["node_name"].add(nm)
        for cl in df_ref["cluster"].unique():
            if str(cl).lower() in combined:
                coverage["cluster"].add(cl)
    return coverage

def create_side_by_side_chart(title, all_items, covered_set, filename):
    import matplotlib.pyplot as plt
    xcats = sorted(all_items)
    cvals = []
    mvals = []
    for x in xcats:
        if x in covered_set:
            cvals.append(1)
            mvals.append(0)
        else:
            cvals.append(0)
            mvals.append(1)
    plt.figure(figsize=(6,4))
    x = np.arange(len(xcats))
    w=0.4
    plt.bar(x-w/2, cvals, w, color='green', label='Covered')
    plt.bar(x+w/2, mvals, w, color='red', label='Missed')
    plt.xticks(x, xcats, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()

if st.button("Analyze Coverage"):
    lines = [x.strip() for x in user_input.split("\n") if x.strip()]
    if not lines:
        st.warning("No lines provided.")
    else:
        cov = coverage_analysis(lines, df_risks)

        all_types = df_risks["risk_type"].unique()
        all_nodes = df_risks["node_name"].unique()
        all_clust = df_risks["cluster"].unique()

        missed_types = set(all_types) - cov["risk_type"]
        missed_nodes = set(all_nodes) - cov["node_name"]
        missed_clust = set(all_clust) - cov["cluster"]

        st.write(f"**Risk Types Covered**: {sorted(list(cov['risk_type']))}, Missed: {sorted(list(missed_types))}")
        st.write(f"**Stakeholders Covered**: {sorted(list(cov['node_name']))}, Missed: {sorted(list(missed_nodes))}")
        st.write(f"**Clusters Covered**: {sorted(list(cov['cluster']))}, Missed: {sorted(list(missed_clust))}")

        # bar charts
        create_side_by_side_chart("Risk Type Coverage", all_types, cov["risk_type"], "risk_type_cov.png")
        st.image("risk_type_cov.png", use_column_width=True)
        create_side_by_side_chart("Stakeholder Coverage", all_nodes, cov["node_name"], "stakeholder_cov.png")
        st.image("stakeholder_cov.png", use_column_width=True)
        create_side_by_side_chart("Cluster Coverage", all_clust, cov["cluster"], "cluster_cov.png")
        st.image("cluster_cov.png", use_column_width=True)

        # optional LLM feedback
        if st.checkbox("Generate LLM Gap Feedback?"):
            prompt = f"""
You are an AI coverage assistant. 
User lines:
{chr(10).join("- "+l for l in lines)}

Coverage found:
 - risk_types: {sorted(list(cov['risk_type']))}, missed: {sorted(list(missed_types))}
 - stakeholders: {sorted(list(cov['node_name']))}, missed: {sorted(list(missed_nodes))}
 - clusters: {sorted(list(cov['cluster']))}, missed: {sorted(list(missed_clust))}

Explain the significance of the missed categories and how to improve coverage.
"""
            try:
                resp = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=300
                )
                st.markdown("### LLM Gap Feedback")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {str(e)}")

############################################################
# 10) Reset
############################################################
if st.button("Reset Session"):
    for k in ["mural_access_token","mural_refresh_token","mural_expires_in","mural_token_ts","mural_notes"]:
        if k in st.session_state:
            st.session_state.pop(k)
    st.set_query_params()  # remove any leftover code param
    st.rerun()

st.info("Done. This script ensures single-tab OAuth: the user must click 'Authorize Mural' to do it in the same tab, then returns here with ?code=... and sees the dashboard.")
