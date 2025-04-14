############################################################
# method2.py
#
# Single-tab Mural OAuth with new st.query_params & st.set_query_params,
# only showing the coverage analysis AFTER successful authorization.
############################################################

import os
import sys
import uuid
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

############################################################
# 1) PAGE & STYLING
############################################################
st.set_page_config(page_title="Method 2 - Mural OAuth Single Tab", layout="wide")
plt.style.use("ggplot")

st.title("Method 2 - Single Tab Mural Auth & Coverage Dashboard")

############################################################
# 2) LOAD SECRETS
############################################################
try:
    # Mural
    MURAL_CLIENT_ID = st.secrets["MURAL_CLIENT_ID"]
    MURAL_CLIENT_SECRET = st.secrets["MURAL_CLIENT_SECRET"]
    MURAL_BOARD_ID = st.secrets["MURAL_BOARD_ID"]
    MURAL_REDIRECT_URI = st.secrets["MURAL_REDIRECT_URI"]
    MURAL_WORKSPACE_ID = st.secrets.get("MURAL_WORKSPACE_ID","someWorkspace")

    # For LLM or coverage if needed
    import openai
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY

except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()

############################################################
# 3) MURAL OAUTH UTILS
############################################################
def get_mural_auth_url():
    """Build the OAuth link to Mural for single-tab auth."""
    params = {
        "client_id": MURAL_CLIENT_ID,
        "redirect_uri": MURAL_REDIRECT_URI,
        "scope": "murals:read murals:write",
        "response_type": "code",
        "state": "unique-xyz-state"
    }
    return "https://app.mural.co/api/public/v1/authorization/oauth2/?" + urlencode(params)

def exchange_code_for_token(auth_code):
    """Exchange the ?code= for an access token in same tab."""
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

def refresh_token(refresh_token):
    """Refresh the Mural token if near expiry."""
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
            st.error(f"Mural refresh token error: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Refresh token exception: {str(e)}")
        return None

# Keep tokens in session
if "mural_access_token" not in st.session_state:
    st.session_state["mural_access_token"] = None
    st.session_state["mural_refresh_token"] = None
    st.session_state["mural_expires_in"] = None
    st.session_state["mural_token_ts"] = None

############################################################
# 4) CHECK FOR ?code= OR existing token
############################################################
qs = st.query_params
if "code" in qs and not st.session_state["mural_access_token"]:
    # user returned from Mural with code
    code_val = qs["code"]
    if code_val:
        token_data = exchange_code_for_token(code_val)
        if token_data:
            st.session_state["mural_access_token"] = token_data["access_token"]
            st.session_state["mural_refresh_token"] = token_data.get("refresh_token")
            st.session_state["mural_expires_in"] = token_data.get("expires_in",900)
            st.session_state["mural_token_ts"] = time.time()
            # clear code param
            st.set_query_params()
            st.success("Authenticated with Mural in same tab!")
            st.rerun()

# If we have token, see if it needs refresh
if st.session_state["mural_access_token"]:
    now_ts = time.time()
    if (now_ts - st.session_state["mural_token_ts"]) > (st.session_state["mural_expires_in"]-60):
        new_data = refresh_token(st.session_state["mural_refresh_token"])
        if new_data:
            st.session_state["mural_access_token"] = new_data["access_token"]
            st.session_state["mural_refresh_token"] = new_data.get("refresh_token", st.session_state["mural_refresh_token"])
            st.session_state["mural_expires_in"] = new_data.get("expires_in",900)
            st.session_state["mural_token_ts"] = time.time()

############################################################
# 5) If user STILL not authorized => only show "Authorize" button
############################################################
if not st.session_state["mural_access_token"]:
    st.warning("No Mural token. Must authorize first.")
    auth_url = get_mural_auth_url()

    # We'll do a single-tab approach with JS on button press
    if st.button("Authorize Mural in Same Tab"):
        # JavaScript redirect
        js_code = f"""
        <script>
        window.location.href = "{auth_url}";
        </script>
        """
        st.write(js_code, unsafe_allow_html=True)

    st.stop()

############################################################
# 6) Now that user is authorized => Show the MAIN DASHBOARD
############################################################
st.success("You are authorized with Mural! Proceed to coverage analysis...")

# For your "pull from Mural" logic, define e.g. a function:
def pull_mural_stickies(mural_id):
    url = f"https://app.mural.co/api/public/v1/murals/{mural_id}/widgets"
    headers = {
        "Authorization": f"Bearer {st.session_state['mural_access_token']}",
        "Accept": "application/json"
    }
    try:
        sess = requests.Session()
        rtry = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        sess.mount("https://", HTTPAdapter(max_retries=rtry))
        resp = sess.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            wlist = data.get("value",[]) or data.get("data",[])
            lines=[]
            for w in wlist:
                wtype = w.get("type","").replace(" ","_").lower()
                if wtype == "sticky_note":
                    raw = w.get("htmlText") or w.get("text") or ""
                    soup = BeautifulSoup(raw,"html.parser")
                    txt = soup.get_text(separator=" ").strip()
                    if txt:
                        lines.append(txt)
            return lines
        else:
            st.error(f"Failed to pull mural notes: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Pull mural exception: {str(e)}")
        return []

st.sidebar.header("Mural Integration")
mural_override = st.sidebar.text_input("Mural Board ID", value=MURAL_BOARD_ID)
if st.sidebar.button("Pull Sticky Notes"):
    lines = pull_mural_stickies(mural_override)
    if lines:
        st.session_state["mural_notes"] = lines
        st.sidebar.success(f"Pulled {len(lines)} sticky notes!")
    else:
        st.sidebar.warning("No notes found or error.")


############################################################
# 7) Load "clean_risks.csv" from Method 1
############################################################
csv_file = "clean_risks.csv"
try:
    df_risks = pd.read_csv(csv_file)
    st.write(f"Loaded {len(df_risks)} rows from {csv_file}.")
    df_risks.fillna("", inplace=True)
except FileNotFoundError:
    st.error(f"{csv_file} not found. Please run Method 1 first.")
    st.stop()

# If you want embeddings coverage approach
embedder = None
index = None
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    embed_np = np.load("embeddings.npy")
    index = faiss.read_index("faiss_index.faiss")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    st.write("Embeddings found => coverage can use nearest neighbors.")
except:
    st.warning("No embeddings or faiss index found => naive coverage approach.")


############################################################
# 8) Gather final lines from Mural or user
############################################################
st.subheader("1) Provide Final Lines")
default_text = "\n".join(st.session_state.get("mural_notes", []))
user_input = st.text_area("Paste final lines (one per line):", default_text, height=180)

############################################################
# 9) Coverage + Gaps
############################################################
st.subheader("2) Coverage & Gaps")

def coverage_analysis(user_lines, ref_df):
    coverage = {
        "risk_type": set(),
        "node_name": set(),
        "cluster": set()
    }
    lines = [l.strip() for l in user_lines if l.strip()]
    if not lines:
        return coverage

    if embedder and index:
        # use embeddings
        import numpy as np
        user_embs = embedder.encode(lines)
        user_embs = np.array(user_embs, dtype="float32")
        k=3
        D,I = index.search(user_embs, k)
        for i in range(len(lines)):
            neighbors = ref_df.iloc[I[i]]
            coverage["risk_type"].update(neighbors["risk_type"].unique())
            coverage["node_name"].update(neighbors["node_name"].unique())
            coverage["cluster"].update(neighbors["cluster"].unique())
    else:
        # naive substring
        combined = " ".join(l.lower() for l in lines)
        for rt in ref_df["risk_type"].unique():
            if rt.lower() in combined:
                coverage["risk_type"].add(rt)
        for nm in ref_df["node_name"].unique():
            if nm.lower() in combined:
                coverage["node_name"].add(nm)
        for cl in ref_df["cluster"].unique():
            if str(cl).lower() in combined:
                coverage["cluster"].add(cl)
    return coverage

def side_by_side_coverage_chart(title, all_items, covered_set, filename):
    cat_list = sorted(all_items)
    covered_counts = []
    missed_counts = []
    for c in cat_list:
        if c in covered_set:
            covered_counts.append(1)
            missed_counts.append(0)
        else:
            covered_counts.append(0)
            missed_counts.append(1)

    fig = plt.figure(figsize=(6,4))
    x = np.arange(len(cat_list))
    w=0.4
    plt.bar(x - w/2, covered_counts, w, color='green', label='Covered')
    plt.bar(x + w/2, missed_counts, w, color='red', label='Missed')
    plt.xticks(x, cat_list, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()

if st.button("Analyze Coverage"):
    lines = [l for l in user_input.split("\n") if l.strip()]
    if not lines:
        st.warning("No lines to analyze.")
    else:
        cov = coverage_analysis(lines, df_risks)
        all_types = df_risks["risk_type"].unique()
        all_nodes = df_risks["node_name"].unique()
        all_clusters = df_risks["cluster"].unique()

        missed_types = set(all_types) - cov["risk_type"]
        missed_nodes = set(all_nodes) - cov["node_name"]
        missed_clusters = set(all_clusters) - cov["cluster"]

        st.write(f"**Risk Types**: covered={sorted(list(cov['risk_type']))}, missed={sorted(list(missed_types))}")
        st.write(f"**Stakeholders**: covered={sorted(list(cov['node_name']))}, missed={sorted(list(missed_nodes))}")
        st.write(f"**Clusters**: covered={sorted(list(cov['cluster']))}, missed={sorted(list(missed_clusters))}")

        side_by_side_coverage_chart("Risk Type Coverage", all_types, cov["risk_type"], "risk_type_cov.png")
        st.image("risk_type_cov.png", use_column_width=True)
        side_by_side_coverage_chart("Stakeholder Coverage", all_nodes, cov["node_name"], "stakeholder_cov.png")
        st.image("stakeholder_cov.png", use_column_width=True)
        side_by_side_coverage_chart("Cluster Coverage", all_clusters, cov["cluster"], "cluster_cov.png")
        st.image("cluster_cov.png", use_column_width=True)

        # Optional LLM gap feedback
        if st.checkbox("Generate LLM Gap Feedback?"):
            prompt = f"""
You are an AI coverage assistant. The user lines:
{chr(10).join("- "+l for l in lines)}

We found coverage:
 - risk_types covered: {sorted(list(cov['risk_type']))}, missed: {sorted(list(missed_types))}
 - stakeholders covered: {sorted(list(cov['node_name']))}, missed: {sorted(list(missed_nodes))}
 - clusters covered: {sorted(list(cov['cluster']))}, missed: {sorted(list(missed_clusters))}

Explain why the missing categories matter, and how to fix.
"""
            try:
                resp = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=300
                )
                st.markdown("#### Gap Feedback")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {str(e)}")


############################################################
# 10) RESET
############################################################
if st.button("Reset & Clear"):
    for k in ["mural_access_token","mural_refresh_token","mural_expires_in","mural_token_ts","mural_notes"]:
        if k in st.session_state:
            st.session_state.pop(k)
    st.set_query_params()
    st.rerun()

st.info("Done. This script won't load coverage or the main UI until after Mural auth is successful.")
