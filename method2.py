import streamlit as st
import openai
import time
import uuid
import os
import re
import pandas as pd
import numpy as np
import asyncio
from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import hdbscan
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# Environment & Setup
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

st.set_page_config(layout="wide", page_title="Enhanced Method 1 - Risk Landscape Modelling")

# Session & Defaults
def init_state():
    if "scenario_desc" not in st.session_state:
        st.session_state[
            "scenario_desc"] = "An AI system that estimates property values from large historical datasets."
    if "stakeholders" not in st.session_state:
        st.session_state["stakeholders"] = {}
    if "agent_personas" not in st.session_state:
        st.session_state["agent_personas"] = {}
    if "depth_prompts" not in st.session_state:
        st.session_state["depth_prompts"] = {
            1: "List common or obvious AI deployment risks from this perspective.",
            2: "List second-order or synergy-based risks, focusing on factor interactions.",
            3: "Identify rare or severe risks, potentially catastrophic, often overlooked.",
            4: "Consider long-term or emergent risks where factors combine or escalate systemically."
        }
    if "general_prefix" not in st.session_state:
        st.session_state["general_prefix"] = (
            "Enumerate potential AI deployment risks, avoiding duplicates from prior expansions."
        )
    if "max_depth" not in st.session_state:
        st.session_state["max_depth"] = 4
    if "time_limit" not in st.session_state:
        st.session_state["time_limit"] = 1200
    if "temp" not in st.session_state:
        st.session_state["temp"] = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state["max_tokens"] = 600
    if "num_runs" not in st.session_state:
        st.session_state["num_runs"] = 1
    if "enable_critic" not in st.session_state:
        st.session_state["enable_critic"] = True
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = "gpt-4o-mini"
    if "do_scoring" not in st.session_state:
        st.session_state["do_scoring"] = True
    if "score_model" not in st.session_state:
        st.session_state["score_model"] = "gpt-4o-mini"
    if "attributes" not in st.session_state:
        st.session_state["attributes"] = {
            "Model Robustness": {"low": 3, "high": 7},
            "Human Oversight": {"low": 3, "high": 7},
            "Transparency": {"low": 3, "high": 7},
            "Data Quality": {"low": 3, "high": 7}
        }
    if "do_clustering" not in st.session_state:
        st.session_state["do_clustering"] = True
    if "min_cluster_size" not in st.session_state:
        st.session_state["min_cluster_size"] = 10
    if "results_detailed" not in st.session_state:
        st.session_state["results_detailed"] = pd.DataFrame()
    if "results_clean" not in st.session_state:
        st.session_state["results_clean"] = pd.DataFrame()
    if "embedder" not in st.session_state:
        st.session_state["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")

# LLM Helpers
@lru_cache(maxsize=1000)
def call_llm(prompt, temperature=0.7, max_tokens=600, model="gpt-4o-mini"):
    for attempt in range(3):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"
    return ""

async def async_call_llm(prompt, temperature=0.7, max_tokens=600, model="gpt-4o-mini"):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, lambda: call_llm(prompt, temperature, max_tokens, model))
    return result

def parse_bullets(text):
    lines = [re.sub(r"^\d+\.\s*|-•*\s*|\*+\s*", "", ln).strip() for ln in text.split("\n") if ln.strip()]
    return [ln for ln in lines if ln]

# Generate Stakeholders & Personas
async def generate_stakeholders(over_inclusive: bool, scenario: str, llm_model: str):
    count_min = 8 if over_inclusive else 5
    count_max = 12 if over_inclusive else 8
    prompt_s = f"""
Given the AI scenario: '{scenario}'
Identify {count_min} to {count_max} distinct STAKEHOLDERS (groups or organizations impacted).
Aim to be comprehensive, listing each as a short bullet.
"""
    resp = await async_call_llm(prompt_s, temperature=0.7, max_tokens=400, model=llm_model)
    lines = parse_bullets(resp)
    return {ln: 0.7 for ln in lines if ln}

async def generate_personas(scenario: str, llm_model: str):
    prompt_p = f"""
Given the AI scenario: '{scenario}'
Propose 5 to 6 AGENT PERSONAS (individual archetypes with unique concerns).
Return each as: 'Persona Name: brief focus'.
"""
    resp = await async_call_llm(prompt_p, temperature=0.7, max_tokens=400, model=llm_model)
    lines = parse_bullets(resp)
    results = {}
    for ln in lines:
        name = ln.split(":")[0].strip() if ":" in ln else ln.strip()
        if name:
            results[name] = 0.5
    return results

# Classification, Scoring & Mitigation
async def enhanced_classify_score_mitigate(line: str, scenario: str, attributes: dict, classification_model: str):
    categories = ["Technical", "Financial", "Ethical", "Operational", "Regulatory", "Social", "Legal", "Unknown"]
    line_clean = re.sub(r"^\d+\.\s*|\*+", "", line).strip()

    attr_scores = {}
    for attr, rng in attributes.items():
        score = np.mean([rng["low"], rng["high"]]) / 10
        attr_scores[attr] = score

    prompt = f"""
Given the AI scenario: '{scenario}'
And this risk: '{line_clean}'
With attributes: {attr_scores}

1) Classify into ONE category: {', '.join(categories)} (pick Unknown if unclear).
2) Assign severity (1-5).
3) Assign probability (1-5).
4) Estimate confidence in scoring (0.0-1.0).
5) Suggest one mitigation strategy.

Format EXACTLY as:
Class= <category>
Severity= <number>
Probability= <number>
Confidence= <number>
Mitigation= <strategy>
"""
    resp = await async_call_llm(prompt, temperature=0, max_tokens=200, model=classification_model)

    cat = "Unknown"
    severity = 3.0
    probability = 3.0
    confidence = 0.5
    mitigation = "No mitigation suggested."

    for line in resp.split("\n"):
        if line.startswith("Class="):
            c_raw = line.split("=")[1].strip().capitalize()
            cat = c_raw if c_raw in categories else "Unknown"
        elif line.startswith("Severity="):
            severity = float(line.split("=")[1].strip())
        elif line.startswith("Probability="):
            probability = float(line.split("=")[1].strip())
        elif line.startswith("Confidence="):
            confidence = float(line.split("=")[1].strip())
        elif line.startswith("Mitigation="):
            mitigation = line.split("=", 1)[1].strip()

    if cat == "Technical" and "Data Quality" in attr_scores:
        severity *= (1 + (1 - attr_scores["Data Quality"]) * 0.2)
        probability *= (1 + (1 - attr_scores["Data Quality"]) * 0.2)
    elif cat == "Ethical" and "Transparency" in attr_scores:
        severity *= (1 + (1 - attr_scores["Transparency"]) * 0.2)

    combined_score = severity * probability
    return line_clean, cat, severity, probability, confidence, combined_score, mitigation

async def second_pass_classify_score_mitigate(df_clean, scenario, attributes, classification_model):
    df = df_clean.copy()
    tasks = [
        enhanced_classify_score_mitigate(row["risk_description"], scenario, attributes, classification_model)
        for _, row in df.iterrows()
    ]
    results = await asyncio.gather(*tasks)

    for i, (line_clean, cat, sev, prob, conf, cscore, mitig) in enumerate(results):
        df.at[i, "risk_description"] = line_clean
        df.at[i, "risk_type"] = cat
        df.at[i, "severity"] = sev
        df.at[i, "probability"] = prob
        df.at[i, "confidence"] = conf
        df.at[i, "combined_score"] = cscore
        df.at[i, "mitigation_suggestion"] = mitig
    return df

# Critic Pass
async def critic_pass(new_lines, node_name, depth, scenario, llm_model):
    if not new_lines:
        return ""
    prompt_c = f"""
For the AI scenario: '{scenario}'
You generated these risks for '{node_name}' at depth {depth}:
{"; ".join(new_lines)}

Provide a concise rationale for their importance or novelty from {node_name}'s perspective.
"""
    resp = await async_call_llm(prompt_c, temperature=0, max_tokens=150, model=llm_model)
    return re.sub(r"(From the perspective of.*)|(From the viewpoint of.*)", "", resp, flags=re.IGNORECASE).strip()

# Semantic Deduplication
def deduplicate_risks(df_clean, embedder, similarity_threshold=0.85):
    df = df_clean.copy()
    texts = df["risk_description"].tolist()
    embeddings = embedder.encode(texts, show_progress_bar=False)

    sim_matrix = cosine_similarity(embeddings)
    to_drop = set()

    for i in range(len(texts)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(texts)):
            if sim_matrix[i, j] > similarity_threshold:
                if df.at[i, "combined_score"] >= df.at[j, "combined_score"]:
                    to_drop.add(j)
                else:
                    to_drop.add(i)

    df = df.drop(index=list(to_drop)).reset_index(drop=True)
    return df

# Bayesian Tree
global_node_counter = 0

def generate_node_id():
    global global_node_counter
    global_node_counter += 1
    return global_node_counter

async def run_bayesian_tree(
        scenario_desc,
        node_dict,
        temperature,
        max_tokens,
        max_depth,
        time_limit,
        run_index,
        enable_critic,
        llm_model,
        attributes
):
    start_time = time.time()
    expansions = []
    discovered_set = set()

    all_nodes = {}
    att_str_list = [f"{k}: {rng['low']}-{rng['high']}" for k, rng in attributes.items()]
    attribute_text = ", ".join(att_str_list) if att_str_list else "None"

    for name, weight in node_dict.items():
        nid = generate_node_id()
        all_nodes[(nid, 1)] = {
            "node_name": name,
            "weight": weight,
            "alpha": 1.0,
            "beta": 1.0,
            "linesSoFar": [],
            "depth": 1,
            "no_new_count": 0,
            "parent_node_id": None
        }

    def sample_beta(a, b):
        return np.random.beta(max(a, 0.01), max(b, 0.01))

    async def expand_node(nid, dpt, ndict, forced=False):
        node_name = ndict["node_name"]
        depth_prompt = st.session_state["depth_prompts"].get(dpt, "Provide more AI risks.")
        prompt_text = f"""
{st.session_state["general_prefix"]}

Scenario: {scenario_desc}

Attributes: {attribute_text}

Node: {node_name} (importance={ndict["weight"]})
Depth: {dpt}
Task: {depth_prompt}
List 5-10 new risks from {node_name}'s perspective, avoiding known risks.
"""
        t0 = time.time()
        resp = await async_call_llm(prompt_text, temperature, max_tokens, model=llm_model)
        lines_parsed = parse_bullets(resp)
        new_lines = [ln for ln in lines_parsed if ln not in discovered_set]
        discovered_set.update(new_lines)
        ndict["linesSoFar"].extend(new_lines)

        critic_text = ""
        if enable_critic and new_lines:
            critic_text = await critic_pass(new_lines, node_name, dpt, scenario_desc, llm_model)

        expansions.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_index": run_index,
            "node_id": nid,
            "parent_node_id": ndict["parent_node_id"],
            "node_name": node_name,
            "depth": dpt,
            "forced_mini_pass": forced,
            "prompt_used": prompt_text,
            "lines_generated": lines_parsed,
            "lines_added": new_lines,
            "critic_text": critic_text,
            "alpha_before": ndict["alpha"],
            "beta_before": ndict["beta"],
            "time_spent": time.time() - t0
        })

        new_count = len(new_lines)
        if new_count == 0:
            ndict["no_new_count"] += 1
            ndict["beta"] += 3
        else:
            ndict["alpha"] += new_count * ndict["weight"]
            ndict["beta"] += max(1, 10 - new_count)
        return new_count

    # Coverage pass
    d = 1
    while d <= max_depth and (time.time() - start_time < time_limit):
        depth_nodes = [(k, v) for (k, dp), v in all_nodes.items() if dp == d]
        tasks = [expand_node(nid, d, ndict, forced=True) for nid, ndict in depth_nodes]
        await asyncio.gather(*tasks)

        for nid, ndict in depth_nodes:
            if d < max_depth:
                child_id = generate_node_id()
                all_nodes[(child_id, d + 1)] = {
                    "node_name": ndict["node_name"],
                    "weight": ndict["weight"],
                    "alpha": 1.0,
                    "beta": 1.0,
                    "linesSoFar": ndict["linesSoFar"][:],
                    "depth": d + 1,
                    "no_new_count": 0,
                    "parent_node_id": nid
                }
        d += 1

    # Thompson sampling
    while time.time() - start_time < time_limit:
        valid_nodes = {
            (nid, dp): nodeval
            for (nid, dp), nodeval in all_nodes.items()
            if nodeval["depth"] <= max_depth and nodeval["no_new_count"] < 2
        }
        if not valid_nodes:
            break
        best_val = -1
        best_key = None
        for nk, nv in valid_nodes.items():
            draw = sample_beta(nv["alpha"], nv["beta"])
            if draw > best_val:
                best_val = draw
                best_key = nk

        if not best_key:
            break
        chosen_nid, chosen_depth = best_key
        chosen_dict = valid_nodes[best_key]
        await expand_node(chosen_nid, chosen_depth, chosen_dict, forced=False)

        if chosen_depth < max_depth:
            c_id = generate_node_id()
            all_nodes[(c_id, chosen_depth + 1)] = {
                "node_name": chosen_dict["node_name"],
                "weight": chosen_dict["weight"],
                "alpha": 1.0,
                "beta": 1.0,
                "linesSoFar": chosen_dict["linesSoFar"][:],
                "depth": chosen_depth + 1,
                "no_new_count": 0,
                "parent_node_id": chosen_nid
            }

    return expansions

# Build DataFrames, Clustering
def build_dataframes_from_expansions(expansions):
    detail_records = []
    clean_records = []

    for exp in expansions:
        drow = exp.copy()
        drow["num_lines_added"] = len(exp["lines_added"])
        detail_records.append(drow)

        for ln in exp["lines_added"]:
            clean_records.append({
                "risk_id": str(uuid.uuid4()),
                "run_index": exp["run_index"],
                "node_id": exp["node_id"],
                "node_name": exp["node_name"],
                "depth": exp["depth"],
                "risk_description": ln,
                "critic_text": exp["critic_text"],
                "risk_type": "Unknown",
                "severity": 0.0,
                "probability": 0.0,
                "confidence": 0.0,
                "combined_score": 0.0,
                "mitigation_suggestion": "",
                "cluster": -1
            })

    df_detailed = pd.DataFrame(detail_records)
    col_order_d = [
        "timestamp", "run_index", "node_id", "parent_node_id", "node_name", "depth",
        "forced_mini_pass", "prompt_used", "lines_generated", "lines_added", "critic_text",
        "alpha_before", "beta_before", "num_lines_added", "time_spent"
    ]
    df_detailed = df_detailed[[c for c in col_order_d if c in df_detailed.columns]]

    df_clean = pd.DataFrame(clean_records)
    col_order_c = [
        "risk_id", "run_index", "node_id", "node_name", "depth", "risk_description", "critic_text",
        "risk_type", "severity", "probability", "confidence", "combined_score", "mitigation_suggestion", "cluster"
    ]
    df_clean = df_clean[col_order_c]
    return df_detailed, df_clean

def embed_and_cluster(df_clean, min_cluster_size=10, embed_model="all-MiniLM-L6-v2"):
    df = df_clean.copy()
    embedder = st.session_state["embedder"]
    texts = df["risk_description"].tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    c_ids = clusterer.fit_predict(embeddings)
    df["cluster"] = c_ids

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index
