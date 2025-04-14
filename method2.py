###############################################################################
# method2.py
#
# Method 2: Advanced Synergy Coverage & Visualization
#  - Reads "clean_risks.csv" from Method 1
#  - Adds RAG color-coding, synergy coverage checks (stakeholder × risk type × attributes),
#  - Visual dashboards including synergy pivot, coverage heatmaps,
#  - Provides more advanced synergy-based textual feedback from GPT,
#  - (Optional) advanced semantic coverage with embeddings & FAISS index from Method 1.
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
st.set_page_config(page_title="Method 2 - Advanced Synergy Coverage", layout="wide")
st.title("AI Risk Coverage Dashboard (Method 2) - Advanced Synergy Version")

###############################################################################
# 1) Utility & RAG
###############################################################################
def assign_rag(combined_score):
    """
    Example thresholds:
     - combined_score >= 13 => Red
     - 9..12 => Amber
     - below 9 => Green
    """
    if combined_score >= 13:
        return "Red"
    elif combined_score >= 9:
        return "Amber"
    else:
        return "Green"

def color_rag(val):
    if val == "Red":
        return "background-color: #f8d0d0"
    elif val == "Amber":
        return "background-color: #fcebcf"
    elif val == "Green":
        return "background-color: #ccf2d0"
    return ""

def style_rag_col(df):
    if "rag" in df.columns:
        return df.style.apply(lambda col: [color_rag(v) for v in col] if col.name=="rag" else ["" for _ in col], axis=0)
    return df

###############################################################################
# 2) Load CSV
###############################################################################
csv_file = st.text_input("CSV from Method 1 (e.g. 'clean_risks.csv')", "clean_risks.csv")

if st.button("Load & Process CSV"):
    try:
        df = pd.read_csv(csv_file)
        st.success(f"Loaded {df.shape[0]} rows from {csv_file}.")
        # Ensure columns we expect
        needed_cols = ["risk_id","risk_description","risk_type","stakeholder","severity","probability","combined_score"]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}. Please confirm your CSV structure.")
            st.stop()

        # 1) Assign RAG
        df["rag"] = df["combined_score"].apply(assign_rag)

        # 2) Show color-coded table
        st.subheader("RAG-Assigned Risks")
        styled_df = style_rag_col(df)
        st.dataframe(styled_df, use_container_width=True)

        # 3) Download updated
        csv_rag = df.to_csv(index=False)
        st.download_button("Download RAG CSV", data=csv_rag, file_name="clean_risks_with_rag.csv")

        # 4) Basic synergy coverage approach: stakeholder × risk_type
        #    We count how many lines exist for each combination
        st.subheader("Synergy Coverage (Stakeholder × Risk Type)")
        pivot_synergy = df.groupby(["stakeholder","risk_type"]).size().reset_index(name="count")
        st.dataframe(pivot_synergy.head(30))

        # 5) Heatmap of synergy coverage density
        st.markdown("#### Heatmap: Stakeholder vs. Risk Type (coverage count)")
        coverage_chart = alt.Chart(pivot_synergy).mark_rect().encode(
            x=alt.X("risk_type:N", sort=alt.SortField("risk_type", order="ascending")),
            y=alt.Y("stakeholder:N", sort=alt.SortField("stakeholder", order="ascending")),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="reds"), title="Coverage Count"),
            tooltip=["stakeholder","risk_type","count"]
        ).properties(width=500, height=300)
        st.altair_chart(coverage_chart, use_container_width=True)

        # 6) Heatmap for average combined_score as well
        st.markdown("#### Heatmap: Stakeholder vs. Risk Type (avg combined_score)")
        pivot_score = df.groupby(["stakeholder","risk_type"])["combined_score"].mean().reset_index()
        score_chart = alt.Chart(pivot_score).mark_rect().encode(
            x=alt.X("risk_type:N", sort=alt.SortField("risk_type", order="ascending")),
            y=alt.Y("stakeholder:N", sort=alt.SortField("stakeholder", order="ascending")),
            color=alt.Color("combined_score:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0,25], clamp=True), title="Avg Score"),
            tooltip=["stakeholder","risk_type","combined_score"]
        ).properties(width=500, height=300)
        st.altair_chart(score_chart, use_container_width=True)

        # 7) If you used "attributes" in Method 1, you might have columns like "transparency", "data_quality"...
        #    We can do synergy coverage for each attribute too (optional).
        possible_attributes = [c for c in df.columns if c.lower() in ("transparency","data_quality","human_oversight","model_robustness")]
        if possible_attributes:
            st.subheader("Attribute-based Synergy Coverage")
            st.write("Checking if each risk line mentions or includes these attribute columns.")
            # For demonstration: we just do a pivot for 'stakeholder' × 'risk_type' with average of 'transparency'
            for attr in possible_attributes:
                st.markdown(f"##### {attr} coverage pivot (mean or sum)")
                pivot_attr = df.groupby(["stakeholder","risk_type"])[attr].mean().reset_index()
                attr_chart = alt.Chart(pivot_attr).mark_rect().encode(
                    x=alt.X("risk_type:N", sort=alt.SortField("risk_type", order="ascending")),
                    y=alt.Y("stakeholder:N", sort=alt.SortField("stakeholder", order="ascending")),
                    color=alt.Color(f"{attr}:Q", scale=alt.Scale(scheme="blues"), title=f"Avg {attr}"),
                    tooltip=["stakeholder","risk_type",attr]
                ).properties(width=500, height=300)
                st.altair_chart(attr_chart, use_container_width=True)

        # 8) Scatter probability vs severity
        st.subheader("Scatter: Probability vs. Severity (Color = RAG)")
        scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0,5])),
            y=alt.Y("severity:Q", scale=alt.Scale(domain=[0,5])),
            color=alt.Color("rag:N", scale=alt.Scale(domain=["Green","Amber","Red"], range=["green","orange","red"])),
            tooltip=["risk_description","stakeholder","risk_type","severity","probability","combined_score","rag"]
        ).interactive()
        st.altair_chart(scatter_plot, use_container_width=True)

        st.success("Synergy coverage visuals generated successfully.")

        # 9) Provide synergy-based textual feedback with GPT
        #    We'll highlight stakeholder–risk_type combos with low coverage counts.
        synergy_gaps = pivot_synergy[pivot_synergy["count"] < 2]  # e.g., combos with <2 lines
        if synergy_gaps.empty:
            st.info("No synergy combos found with low coverage (all combos have at least 2 lines).")
        else:
            st.warning("Some synergy combos have <2 lines. Summarizing them below:")
            st.dataframe(synergy_gaps)

            # Let's produce GPT synergy feedback
            synergy_info_str = "\n".join(
                f"- {row['stakeholder']} & {row['risk_type']}: {row['count']} lines"
                for _, row in synergy_gaps.iterrows()
            )
            domain = df['domain'].iloc[0] if 'domain' in df.columns else "the AI system"
            prompt = f"""
You are an AI synergy coverage expert. We have discovered some stakeholder–risk_type combinations
with low coverage (under 2 lines) in the risk analysis for {domain}.

These combos are:
{synergy_info_str}

Explain why synergy coverage for these combos is important, what might be missing,
and how to strengthen the analysis to ensure completeness for each synergy pair.
Focus on encouraging deeper investigation into overlooked angles and bridging coverage gaps.
"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"You are a synergy coverage advisor focusing on AI harm analysis."},
                        {"role":"user","content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                synergy_feedback = response.choices[0].message.content.strip()
                st.subheader("Synergy-Based Coverage Feedback (GPT):")
                st.write(synergy_feedback)
            except Exception as e:
                st.error(f"Error generating synergy coverage feedback: {str(e)}")

    except FileNotFoundError:
        st.error(f"CSV not found: {csv_file}")
    except Exception as exc:
        st.error(f"Error processing file: {str(exc)}")

###############################################################################
# 3) Optional: Semantic Coverage with Embeddings
###############################################################################
st.subheader("Optional: Semantic Coverage with Embeddings & FAISS")

embeddings_file = st.text_input("Embeddings File (Method 1)", "embeddings.npy")
faiss_index_file = st.text_input("FAISS Index (Method 1)", "faiss_index.faiss")
user_input_sem = st.text_area("Enter lines to check coverage semantically:", height=120)

if st.button("Check Semantic Coverage"):
    if not user_input_sem.strip():
        st.warning("No user lines to check.")
    else:
        try:
            # Load the CSV & embeddings
            df_main = pd.read_csv(csv_file)
            main_embeddings = np.load(embeddings_file)
            index = faiss.read_index(faiss_index_file)

            # Embed user lines
            lines = [ln.strip() for ln in user_input_sem.split("\n") if ln.strip()]
            embed_model_name = "all-MiniLM-L6-v2"
            embedder = SentenceTransformer(embed_model_name)
            user_vecs = embedder.encode(lines, show_progress_bar=False)
            user_vecs = user_vecs.astype("float32")

            k = 3
            distances, indices = index.search(user_vecs, k)
            st.markdown("### Semantic Coverage Results:")
            for i, line in enumerate(lines):
                st.markdown(f"**User line {i+1}:** {line}")
                nn_idx = indices[i]
                nn_dst = distances[i]
                for rank, (nid, dist_) in enumerate(zip(nn_idx, nn_dst), start=1):
                    row_info = df_main.iloc[nid]
                    st.write(f"Match {rank}: {row_info['risk_description']} (dist={dist_:.3f}, stkh={row_info.get('stakeholder','')}, type={row_info.get('risk_type','')})")
                st.write("---")

        except Exception as e:
            st.error(f"Semantic coverage error: {str(e)}")

st.info("This advanced synergy coverage approach offers deeper stakeholder-type analysis plus optional embeddings for coverage checks.")
