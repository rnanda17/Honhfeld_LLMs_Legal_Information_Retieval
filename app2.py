# app.py
"""
Treaty Actor Browser – v2.1
- Kimi K2 (pipeline_kimik2) is the default model/pipeline.
- GPT-5 available as an alternative.
- Dark, wrapped, larger-font relationships table.
"""

import os
import textwrap
import importlib

import streamlit as st

# Use the Kimi pipeline as the default for loading questions etc.
# (Questions & data are the same across pipelines.)
import pipeline_kimik2 as default_pipeline


# ---------------- STREAMLIT BASIC CONFIG ----------------

st.set_page_config(
    page_title="Treaty Actor Browser",
    layout="wide",
)

# Custom CSS for dark theme + cards + tables
st.markdown(
    """
    <style>
    .main {
        background-color: #050816;
        color: #f5f7ff;
    }
    .card {
        background-color: #0b1020;
        padding: 1rem 1.2rem;
        border-radius: 0.7rem;
        border: 1px solid #1e2745;
    }
    .card-title {
        font-weight: 600;
        margin-bottom: 0.35rem;
        color: #e5ecff;
        font-size: 1.05rem;
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #9ca9ff;
        margin-bottom: 0.35rem;
    }
    .card-question-text {
        color: #ffffff;          /* bright white for readability */
        line-height: 1.5;
        font-size: 0.98rem;
    }

    /* Relationships table styling */
    .rel-table-container {
        margin-top: 0.7rem;
        max-height: 650px;
        overflow-y: auto;
        border-radius: 0.7rem;
        border: 1px solid #1e2745;
    }

    table.rel-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.97rem;
        background-color: #050816;
        color: #e5ecff;
    }

    table.rel-table th,
    table.rel-table td {
        border: 1px solid #1e2745;
        padding: 0.55rem 0.8rem;
        text-align: left;
        vertical-align: top;
        white-space: normal;
        word-wrap: break-word;
    }

    table.rel-table th {
        background-color: #0f172a;
        font-weight: 600;
    }

    table.rel-table tr:nth-child(even) td {
        background-color: #060b1b;
    }

    table.rel-table tr:nth-child(odd) td {
        background-color: #050816;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Treaty Actor Browser")

# ---------------- MODEL ↔ PIPELINE MAPPING ----------------

# UI label -> (OpenRouter model id, pipeline module name)
# Kimi K2 is listed first and will be the default.
MODEL_CONFIG = {
    "Kimi K2 (moonshotai/kimi-k2)": (
        "moonshotai/kimi-k2",
        "pipeline_kimik2",
    ),
    "GPT-5 (openai/gpt-5)": (
        "openai/gpt-5",
        "pipeline_gpt5",
    ),
}


# ---------------- BUILD QUESTION DROPDOWN (from default pipeline) ----------------

question_labels = []
qid_to_question = {}

for q in default_pipeline.questions:
    qid = q["question_id"]
    qtext = q["question_text"].replace("\n", " ")
    short = textwrap.shorten(qtext, width=80, placeholder="…")
    label = f"{qid}: {short}"
    question_labels.append(label)
    qid_to_question[label] = q


# ---------------- TOP BAR: QUESTION + SETTINGS ----------------

top_left, top_right = st.columns([3, 2])

with top_left:
    st.markdown("#### Question")
    selected_label = st.selectbox(
        "Choose a fact-pattern question:",
        options=question_labels,
        label_visibility="collapsed",
    )

with top_right:
    st.markdown("#### Settings")
    selected_model_label = st.selectbox(
        "LLM model",
        options=list(MODEL_CONFIG.keys()),
        index=0,   # 0 = Kimi K2
        label_visibility="collapsed",
    )

    selected_model_id, selected_pipeline_module = MODEL_CONFIG[selected_model_label]

    # Load the correct pipeline module for the selected model
    pipeline = importlib.import_module(selected_pipeline_module)

    # Optionally enforce the model ID inside the pipeline (not strictly needed,
    # but safe if you ever override MODEL_NAME in that file).
    if hasattr(pipeline, "MODEL_NAME"):
        pipeline.MODEL_NAME = selected_model_id

    key_set = bool(os.getenv("OPENROUTER_API_KEY"))
    st.caption(
        f"OpenRouter key: {'✅ detected' if key_set else '❌ not set in environment'}"
    )

    run_button = st.button("🔍 Run LLM retrieval for this question", type="primary")


selected_q = qid_to_question[selected_label]
selected_text = selected_q["question_text"]


# ---------------- MAIN 2-COLUMN LAYOUT ----------------

col_left, col_right = st.columns([2, 4])

# LEFT – question
with col_left:
    st.markdown("#### Question (full text)")
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{selected_q['question_id']}</div>
            <div class="card-question-text">
                {selected_text.replace('\n', '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# RIGHT – relationships
with col_right:
    st.markdown("#### Relationships (retrieved from HSNKB)")

    if run_button:
        with st.spinner(
            f"Calling {selected_model_label} via OpenRouter using {selected_pipeline_module}…"
        ):
            # Use the selected pipeline module for retrieval
            uids = pipeline.retrieve_ranked_uids_for_question(selected_text)
            result_df = pipeline.get_rows_for_uids(uids)

        if not uids:
            st.info("No Unique_IDs were returned by the model for this question.")
        else:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">
                        Retrieved {len(uids)} Unique_IDs
                    </div>
                    <div class="card-subtitle">
                        Rows from <code>fixed_mappings_with_nls_new.csv</code>
                        (text wrapped; <code>natural_language_sentence</code> hidden).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            preferred_cols = [
                "Type",
                "provision",
                "actorholder",
                "action",
                "actoraffected",
                "Treaty",
                "Unique_ID",
            ]
            cols_to_show = [c for c in preferred_cols if c in result_df.columns]

            extra_cols = [
                c
                for c in result_df.columns
                if c not in cols_to_show and c != "natural_language_sentence"
            ]
            cols_to_show.extend(extra_cols)

            html_table = (
                result_df[cols_to_show]
                .to_html(index=False, escape=True, classes="rel-table")
            )

            st.markdown(
                f"""
                <div class="rel-table-container">
                    {html_table}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Click **“Run LLM retrieval for this question”** to see relationships.")
