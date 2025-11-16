# app.py
"""
Simple Treaty Actor Browser – v1
- Uses your existing pipeline code in pipeline.py
- Select question from CSV (15_questions_ground_truth.csv)
- Calls LLM via OpenRouter at run time
- Shows retrieved HSNKB rows

Run locally with:
    streamlit run app.py
"""

import os
import textwrap

import streamlit as st

# 🔁 Import your existing code (keep your file name as pipeline.py, or change import)
import pipeline

# We use:
# - pipeline.questions           -> list[dict] with question_id, question_text, gold_unique_ids
# - pipeline.actors_df           -> DataFrame with actor_name, actor_definition
# - pipeline.retrieve_ranked_uids_for_question(question_text) -> list[str]
# - pipeline.get_rows_for_uids(uids) -> DataFrame with HSNKB rows
# - pipeline.MODEL_NAME          -> name of current OpenRouter model


# ---------------- STREAMLIT BASIC CONFIG ----------------

st.set_page_config(
    page_title="Treaty Actor Browser",
    layout="wide",
)

# Small CSS to get a darker look and card-like panels
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
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #9ca9ff;
        margin-bottom: 0.35rem;
    }
    .actor-card {
        margin-bottom: 0.6rem;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Treaty Actor Browser")

# ---------------- SIDEBAR: MODEL SELECTION ----------------

st.sidebar.header("Settings")

# Map pretty labels to OpenRouter model IDs
MODEL_OPTIONS = {
    "GPT-5 (openai/gpt-5)": "openai/gpt-5",
    "Gemini 2.5 Pro (google/gemini-2.5-pro-exp)": "google/gpt-4.1-mini",  # change if you like
    # You can add more models here later
}

selected_model_label = st.sidebar.selectbox(
    "LLM model (OpenRouter)",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
)

selected_model_id = MODEL_OPTIONS[selected_model_label]

# Assign to your global MODEL_NAME so your existing retrieval function uses it
pipeline.MODEL_NAME = selected_model_id

st.sidebar.markdown("---")
st.sidebar.write(
    f"OpenRouter key set: "
    f"{'✅' if os.getenv('OPENROUTER_API_KEY') else '❌ (set OPENROUTER_API_KEY)'}"
)

# ---------------- QUESTION SELECTION (TOP ROW) ----------------

# Build nice labels for dropdown: "Q01: During an international..."
question_labels = []
qid_to_question = {}

for q in pipeline.questions:
    qid = q["question_id"]
    qtext = q["question_text"].replace("\n", " ")
    short = textwrap.shorten(qtext, width=80, placeholder="…")
    label = f"{qid}: {short}"
    question_labels.append(label)
    qid_to_question[label] = q

st.markdown("### Question")

selected_label = st.selectbox(
    "Choose a fact-pattern question (from 15_questions_ground_truth.csv):",
    options=question_labels,
)

selected_q = qid_to_question[selected_label]
selected_text = selected_q["question_text"]

# Button to trigger LLM retrieval
run_button = st.button("🔍 Run LLM retrieval for this question", type="primary")


# ---------------- MAIN 3-COLUMN LAYOUT ----------------

col_left, col_mid, col_right = st.columns([2, 1.3, 3.2])

# LEFT COLUMN: full question text
with col_left:
    st.markdown("#### Question (full text)")
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{selected_q['question_id']}</div>
            <div>{selected_text.replace('\n', '<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# MIDDLE COLUMN: list all actors (for v1 we don’t filter by question)
with col_mid:
    st.markdown("#### Actors")
    st.markdown(
        """
        <div class="card">
            <div class="card-subtitle">
                From actor_defintions.csv (no per-question filtering yet).
                A later version can highlight only actors implicated by the LLM.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for _, row in pipeline.actors_df.iterrows():
        name = row["actor_name"]
        definition = row["actor_definition"]
        st.markdown(
            f"""
            <div class="card actor-card">
                <div class="card-title">{name}</div>
                <div class="card-subtitle">{definition}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# RIGHT COLUMN: relationships / retrieved rows
with col_right:
    st.markdown("#### Relationships (retrieved from HSNKB)")

    if run_button:
        # --------- CALL YOUR EXISTING PIPELINE HERE ---------
        with st.spinner(f"Calling {selected_model_label} via OpenRouter…"):
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
                        Showing rows from fixed_mappings_with_nls_new.csv
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # For readability, order columns the way you probably want to see them
            cols_to_show = [
                "Unique_ID",
                "Treaty",
                "provision",
                "actorholder",
                "action",
                "actoraffected",
                "natural_language_sentence",
            ]
            cols_to_show = [c for c in cols_to_show if c in result_df.columns]

            st.dataframe(
                result_df[cols_to_show],
                use_container_width=True,
            )
    else:
        st.info("Click **“Run LLM retrieval for this question”** to see relationships.")
