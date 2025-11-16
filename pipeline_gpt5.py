import os, re, json, unicodedata
import pandas as pd
import numpy as np
import tiktoken
from openai import OpenAI
import requests
from litellm import token_counter
from typing import List, Dict
from sklearn.metrics import ndcg_score, average_precision_score
from nltk.metrics.scores import precision as nltk_precision, recall as nltk_recall, f_measure as nltk_f1
from sklearn.metrics import confusion_matrix
import streamlit as st

#open router key
#os.environ["OPENROUTER_API_KEY"] = "set your key"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Set it as an environment variable locally or in Streamlit secrets when deployed."
    )

# --- Client (OpenAI-compatible) ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


MODEL_NAME = "openai/gpt-5"


MAPPINGS_CSV = "fixed_mappings_with_nls_new.csv"
ACTOR_CSV   = "actor_defintions.csv"
CSV_path_questions = "15_questions_ground_truth.csv"


# regex to extract UID-like tokens as a fallback
UID_REGEX = re.compile(r"[A-Za-z]+_[A-Za-z0-9]+")


def _parse_gold_ids_robust(raw: str) -> List[str]:
    """
    Parse `gold_unique_ids` into a list[str].
    Handles:
      - Proper JSON lists
      - Comma-separated strings with stray quotes/brackets
      - Messy strings by regex fallback
    """
    if raw is None:
        return []
    s = str(raw).strip() # if not string, then convert to string and remove trailing whitespace

    # 1) Try JSON directly
    try:
        if s.startswith("[") and s.endswith("]"):
            data = json.loads(s)
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    # 2) Try manual splitting if JSON fails
    try:
        ss = s.replace("'", '"')
        # Sometimes the whole list is double-quoted
        if ss.startswith('"') and ss.endswith('"'):
            ss = ss[1:-1]
        if ss.startswith("[") and ss.endswith("]"):
            return [
                x.strip().strip('"').strip("'")
                for x in ss[1:-1].split(",")
                if x.strip().strip('"').strip("'")
            ]
    except Exception:
        pass

    # 3) Regex fallback: just grab UID-like tokens
    matches = UID_REGEX.findall(s)
    return matches



def load_questions_csv_robust(path: str) -> List[Dict[str, object]]:
    """
    Load questions from a ';'-separated CSV with columns:
      question_id, question_text, gold_unique_ids
    Returns a list of dicts in the same format your pipeline expects:
      {"question_id": str, "question_text": str, "gold_unique_ids": List[str]}
    """
    df = pd.read_csv(path, sep=";", dtype=str) #separator is ;, reading all columns as string

    # Case-insensitive column mapping
    cols_map = {c.lower(): c for c in df.columns} #lowercase the columns 
    
    qid_col = cols_map["question_id"]
    qtext_col = cols_map["question_text"]
    gold_col = cols_map["gold_unique_ids"]

    questions: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        qid = str(row[qid_col]).strip()
        qtext = str(row[qtext_col]).strip()
        gold_list = _parse_gold_ids_robust(row[gold_col])
        
        questions.append({
            "question_id": qid,
            "question_text": qtext,
            "gold_unique_ids": gold_list,
        })
    return questions



# --------------- OPENROUTER CLIENT + LiteLLM TOKEN COUNT ---------------


# LiteLLM token_counter (model-aware)

def estimate_tokens_with_litellm(messages, model_id: str) -> int:
    """Return the number of prompt tokens as estimated by LiteLLM.
    Prompt tokens are the tokens that you input into the model. This is the number of tokens in your prompt.
    """
    return token_counter(model=model_id, messages=messages)

# --------------- DATA LOADING & NORMALIZATION ---------------
def _norm(s):
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


#reads the csv mapping file by filtering only for UID, provision and natural language sentence

def load_mappings(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["Unique_ID", "provision", "natural_language_sentence"]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    df["provision"] = df["provision"].apply(_norm)
    df["natural_language_sentence"] = df["natural_language_sentence"].apply(_norm)
    return df

# --- Load actor definitions from CSV ---

def load_actor_defs_csv(path: str) -> pd.DataFrame:
    """
    Load actor definitions from a CSV file with 2 columns:
      - actor_name
      - actor_definition
    Returns DataFrame with those two columns.
    """
    df = pd.read_csv(path, dtype=str,  sep=";")   # ensure all text
    return df


# --- Convert actor definitions into a string for the LLM ---
def actor_defs_to_text(df: pd.DataFrame) -> str:
    """
    Convert the actor definitions DataFrame into a neat text block
    to inject into the LLM prompt.
    """
    lines = []
    for _, row in df.iterrows():
        lines.append(f"{row['actor_name']}: {row['actor_definition']}")
    return "\n".join(lines)



def build_mapping_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for _, r in df.iterrows():
        uid = str(r["Unique_ID"])
        prv = r["provision"]
        nls = r["natural_language_sentence"]
        lines.append(f"[UID={uid}] (Provision: {prv}) {nls}")
    return lines



mappings_df = load_mappings(MAPPINGS_CSV)
actors_df   = load_actor_defs_csv(ACTOR_CSV)


MAPPING_LINES = build_mapping_lines(mappings_df) # list of the mapping lines (rows)

ACTORS_TEXT = actor_defs_to_text(actors_df)  
questions = load_questions_csv_robust(CSV_path_questions)




# --------------- HOHFELDIAN DEFINITIONS ---------------
HOHFELDIAN_TEXT = """Basic Legal Positions:
• Claim-Right: X has a right that Y must (or must not) do something.
• Duty/Obligation: Y is required to act (or refrain) because X has a claim-right.
• Liberty/Privilege: X may act without Y having the right to prevent it.
• No-Right: Y has no claim-right against X’s action.

Second-Order Legal Positions:
• Power: X can change legal relations (e.g., create, alter, or extinguish rights).
• Liability: Y is subject to X’s exercise of power.
• Immunity: X is protected from changes to their legal relations.
• Disability: Y lacks the power to change X’s legal relations.
"""

SYSTEM_PROMPT = """
You are an expert in international law across the Biodiversity Beyond National Jurisdiction (BBNJ) Agreement, the Convention on Biological Diversity (CBD), and the Nagoya Protocol.

Glossary of key abbreviations:
- ABNJ = Areas Beyond National Jurisdiction
- GR = Genetic Resources.
- MGR = Marine Genetic Resources.
- DSI = Digital Sequence Information
- TK = Traditional Knowledge.
- IPLC = Indigenous Peoples and Local Communities.
- PIC = Prior Informed Consent.
- MAT = Mutually Agreed Terms.
- R&D = Research and Development.

You will receive: (1) actor definitions, (2) Hohfeldian definitions, (3) a list of treaty mappings as natural-language statements with Unique_IDs, and (4) ONE fact pattern question.

TASK
Return ONLY the Unique_IDs of mapping rows that DIRECTLY allocate legal positions (duty, claim-right, power, liability, immunity, disability, liberty, no-right) to the actors implicated by the fact pattern.

INTERNAL METHOD (do not reveal notes or steps)
• Parse the fact pattern into facets: actors; jurisdiction (national jurisdiction vs ABNJ; any cross-border transfers; repositories); resource/data types (MGR/GR, derivatives, DSI); TK/IPLC provenance; lifecycle stage(s) (access, publication, utilisation/R&D, transfer/repository, monitoring/checkpoints/traceability, downstream commercialisation); PIC/MAT/benefit-sharing posture; disclosure/notification posture.
• Jurisdictional gating (apply strictly):
  – If the activity is solely within national jurisdiction, consider national-jurisdiction regimes; do NOT include ABNJ-specific rows unless the pattern explicitly involves ABNJ.
  – If the activity is solely in ABNJ, consider ABNJ-specific rows; do NOT import national-jurisdiction rows unless the pattern explicitly invokes access/TK within national jurisdiction.
  – If both zones are implicated, include both sets where each zone’s obligations are triggered.
• Allocative filter (hard): keep rows that impose or allocate concrete legal positions to identified actors (e.g., “shall/must/required to/entitled to/liable to”) for the relevant lifecycle facet. Drop rows that are purely objectives, principles, scope/definitions, general cooperation/capacity/technology-transfer, finance, or institution-building unless they impose a concrete obligation tied to the specific fact pattern.
• Specificity preference: prefer rows that explicitly match (i) the exact jurisdictional setting (national vs ABNJ), (ii) resource/data type (MGR/GR/derivatives/DSI), (iii) TK involvement, and (iv) the lifecycle facet(s) implicated. 
• Consolidate near-duplicates by keeping the most specific.

RANKING & OUTPUT
• Deduplicate and rank by: (1) directness to the fact pattern, (2) specificity to jurisdiction and resource/data type, (3) relevance to the lifecycle facet(s), (4) clarity of Hohfeldian allocation.
• Output STRICT JSON ONLY (no explanations):
{"ranked_uids": ["UID_1", "UID_2", "..."]}
• Do not invent IDs; return an empty list if nothing directly applies.
"""



USER_TEMPLATE_GLOBAL = """Actor Definitions (table):
{actors}

Hohfeldian Definitions:
{hohfeld}

Mappings (UID, provision, NLS):
{mappings}

Question:
{question}

Return ONLY strict JSON:
{{"ranked_uids": ["UID_1", "UID_2", "..."]}}
"""

# --------------- JSON PARSING HELPER 
#Extract the JSON block from LLM's output
def extract_json_dict(text: str) -> dict:
    """
    Try to extract a JSON dictionary from a string.
    Useful when the LLM response contains extra text around a JSON object.
    """
    # Step 1: Use regex to search for the first {...} block in the text
    # re.S flag makes '.' also match newlines (so JSON spanning multiple lines works)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:   # If no {...} block was found, return empty dict
        return {}
    try:  # Step 2: Try to parse the matched substring (the full {...}) as JSON
        return json.loads(m.group(0))
    except Exception:
        try:
            # Step 3 (fallback): Sometimes LLMs generate trailing commas before ]
            # Example: {"a": [1,2,], "b": 3}
            # That is invalid JSON, so we remove ", ]" → "]"
            cleaned = re.sub(r",\s*]", "]", m.group(0))
            return json.loads(cleaned)
        except Exception:
            return {}

# --------------- OPENAI CALL WRAPPER (now OpenRouter) ---------------
def call_llm(system_prompt: str, user_prompt: str) -> dict:
    """
    Call the model via OpenRouter and return parsed JSON (dict). If parsing fails, returns {}.
    """
    # Build messages for both counting and the actual call
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]

    #estimate prompt (input) tokens via LiteLLM (for this single model)
    est_tokens = estimate_tokens_with_litellm(messages, MODEL_NAME)
    #print(f"\n\n[Token estimate] {MODEL_NAME}: prompt_tokens ≈ {est_tokens}")

    # Make the actual request to LLM

    resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    extra_body={
            "transforms": [],
            "seed": 42,
            "response_format": {"type": "json_object"},
            "provider": {"allow_fallbacks": False},
            "reasoning": {"effort": "high", "exclude": True} 
        },
    )
    
    # If you want server-accounted tokens too:
    try:
        server_prompt_tokens = resp.usage.prompt_tokens #after response from the server
        print(f"\n[Server usage] prompt_tokens = {server_prompt_tokens}, completion_tokens = {resp.usage.completion_tokens}, total_tokens = {resp.usage.total_tokens}")
    except Exception:
        pass

    text = resp.choices[0].message.content
    
    #print(f"\nLLM's raw output: {text}")
    #print(f"\nLLM's output after extract_json_dict(): {extract_json_dict(text or "")}")
    return extract_json_dict(text or "")

# --------------- RETRIEVAL (PER QUESTION) — -----------
def retrieve_ranked_uids_for_question(question_text: str) -> list[str]:
    """
    Single-pass retrieval:
      - Build one prompt with actor definitions, Hohfeldian definitions,
        ALL mapping lines, and the question.
      - Return the FULL list of UIDs predicted by the model.
    Notes:
      - Suits set-based evaluation (precision/recall/F1).
    """
    user = USER_TEMPLATE_GLOBAL.format(   #fill the user template
        actors=ACTORS_TEXT,              # prebuilt text of actors and definitions
        hohfeld=HOHFELDIAN_TEXT,        # Hohfeldian glossary
        mappings="\n".join(MAPPING_LINES), #all mapping rows: UID, provision, natural language sentence
        question=question_text          # the current question/fact pattern
    )

    #print(f"\n Printing the entire user prompt: {user}")


    #here the entire prompt is passed to the LLM inlcuding system prompt + user prompt + mappings + actor defintions + hohfeld context + question
    obj = call_llm(SYSTEM_PROMPT, user) 
    ranked = obj.get("ranked_uids", []) #gets the value of the key ranked_uids . if the key is missing, it returns an empty list
    #print(f"\nRanked uids before cleaning:{ranked}")
    #print(f"\nRanked uids after cleaning:{[str(u).strip() for u in ranked if str(u).strip()]}")
    return [str(u).strip() for u in ranked if str(u).strip()]     # Normalize: cast to str and strip whitespace; drop empties




def eval_one(
    question_id: str,
    question_text: str,
    gold_unique_ids: list[str],
    ranked_pred_ids: list[str],
    do_manual_counts: bool = True,   # toggle printing TP/FP/FN/ manual metrics
):
    """
    Evaluate a single question:
      - Computes precision, recall, F1 directly on sets (via NLTK).
      - Optionally also computes TP, FP, FN, TN and manual Precision/Recall/F1
        using a confusion matrix for transparency.
    """

    
    # Normalize to sets of clean strings
    gold_set = {str(x).strip() for x in gold_unique_ids if str(x).strip()}
    pred_set = {str(x).strip() for x in ranked_pred_ids if str(x).strip()}

    # Metrics directly from NLTK
    # NLTK returns None when a metric is undefined (e.g., no retrieved items → precision undefined).
    # The `or 0.0` turns None into 0.0 so downstream code stays numeric and printable.
    p_nltk = nltk_precision(gold_set, pred_set) or 0.0
    r_nltk = nltk_recall(gold_set, pred_set) or 0.0
    f1_nltk = nltk_f1(gold_set, pred_set) or 0.0

    # ---- Optional: TP, FP, FN, TN counts + manual metrics ----
    
    tp = fp = fn = tn = 0
    p_manual = r_manual = f1_manual = 0.0
    if do_manual_counts:
        # "universe" = union of gold and predicted IDs
        U = list(gold_set | pred_set)

        # Ground-truth and prediction boolean vectors over U
        y_true = [u in gold_set for u in U]
        y_pred = [u in pred_set for u in U]

        # Confusion matrix → [TN, FP, FN, TP]
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[False, True]
        ).ravel()

        # Manual metrics from counts (safe divisions)
        p_manual = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_manual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_manual = (
            2 * p_manual * r_manual / (p_manual + r_manual)
            if (p_manual + r_manual) > 0 else 0.0
        )
   
    # ---- Print results ----
    print(f"\n=== {question_id} ===")
    print(f"Question: {question_text}")
    print(f"Gold ({len(gold_set)}): {sorted(gold_set)}")
    print(f"Preds({len(pred_set)}): {sorted(pred_set)}")

    # NLTK (set-based) metrics
    print("\n[NLTK metrics]")
    print(f" Precision={p_nltk:.3f}, Recall={r_nltk:.3f}, F1={f1_nltk:.3f}")
    
    # Manual (from TP/FP/FN) metrics + counts
    if do_manual_counts:
        print("\n[Manual metrics from TP/FP/FN]")
        print(f" Precision={p_manual:.3f}, Recall={r_manual:.3f}, F1={f1_manual:.3f}")
        print(f" Counts -> TP={tp}, FP={fp}, FN={fn}")
    
    # ---- Return results in a dict (row for DataFrame) ----
    

    return {
        "question_id": question_id,
        "Precision": p_nltk,
        "Recall": r_nltk,
        "F1": f1_nltk
    }




def run(questions, do_manual_counts: bool = True):
    """
    Orchestration layer for the whole pipeline:
      1) For each question:
         - Retrieve predicted UIDs (via retrieve_ranked_uids_for_question).
         - Evaluate predictions against gold (via eval_one).
      2) Collect results into a DataFrame.
      3) Print macro-average Precision/Recall/F1 across all questions.
    """

    rows = []

    print("=== PIPELINE START ===")
    for q in questions:
        print("\n--- Processing Question", q["question_id"], "---")

        # Step 1: Retrieval
        print("\n[Step 1] Retrieving predicted UIDs...")
        preds = retrieve_ranked_uids_for_question(q["question_text"])

        # Step 2: Evaluation
        print("\n[Step 2] Evaluating predictions...")
        row = eval_one(
            q["question_id"],
            q["question_text"],
            q["gold_unique_ids"],
            preds,
            do_manual_counts=do_manual_counts,
        )
        rows.append(row)

    # Step 3: Aggregate results
    print("\n=== Aggregating results into DataFrame ===")
    df = pd.DataFrame(rows).set_index("question_id")

    # Step 4: Macro-average
    macro = df[["Precision", "Recall", "F1"]].mean().to_dict()
    print("\n=== MACRO AVERAGE over questions ===")
    print(f"Precision={macro['Precision']:.3f}, "
          f"Recall={macro['Recall']:.3f}, "
          f"F1={macro['F1']:.3f}")

    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df)

    print("\n=== PIPELINE END ===")
    return df


# run the whole thing:
#results_df = run(questions, do_manual_counts=True)


# --------------- SMALL HELPER FOR THE WEB APP ---------------

def get_rows_for_uids(uids: list[str]) -> pd.DataFrame:
    """
    Given a list of Unique_IDs, return the corresponding rows from mappings_df.
    If the list is empty, return an empty DataFrame with the same columns.
    """
    if not uids:
        return mappings_df.iloc[0:0].copy()
    return mappings_df[mappings_df["Unique_ID"].isin(uids)].copy()


