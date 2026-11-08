import json
import time
import ollama

from collections import Counter
from utils.llm_config import (
    MODEL_NAME, OLLAMA_MODEL,
    TAXONOMY_MODEL_OPTIONS, ANNOTATION_MODEL_OPTIONS, AGNOSTIC_LABEL_CAP,
    MIN_LARGETAX_SUPPORT, DEV_MODE, TAXONOMY_6, BANNED_EMOTIONS, 
    log,COND_AGNOSTIC_SMALL, COND_LARGE, COND_SMALL, COND_AGNOSTIC_SMALL,
    PER_HUMAN_CONDITIONS
)


def normalize_annotator_name(name):
    name = str(name)
    name = name.replace("Annotator ", "Annotator_")
    name = name.replace("AriEmozione 1", "AriEmozione")
    return name

def normalize_output_rows(rows, *, drop_llama=True, drop_primitive=True):
    normalized = []
    for row in rows:
        analysis_type = str(row.get("Analysis_Type", ""))
        model = str(row.get("Model", ""))

        # Mirrors notebook filtering done after pivoting.
        if drop_llama and "llama" in model.lower():
            continue
        if drop_primitive and "Primitive" in analysis_type:
            continue

        row_out = dict(row)
        row_out["Analysis_Type"] = analysis_type
        row_out["Annotator_Source"] = normalize_annotator_name(row_out.get("Annotator_Source", ""))
        normalized.append(row_out)

    return normalized

# ============================================================================
# OLLAMA CALL (ROBUST JSON)
# ============================================================================

def ollama_call(system, user, *, condition="", aria_idx=None, iteration=None, is_taxonomy=False):
    if DEV_MODE:
        if "Generate EXACTLY" in user:
            n = int(user.split("Generate EXACTLY ")[1].split(" ")[0])
            return {"labels": [f"emotion_{i}" for i in range(n)]}
        if "Choose EXACTLY ONE emotion from this list:" in user:
            import re, random, ast
            match = re.search(r"Choose EXACTLY ONE emotion from this list:\n(\[.*?\])", user, re.DOTALL)
            if match:
                taxonomy = ast.literal_eval(match.group(1))
                return {"id": "x", "emotion": random.choice(taxonomy), "confidence": 0.9}
        return {"id": "x", "emotion": "grief", "confidence": 0.9}

    options = TAXONOMY_MODEL_OPTIONS if is_taxonomy else ANNOTATION_MODEL_OPTIONS

    tag = f"[{MODEL_NAME}]"
    if condition:
        tag += f" [{condition}]"
    if iteration is not None:
        tag += f" [iter {iteration}]"
    if aria_idx is not None:
        tag += f" [aria {aria_idx}]"

    start = time.time()
    log(f"    {tag} START")

    r = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        options=options,
        stream=False,
    )

    elapsed = time.time() - start
    log(f"    {tag} DONE ({elapsed:.1f}s)")

    try:
        return json.loads(r["message"]["content"])
    except Exception as e:
        log(f"    {tag} JSON ERROR — skipped ({e})")
        return None

# ============================================================================
# DATA
# ============================================================================

def get_arias(df):
    """
    Expects your df to have:
      Prefix, opera, aria_name, section_a
    """
    rows = df[df["section_a"].notna()]
    arias = []
    for _, r in rows.iterrows():
        arias.append({
            "id": str(r["Prefix"]),
            "text": r["section_a"],
            "opera": r.get("opera", ""),
            "aria_name": r.get("aria_name", ""),
        })
    return arias

# ============================================================================
# ANNOTATION (SINGLE-ARIA; STABLE)
# ============================================================================

def annotate_aria(a, taxonomy, condition, aria_idx, iteration=None):
    """
    One aria per call (stable JSON behavior across models).
    Returns dict or None.
    """
    # Normalize taxonomy to lowercase strings for membership checks
    taxonomy_lc = [str(t).strip().lower() for t in taxonomy if str(t).strip()]
    if not taxonomy_lc:
        return None

    taxonomy_str = "\n".join(f"{i+1}. {e}" for i, e in enumerate(taxonomy_lc))

    prompt = (
        f"OPERA: {str(a['opera'])}\n"
        f"ARIA TEXT: {str(a['text'])}\n\n"
        f"Choose EXACTLY ONE emotion from this numbered list:\n{taxonomy_str}\n\n"
        "Reply with ONLY this JSON:\n"
        '{"id": "...", "emotion": "...", "confidence": 0.0}'
    )
    
    out = ollama_call(
        system="You are an expert emotion annotator in opera. Follow the instructions exactly.",
        user=prompt,
        condition=condition,
        aria_idx=aria_idx,
        iteration=iteration,
    )

    if out is None or not isinstance(out, dict):
        return None

    emo = str(out.get("emotion", taxonomy_lc[0])).strip().lower()
    if emo not in taxonomy_lc:
        emo = taxonomy_lc[0]

    # Confidence can be absent or malformed; keep as None if so
    conf = out.get("confidence", None)
    try:
        conf = float(conf) if conf is not None else None
    except Exception:
        conf = None

    return {
        "id": a["id"],
        "emotion": emo,
        "confidence": conf,
        "Analysis_Type": condition,
    }


def run_condition(arias, taxonomy, condition):
    log(f"\n[START] {condition} | taxonomy size = {len(taxonomy)}")
    results = []
    n = len(arias)

    for i, a in enumerate(arias, 1):
        if i == 1 or i % 10 == 0 or i == n:
            log(f"  [{MODEL_NAME}] {condition} aria {i}/{n}")
        r = annotate_aria(a, taxonomy, condition, i)
        if r is not None:
            results.append(r)

    log(f"[DONE] {condition} | produced {len(results)}/{n} labels")
    return results

# ============================================================================
# AGNOSTIC TAXONOMY GENERATION (ROBUST + FALLBACK)
# ============================================================================

def generate_agnostic_taxonomy(arias, n_labels, *, max_retries=2):
    n_labels = int(n_labels)
    if n_labels <= 0:
        raise ValueError("n_labels must be > 0")

    request_n = n_labels + len(BANNED_EMOTIONS)  # Fix 1: over-request


    sample = "\n".join(
        f"- {str(a['text'])[:150].replace(chr(10), ' ').replace(';', ',')}"
        for a in arias[:10]
    )

    prompt = (
        '{"labels": ["'  # force the model to start mid-JSON
        + '", "'.join([f"LABEL_{i}" for i in range(request_n)])  # placeholder structure
        + '"]}\n\n'
        f"Replace the LABEL_N placeholders above with {request_n} DISTINCT English emotion words.\n"
        "Rules: emotional states only, 1-3 words, lowercase, no duplicates.\n"
        f"Do NOT use: admiration, anger, fear, joy, love, sadness.\n"
        "Output ONLY the completed JSON object, nothing else.\n\n"
        f"Context texts:\n{sample}"
    )

    candidates = []  # Fix 2: collect best result across attempts

    for attempt in range(1, max_retries + 1):
        log(f"[AGNOSTIC] Attempt {attempt}/{max_retries}")

        raw = ollama_call(
            system="You generate emotion taxonomies for academic research.",
            user=prompt,
            condition=f"AgnosticTaxonomyGen_attempt{attempt}",
            is_taxonomy=True,  
        )

        # Unwrap dict → list
        if isinstance(raw, dict):
            for key in ("labels", "emotions", "taxonomy", "result", "items"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            else:
                for v in raw.values():
                    if isinstance(v, list):
                        raw = v
                        break

        if not isinstance(raw, list):
            log("[AGNOSTIC] ❌ Not a JSON list")
            continue

        clean = []
        for l in raw:
            l = str(l).strip().lower()
            if not l or l in BANNED_EMOTIONS or l in clean:
                continue
            clean.append(l)
            if len(clean) == n_labels:  # stop once we have enough
                break

        candidates.append(clean)  # always save, even if short

        if len(clean) == n_labels:
            log(f"[AGNOSTIC] ✅ Success: {clean}")
            return clean

        log(f"[AGNOSTIC] ❌ Got {len(clean)}/{n_labels} valid labels")

    # Fix 2: fall back to best result if close enough
    best = max(candidates, key=len) if candidates else []
    if len(best) >= max(2, int(n_labels * 0.9)):
        log(f"[AGNOSTIC] ⚠️  Using best result: {len(best)}/{n_labels} labels")
        return best

    raise RuntimeError(
        f"Agnostic taxonomy generation FAILED after {max_retries} attempts "
        f"(required {n_labels} labels, best was {len(best)}). Aborting run."
    )

# ============================================================================
# ITERATIVE LLM-DESCARTES (MIN SUPPORT GUARANTEE: Not used in paper: MIN_LARGETAX_SUPPORT = 1)
# ============================================================================

def iterative_large_tax(arias, initial_results):
    """
    Fix 2: repeatedly prune labels with < MIN_LARGETAX_SUPPORT support and re-annotate
    until all labels satisfy minimum support or we fall back.
    """
    log(f"\n[FIX 2] Iterative {COND_LARGE} consolidation started")

    current = list(initial_results)
    if not current:
        log("[WARN] LLM-Descartes_Initial produced no labels; falling back to PrimitiveTax")
        return run_condition(arias, TAXONOMY_6, COND_LARGE)

    for iteration in range(1, 11):
        counts = Counter(r["emotion"] for r in current if "emotion" in r)
        top10 = counts.most_common(10)
        log(f"[{MODEL_NAME}] Iter {iteration} label counts (top 10): {top10}")

        valid = [lab for lab, c in counts.items() if c >= MIN_LARGETAX_SUPPORT]
        log(f"[{MODEL_NAME}] Iter {iteration} surviving labels: {len(valid)}")

        if len(valid) < 2:
            log("[WARN] Too few labels survive; fallback to PrimitiveTax")
            return run_condition(arias, TAXONOMY_6, COND_LARGE)

        # stop if all valid labels already meet support (they do by construction)
        # but we still want to ensure that re-annotation won't create new singletons:
        # we perform at most one consolidation pass if iteration==1 and valid changed.
        # A practical stopping rule: if valid set is unchanged across iterations.
        if iteration > 1:
            prev_counts = Counter(r["emotion"] for r in current)
            prev_valid = [lab for lab, c in prev_counts.items() if c >= MIN_LARGETAX_SUPPORT]
            if set(prev_valid) == set(valid):
                log("[STOP] Valid label set stabilized")
                return current

        # Re-annotate ALL arias with the reduced label set
        new_results = []
        for i, a in enumerate(arias, 1):
            if i == 1 or i % 25 == 0 or i == len(arias):
                log(f"  [{MODEL_NAME}] {COND_LARGE} iter {iteration} aria {i}/{len(arias)}")
            r = annotate_aria(a, valid, COND_LARGE, i, iteration=iteration)
            if r is not None:
                new_results.append(r)

        current = new_results

        # Hard stop if something went wrong
        if not current:
            log(f"[WARN] {COND_LARGE} re-annotation produced no labels; fallback to PrimitiveTax")
            return run_condition(arias, TAXONOMY_6, COND_LARGE)

    log(f"[WARN] Max iterations reached; returning last {COND_LARGE}")
    return current

def preflight_check(schemes_to_run, df, arias):
    log("\n[PREFLIGHT] Testing taxonomy generation...")
    for scheme in schemes_to_run:
        human_labels = df[scheme].dropna().astype(str).str.strip().str.lower().unique().tolist()
        n = len(human_labels)
        cap = AGNOSTIC_LABEL_CAP.get(MODEL_NAME, 999)
        n_labels = min(len(human_labels), cap)
        
        if n_labels < len(human_labels):
            log(f"[WARN] {COND_AGNOSTIC_SMALL}: capping {len(human_labels)} → {n_labels} labels for {MODEL_NAME}")
            n = n_labels
        log(f"[PREFLIGHT] {scheme}: {n} labels required")
        try:
            tax = generate_agnostic_taxonomy(arias, n_labels=n, max_retries=1)
            log(f"[PREFLIGHT] ✅ {scheme}: OK ({len(tax)} labels)")
        except RuntimeError as e:
            log(f"[PREFLIGHT] ❌ {scheme}: WILL FAIL — {e}")
            raise  # abort before wasting time on annotation




def make_wide_predictions(df, model_tag, *, per_human_conditions=PER_HUMAN_CONDITIONS,
                          meta_cols=("Aria_Name", "Original_Text")):
    df = df.copy()

    # --- Normalize core column names ---
    if "Prefix_ID" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "Prefix_ID"})
        else:
            raise ValueError("Missing Prefix_ID (or id) column.")

    emotion_col = None
    if "emotion" in df.columns:
        emotion_col = "emotion"
    elif "Emotion" in df.columns:
        emotion_col = "Emotion"
    else:
        candidates = [c for c in df.columns if c.lower().startswith("emotion")]
        if candidates:
            emotion_col = candidates[0]
        else:
            raise ValueError("Missing emotion/Emotion column.")

    model_emotion_col = f"Emotion_{model_tag}"
    if emotion_col != model_emotion_col:
        df = df.rename(columns={emotion_col: model_emotion_col})

    # --- Ensure required columns ---
    if "Analysis_Type" not in df.columns:
        raise ValueError("Missing Analysis_Type column.")

    if "Annotator_Source" not in df.columns:
        df["Annotator_Source"] = "GLOBAL"

    # --- Build disambiguated condition key ---
    df["Effective_Condition"] = df.apply(
        lambda r: f"{r['Analysis_Type']}_{r['Annotator_Source']}"
        if r["Analysis_Type"] in per_human_conditions
        else r["Analysis_Type"],
        axis=1,
    )

    # --- Pivot to wide ---
    wide = df.pivot_table(
        index="Prefix_ID",
        columns="Effective_Condition",
        values=model_emotion_col,
        aggfunc="first",
    )

    # --- Suffix columns with model tag ---
    wide = wide.rename(columns={c: f"{c}_{model_tag}" for c in wide.columns})

    # --- Attach metadata columns (one value per Prefix_ID) ---
    present_meta = [c for c in meta_cols if c in df.columns]
    if present_meta:
        meta = (df[["Prefix_ID", *present_meta]]
                .drop_duplicates(subset="Prefix_ID")
                .set_index("Prefix_ID"))
        wide = meta.join(wide)   # meta first, then prediction columns

    return wide