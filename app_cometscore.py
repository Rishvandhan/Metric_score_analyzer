"""
Neural Metrics Score Explorer
------------------------------
Compares QE metric outputs (COMET / MetricX / GEMBA-MQM / etc.) across
languages and models. Core question answered: does a metric correctly
discriminate quality tiers (good > bad > very bad > worst) per sample?

File naming convention expected:  {Model}_{Lang}_results_qe.json
Example: MetricX24_French_results_qe.json

JSON schema (list of samples):
[
  {
    "source": "...",
    "reference": "...",
    "hypothesis": {
        "good hyp":      {"text": "...", "score": 0.98},
        "bad hyp":       {"text": "...", "score": 0.96},
        "very bad hyp":  {"text": "...", "score": 0.91},
        "worst hyp":     {"text": "...", "score": 0.55}
    }
  },
  ...
]
"""

import json
import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = "results"          # optional folder auto-scanned on load; skipped if absent
FILENAME_PATTERN = re.compile(r"^(.+?)_(.+?)_results_qe\.json$", re.IGNORECASE)

TIER_KEYS = ["good hyp", "bad hyp", "very bad hyp", "worst hyp"]
TIER_LABELS = ["good", "bad", "very bad", "worst"]
TIER_ORDER = TIER_LABELS  # display order for charts

BOUNDARY_DEFS = [
    ("good_gt_bad", "good_score", "bad_score", "good > bad"),
    ("bad_gt_verybad", "bad_score", "verybad_score", "bad > very bad"),
    ("verybad_gt_worst", "verybad_score", "worst_score", "very bad > worst"),
]

st.set_page_config(page_title="Neural Metrics Score Explorer", layout="wide")
st.title("Neural Metrics Score Explorer")
st.caption(
    "Upload result JSON files (or drop them in a `results/` folder next to the app) "
    "to compare how well each metric discriminates quality tiers, across models and languages."
)

# ============================================================
# Helpers
# ============================================================
def parse_filename(fname: str):
    m = FILENAME_PATTERN.match(fname)
    if m:
        return m.group(1), m.group(2)
    return None, None


def flatten_file(fname, model, lang, data):
    rows = []
    for i, sample in enumerate(data):
        hyp = sample.get("hypothesis", {})
        row = {
            "file": fname,
            "model": model,
            "lang": lang,
            "sample_idx": i,
            "source": sample.get("source", ""),
            "reference": sample.get("reference", ""),
        }
        ok = True
        for key, label in zip(TIER_KEYS, TIER_LABELS):
            entry = hyp.get(key)
            if entry is None:
                ok = False
                break
            row[f"{label.replace(' ', '')}_score"] = entry.get("score")
            row[f"{label.replace(' ', '')}_text"] = entry.get("text")
        if ok:
            rows.append(row)
    return rows


def load_json_from_path(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_from_upload(uploaded_file):
    return json.load(uploaded_file)


# ============================================================
# Gather candidate files: folder scan + manual upload
# ============================================================
candidates = []  # list of dicts: {name, source: 'folder'/'upload', ref}

if os.path.isdir(RESULTS_DIR):
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.lower().endswith(".json"):
            candidates.append({"name": fn, "source": "folder", "ref": os.path.join(RESULTS_DIR, fn)})

uploaded_files = st.file_uploader(
    "Upload result JSON files", type=["json"], accept_multiple_files=True
)
if uploaded_files:
    for f in uploaded_files:
        candidates.append({"name": f.name, "source": "upload", "ref": f})

if not candidates:
    st.info(f"Upload JSON files, or place them in a `{RESULTS_DIR}/` folder next to this app, to begin.")
    st.stop()

# ============================================================
# Resolve model/lang per file (auto-parse, else manual entry)
# ============================================================
st.subheader("File labeling")
all_rows = []
load_errors = []
unresolved = []

for c in candidates:
    fname = c["name"]
    try:
        data = load_json_from_path(c["ref"]) if c["source"] == "folder" else load_json_from_upload(c["ref"])
    except Exception as e:
        load_errors.append(f"**{fname}**: failed to parse JSON ({e})")
        continue

    if not isinstance(data, list):
        load_errors.append(f"**{fname}**: expected a list of samples.")
        continue

    model, lang = parse_filename(fname)
    if model and lang:
        all_rows.extend(flatten_file(fname, model, lang, data))
    else:
        unresolved.append((fname, data))

if unresolved:
    st.warning(
        f"{len(unresolved)} file(s) don't match the `Model_Lang_results_qe.json` naming convention. "
        "Label them manually below."
    )
    for fname, data in unresolved:
        c1, c2 = st.columns(2)
        model = c1.text_input(f"Model for `{fname}`", key=f"model_{fname}")
        lang = c2.text_input(f"Language for `{fname}`", key=f"lang_{fname}")
        if model and lang:
            all_rows.extend(flatten_file(fname, model, lang, data))

for err in load_errors:
    st.error(err)

if not all_rows:
    st.info("No labeled files yet. Fill in model/language above, or check your uploads.")
    st.stop()

df = pd.DataFrame(all_rows)

# ============================================================
# Compute pairwise discrimination + margins
# ============================================================
for col_name, hi, lo, _label in BOUNDARY_DEFS:
    df[col_name] = df[hi] > df[lo]
    df[col_name.replace("_gt_", "_margin_")] = df[hi] - df[lo]

df["fully_monotonic"] = df[[b[0] for b in BOUNDARY_DEFS]].all(axis=1)

st.success(f"Loaded {df['file'].nunique()} file(s) — {df['model'].nunique()} model(s), {df['lang'].nunique()} language(s), {len(df)} total samples.")

# ============================================================
# Leaderboard aggregation
# ============================================================
agg_dict = {"sample_idx": "count", "fully_monotonic": "mean"}
for col_name, *_ in BOUNDARY_DEFS:
    agg_dict[col_name] = "mean"
for col_name, *_ in BOUNDARY_DEFS:
    margin_col = col_name.replace("_gt_", "_margin_")
    agg_dict[margin_col] = "mean"

leaderboard = df.groupby(["model", "lang"]).agg(agg_dict).reset_index()
leaderboard = leaderboard.rename(columns={"sample_idx": "n_samples", "fully_monotonic": "pct_fully_monotonic"})
pct_cols = ["pct_fully_monotonic"] + [b[0] for b in BOUNDARY_DEFS]
for c in pct_cols:
    leaderboard[c] = (leaderboard[c] * 100).round(1)
margin_cols = [b[0].replace("_gt_", "_margin_") for b in BOUNDARY_DEFS]
for c in margin_cols:
    leaderboard[c] = leaderboard[c].round(4)

leaderboard = leaderboard.sort_values("pct_fully_monotonic", ascending=False)

# ============================================================
# Tabs
# ============================================================
tab_lead, tab_heat, tab_pair, tab_raw, tab_drill = st.tabs(
    ["🏆 Leaderboard", "🔥 Heatmap", "📐 Pairwise Breakdown", "📊 Raw Scores", "🔍 Sample Drill-down"]
)

# ---------------- Leaderboard ----------------
with tab_lead:
    st.subheader("Discrimination leaderboard (model × language)")
    st.caption(
        "`pct_fully_monotonic` = % of samples where the metric correctly ranked "
        "good > bad > very bad > worst end-to-end. Higher is better."
    )
    lang_filter = st.multiselect(
        "Filter language(s)", sorted(df["lang"].unique()), default=sorted(df["lang"].unique()), key="lead_lang"
    )
    display_df = leaderboard[leaderboard["lang"].isin(lang_filter)]
    display_cols = ["model", "lang", "n_samples", "pct_fully_monotonic"] + \
                    [b[0] for b in BOUNDARY_DEFS] + margin_cols
    st.dataframe(
        display_df[display_cols].rename(columns={
            "good_gt_bad": "% good>bad", "bad_gt_verybad": "% bad>verybad", "verybad_gt_worst": "% verybad>worst",
            "good_margin_bad": "avg margin good-bad", "bad_margin_verybad": "avg margin bad-verybad",
            "verybad_margin_worst": "avg margin verybad-worst"
        }),
        use_container_width=True, hide_index=True
    )

# ---------------- Heatmap ----------------
with tab_heat:
    st.subheader("Discrimination score heatmap")
    pivot = leaderboard.pivot(index="model", columns="lang", values="pct_fully_monotonic")
    fig_heat = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(color="% fully monotonic"),
        title="% Fully Monotonic Samples — Model × Language",
    )
    fig_heat.update_layout(height=max(350, 60 * len(pivot)))
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------------- Pairwise Breakdown ----------------
with tab_pair:
    st.subheader("Which tier-boundary does each metric struggle with?")
    lang_choice = st.selectbox("Language", ["All"] + sorted(df["lang"].unique()), key="pair_lang")
    sub = leaderboard if lang_choice == "All" else leaderboard[leaderboard["lang"] == lang_choice]
    if lang_choice == "All":
        sub = sub.groupby("model")[[b[0] for b in BOUNDARY_DEFS]].mean().reset_index()

    melt = sub.melt(
        id_vars="model", value_vars=[b[0] for b in BOUNDARY_DEFS],
        var_name="boundary", value_name="pct_correct"
    )
    boundary_label_map = {b[0]: b[3] for b in BOUNDARY_DEFS}
    melt["boundary"] = melt["boundary"].map(boundary_label_map)

    fig_pair = px.bar(
        melt, x="boundary", y="pct_correct", color="model", barmode="group",
        text="pct_correct",
        title=f"Boundary accuracy — {lang_choice}",
        category_orders={"boundary": [b[3] for b in BOUNDARY_DEFS]},
    )
    fig_pair.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_pair.update_layout(yaxis_range=[0, 105], yaxis_title="% correct", height=500)
    st.plotly_chart(fig_pair, use_container_width=True)

# ---------------- Raw Scores ----------------
with tab_raw:
    st.subheader("Average raw score per tier")
    lang_choice_raw = st.selectbox("Language", ["All"] + sorted(df["lang"].unique()), key="raw_lang")
    sub_df = df if lang_choice_raw == "All" else df[df["lang"] == lang_choice_raw]

    score_cols = [f"{lbl.replace(' ', '')}_score" for lbl in TIER_LABELS]
    avg = sub_df.groupby("model")[score_cols].mean().reset_index()
    melt_raw = avg.melt(id_vars="model", value_vars=score_cols, var_name="tier_col", value_name="avg_score")
    tier_map = dict(zip(score_cols, TIER_LABELS))
    melt_raw["tier"] = melt_raw["tier_col"].map(tier_map)

    fig_raw = px.bar(
        melt_raw, x="tier", y="avg_score", color="model", barmode="group",
        text="avg_score", title=f"Average score by tier — {lang_choice_raw}",
        category_orders={"tier": TIER_ORDER},
    )
    fig_raw.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_raw.update_layout(yaxis_range=[0, 1], height=500)
    st.plotly_chart(fig_raw, use_container_width=True)

# ---------------- Sample Drill-down ----------------
with tab_drill:
    st.subheader("Inspect individual samples")
    c1, c2 = st.columns(2)
    model_sel = c1.selectbox("Model", sorted(df["model"].unique()), key="drill_model")
    lang_sel = c2.selectbox("Language", sorted(df[df["model"] == model_sel]["lang"].unique()), key="drill_lang")

    sub = df[(df["model"] == model_sel) & (df["lang"] == lang_sel)].reset_index(drop=True)

    only_failed = st.checkbox("Show only samples that break monotonicity", value=False)
    view = sub[~sub["fully_monotonic"]] if only_failed else sub

    if view.empty:
        st.info("No samples match this filter.")
    else:
        options = [f"{'❌' if not r.fully_monotonic else '✅'} Sample {r.sample_idx + 1}: {r.source[:70]}"
                   for r in view.itertuples()]
        idx = st.selectbox("Select sample", range(len(options)), format_func=lambda i: options[i])
        sample = view.iloc[idx]

        st.info(f"**Source:** {sample['source']}")
        st.success(f"**Reference:** {sample['reference']}")

        rows = []
        for lbl in TIER_LABELS:
            key = lbl.replace(" ", "")
            rows.append({"Tier": lbl, "Score": sample[f"{key}_score"], "Translation": sample[f"{key}_text"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**Boundary checks:**")
        cols = st.columns(3)
        for i, (col_name, hi, lo, label) in enumerate(BOUNDARY_DEFS):
            ok = sample[col_name]
            margin = sample[col_name.replace("_gt_", "_margin_")]
            cols[i].metric(label, "✅ pass" if ok else "❌ fail", f"margin {margin:.4f}")