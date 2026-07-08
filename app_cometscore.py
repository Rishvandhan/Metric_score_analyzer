import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Neural Metrics Score Explorer",
    layout="wide"
)

st.title("Neural Metrics Score Explorer")
st.write(
    "Upload up to **4 JSON files** to compare scores across different models/runs. "
    "Each file should contain source sentences, reference translations, "
    "hypotheses, and scores."
)

# ==========================================================
# Multi-file upload
# ==========================================================
uploaded_files = st.file_uploader(
    "Choose JSON files (max 4)",
    type=["json"],
    accept_multiple_files=True
)

if uploaded_files:
    # Limit to 4 files
    if len(uploaded_files) > 4:
        st.warning("⚠️ Maximum 4 files allowed. Only the first 4 will be used.")
        uploaded_files = uploaded_files[:4]

    # Load all files
    all_data = {}  # {filename: data_list}
    load_errors = []

    for f in uploaded_files:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                load_errors.append(f"❌ **{f.name}**: JSON format should be a list of samples.")
                continue
            all_data[f.name] = data
        except Exception as e:
            load_errors.append(f"❌ **{f.name}**: {e}")

    if load_errors:
        for err in load_errors:
            st.error(err)

    if not all_data:
        st.info("No valid files loaded. Please upload valid JSON files.")
        st.stop()

    # ==========================================================
    # Compute average scores per file per hypothesis type
    # ==========================================================
    file_avg_scores = {}  # {filename: {hyp_type: avg_score}}

    for fname, data in all_data.items():
        hyp_scores = {}
        for sample in data:
            for hyp_type, hyp_data in sample["hypothesis"].items():
                hyp_scores.setdefault(hyp_type, []).append(hyp_data["score"])
        file_avg_scores[fname] = {
            hyp: sum(scores) / len(scores)
            for hyp, scores in hyp_scores.items()
        }

    # ==========================================================
    # Comparison: Grouped Bar Chart
    # ==========================================================
    st.subheader("📊 Average Score Comparison Across Files")

    # Build long-form DataFrame for grouped bar chart
    chart_rows = []
    for fname, hyp_dict in file_avg_scores.items():
        for hyp_type, avg_score in hyp_dict.items():
            chart_rows.append({
                "Hypothesis Type": hyp_type,
                "File": fname,
                "Average Score": avg_score
            })

    chart_df = pd.DataFrame(chart_rows)

    # Apply custom category ordering to the x-axis
    chart_df["Hypothesis Type"] = pd.Categorical(
        chart_df["Hypothesis Type"],
        categories=desired_order,
        ordered=True
    )

    fig_comp = px.bar(
        chart_df,
        x="Hypothesis Type",
        y="Average Score",
        color="File",
        barmode="group",
        text="Average Score",
        title="Average Score by Hypothesis Type — Grouped by File",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig_comp.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_comp.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="Hypothesis Type",
        yaxis_title="Average Score",
        height=500,
        legend_title="File"
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    # ==========================================================
    # Comparison: Table
    # ==========================================================
    st.subheader("📋 Comparison Table (Average Scores)")

    # Build a pivot table: rows=hyp_type, columns=filename
    comparison_data = {}

    # Custom ordering: good → bad → very bad → worst
    desired_order = ["good", "bad", "very bad", "worst"]
    all_hyp_types = sorted(
        {hyp for hyp_dict in file_avg_scores.values() for hyp in hyp_dict},
        key=lambda x: desired_order.index(x) if x in desired_order else len(desired_order)
    )

    for hyp in all_hyp_types:
        row = {"Hypothesis Type": hyp}
        for fname in all_data:
            row[fname] = file_avg_scores[fname].get(hyp, None)
        comparison_data[hyp] = row

    comp_table_df = pd.DataFrame.from_dict(comparison_data, orient="index").reset_index(drop=True)

    # Highlight best score per row
    def highlight_best(val, row_best):
        if pd.isna(val) or pd.isna(row_best):
            return ""
        if val == row_best:
            return "background-color: #90EE90; font-weight: bold"
        return ""

    # Identify numeric columns (file columns)
    numeric_cols = [c for c in comp_table_df.columns if c != "Hypothesis Type"]

    # Compute best per row for numeric columns only
    styled_df = comp_table_df.style.apply(
        lambda row: [
            highlight_best(row[col], max(row[numeric_cols])) if col in numeric_cols else ""
            for col in comp_table_df.columns
        ],
        axis=1
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================================
    # Per-file detailed expanders
    # ==========================================================
    st.subheader("🔍 Per-File Details")

    for fname, data in all_data.items():
        with st.expander(f"{fname} ({len(data)} samples)"):
            # Sample selector for this file
            sample_names = [
                f"Sample {i + 1}: {item['source'][:80]}"
                for i, item in enumerate(data)
            ]

            selected_index = st.selectbox(
                f"Select a sample — {fname}",
                range(len(sample_names)),
                format_func=lambda x: sample_names[x],
                key=f"select_{fname}"
            )

            sample = data[selected_index]

            st.subheader("Source")
            st.info(sample["source"])

            st.subheader("Reference")
            st.success(sample["reference"])

            rows = []
            for hyp_type, hyp_data in sample["hypothesis"].items():
                rows.append({
                    "Hypothesis Type": hyp_type,
                    "Translation": hyp_data["text"],
                    "Score": hyp_data["score"]
                })

            df = pd.DataFrame(rows)
            st.subheader("Detailed Table")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # Per-file average bar chart (compact)
            st.subheader("Average Scores — This File")
            file_avg_rows = []
            for hyp_type, avg_score in file_avg_scores[fname].items():
                file_avg_rows.append({
                    "Hypothesis Type": hyp_type,
                    "Average Score": avg_score
                })
            file_avg_df = pd.DataFrame(file_avg_rows)

            # Apply same custom ordering
            file_avg_df["Hypothesis Type"] = pd.Categorical(
                file_avg_df["Hypothesis Type"],
                categories=desired_order,
                ordered=True
            )

            fig_file = px.bar(
                file_avg_df,
                x="Hypothesis Type",
                y="Average Score",
                text="Average Score",
                title=f"Average Score — {fname}",
                color_discrete_sequence=["#636EFA"]
            )
            fig_file.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside"
            )
            fig_file.update_layout(
                yaxis_range=[0, 1],
                xaxis_title="Hypothesis Type",
                yaxis_title="Average Score",
                height=350
            )
            st.plotly_chart(fig_file, use_container_width=True)

else:
    st.info("Upload JSON files to begin.")