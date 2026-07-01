import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="COMET Score Explorer",
    layout="wide"
)

st.title("COMET Score Explorer")
st.write(
    "Upload a JSON file containing source sentences, reference translations, "
    "hypotheses, and COMET scores."
)

uploaded_file = st.file_uploader(
    "Choose a JSON file",
    type=["json"]
)

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)

        if not isinstance(data, list):
            st.error("JSON format should be a list of samples.")
            st.stop()



        # ==========================================================
        # Sample selection
        # ==========================================================
        sample_names = [
            f"Sample {i + 1}: {item['source'][:80]}"
            for i, item in enumerate(data)
        ]

        selected_index = st.selectbox(
            "Select a sample",
            range(len(sample_names)),
            format_func=lambda x: sample_names[x]
        )

        sample = data[selected_index]

        # ==========================================================
        # Source and Reference
        # ==========================================================
        st.subheader("Source")
        st.info(sample["source"])

        st.subheader("Reference")
        st.success(sample["reference"])

        # st.subheader("Hypotheses")

        rows = []

        for hyp_type, hyp_data in sample["hypothesis"].items():
            rows.append({
                "Hypothesis Type": hyp_type,
                "Translation": hyp_data["text"],
                "COMET Score": hyp_data["score"]
            })

        df = pd.DataFrame(rows)
        # ==========================================================
        # Detailed table
        # ==========================================================
        st.subheader("Detailed Table")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # ==========================================================
        # Dataset-level average score chart
        # ==========================================================
        average_scores = {}

        for sample in data:
            for hyp_type, hyp_data in sample["hypothesis"].items():
                average_scores.setdefault(hyp_type, []).append(
                    hyp_data["score"]
                )

        avg_df = pd.DataFrame({
            "Hypothesis Type": average_scores.keys(),
            "Average Score": [
                sum(scores) / len(scores)
                for scores in average_scores.values()
            ]
        })

        st.subheader("Average COMET Score Across Dataset")

        fig_avg = px.bar(
            avg_df,
            x="Hypothesis Type",
            y="Average Score",
            text="Average Score",
            title="Average COMET Score by Hypothesis Category"
        )

        fig_avg.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

        fig_avg.update_layout(
            yaxis_range=[0, 1],
            xaxis_title="Hypothesis Type",
            yaxis_title="Average COMET Score",
            height=450
        )

        st.plotly_chart(
            fig_avg,
            use_container_width=True
        )

        st.divider()


        

    except Exception as e:
        st.error(
            f"Failed to load JSON file: {e}"
        )

else:
    st.info("Upload a JSON file to begin.")