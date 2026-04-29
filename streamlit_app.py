import streamlit as st
import pandas as pd
import requests
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

#Backend URL for production deployment on localhost
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

#Backend URL for production deployment on Render.com
#API_URL = os.getenv("API_URL", "https://shopease-app-backend-12wr.onrender.com")

st.set_page_config(page_title="ShopEase Sentiment Dashboard", layout="wide")

# ---------- HEADER ----------
st.title("ShopEase Sentiment Analysis Dashboard")
st.markdown("Analyze customer reviews")

# Session state for last checked time
if "last_checked_time" not in st.session_state:
    st.session_state.last_checked_time = "Not checked yet"

# Show top info section
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info(f"**Current Date:** {datetime.now().strftime('%Y-%m-%d')}")
with col_info2:
    st.info(f"**Last Review Check Time:** {st.session_state.last_checked_time}")

st.divider()

# ---------- SINGLE REVIEW ----------
st.header("Single Review Prediction")
user_input = st.text_area("Enter customer review:", key="single_review_input")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict_sentiment",
                    json={"text": user_input},
                    timeout=180
                )

                if response.status_code == 200:
                    result = response.json()

                    # Update time checked
                    st.session_state.last_checked_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # ---------- Compact result layout ----------
                    col1, col2, col3 = st.columns([1.2, 1.2, 1])

                    with col1:
                        st.metric("Sentiment", result["sentiment"].capitalize())

                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.2f}")

                    with col3:
                        if "scores" in result:
                            scores = result["scores"]
                            labels = list(scores.keys())
                            values = list(scores.values())
                        else:
                            labels = [result["sentiment"], "Other"]
                            values = [result["confidence"], 1 - result["confidence"]]

                        # Define consistent colors
                        color_map = {
                        "negative": "#ff4b4b",   # red
                        "neutral": "#ffa600",    # orange
                        "positive": "#00c49a"    # green
                    }
                        
                        theme = st.get_option("theme.base")

                        text_color = "white" if theme == "dark" else "black"

                        # Match colors to labels
                        colors = [color_map.get(label.lower(), "#cccccc") for label in labels]

                        fig, ax = plt.subplots(figsize=(1.0, 1.0), facecolor="none")
                        ax.set_facecolor("none")

                        wedges, texts, autotexts = ax.pie(
                        values,
                        colors=colors,
                        labels=None,
                        autopct="%1.0f%%",
                        startangle=90,
                        wedgeprops=dict(width=0.4)
                        )
                        # 🔧 Reduce font size here
                        for autotext in autotexts:
                            autotext.set_fontsize(7)
                            autotext.set_color(text_color) 

                        ax.axis("equal")
                        fig.patch.set_alpha(0)
                        ax.patch.set_alpha(0)

                        ax.set_title("")
                        st.pyplot(fig, use_container_width=False)
                        

                    # optional small legend under the chart row
                    # if "scores" in result:
                    #     st.caption(
                    #         " | ".join(
                    #             [f"{label.capitalize()}: {value:.2f}" for label, value in scores.items()]
                    #         )
                    #     )

                    # else:
                    #     st.error(f"Error in prediction. Status code: {response.status_code}")
                    #     st.write(response.text)

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.divider()

# ---------- BATCH PREDICTION ----------
st.header("Batch Prediction (CSV Upload)")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

#Backend URL for production deployment on Render.com
#API_URL = os.getenv("API_URL", "https://shopease-app-backend-12wr.onrender.com")

uploaded_file = st.file_uploader(
    "Upload CSV with 'review' column",
    type=["csv"],
    key="batch_file"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column.")
            st.stop()

    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()

    batch_size = st.number_input(
        "Batch size",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    if st.button("Run Batch Prediction", key="batch_predict"):
        results = []
        reviews = df["review"].astype(str).tolist()

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            for i in range(0, len(reviews), batch_size):
                batch = reviews[i:i + batch_size]

                for review in batch:
                    response = requests.post(
                        f"{API_URL}/predict_sentiment",
                        json={"text": review},
                        timeout=180
                    )

                    if response.status_code == 200:
                        pred = response.json()
                        results.append({
                            "review": review,
                            "sentiment": pred.get("sentiment"),
                            "confidence": pred.get("confidence")
                        })
                    else:
                        results.append({
                            "review": review,
                            "sentiment": "error",
                            "confidence": None
                        })

                progress = min((i + batch_size) / len(reviews), 1.0)
                progress_bar.progress(progress)
                status_text.write(f"Processed {min(i + batch_size, len(reviews))} of {len(reviews)} reviews")

            results_df = pd.DataFrame(results)

            st.success("Batch prediction completed successfully!")
            st.dataframe(results_df)

            if "sentiment" in results_df.columns:
                label_counts = results_df["sentiment"].value_counts()

                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(label_counts.index, label_counts.values, width=0.5, color=["#00c49a", "#ffa600", "#ff4b4b"])
                ax.set_xlabel("Sentiment Label")
                ax.set_ylabel("Count")
                ax.set_title("Count of Each Sentiment Label")
                #st.pyplot(fig)

                 # Center the chart and prevent stretching
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.pyplot(fig, use_container_width=False)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv",
                key="download_results"
            )

        except Exception as e:
            st.error(f"An error occurred during batch prediction: {e}")

st.divider()
# ---------- MODEL TRAINING ----------
st.header("Model Training")
st.warning("Model training can take several minutes. Please be patient.")

if st.button("Retrain Model", key="retrain"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    timer_text = st.empty()

    start_time = time.time()

    try:
        for percent in range(0, 91, 10):
            progress_bar.progress(percent)
            elapsed = int(time.time() - start_time)
            status_text.write(f"Training in progress... {percent}%")
            timer_text.write(f"Elapsed time: {elapsed} seconds")
            time.sleep(0.5)

        response = requests.get(f"{API_URL}/train", timeout=300)

        elapsed = int(time.time() - start_time)

        if response.status_code == 200:
            progress_bar.progress(100)
            status_text.success("Training completed successfully!")
            timer_text.write(f"Total training time: {elapsed} seconds")
            st.balloons()
        else:
            status_text.error(f"Training failed. Status code: {response.status_code}")
            timer_text.write(f"Stopped after: {elapsed} seconds")
            st.write(response.text)

    except Exception as e:
        elapsed = int(time.time() - start_time)
        status_text.error(f"An error occurred during model training: {e}")
        timer_text.write(f"Stopped after: {elapsed} seconds")