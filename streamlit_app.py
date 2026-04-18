import streamlit as st
import pandas as pd
import requests
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

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
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Update time checked
                    st.session_state.last_checked_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    st.success(
                        f"Predicted Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})"
                    )

                    # ---------- Doughnut chart ----------
                    st.subheader("Confidence Distribution")

                    if "scores" in result:
                        scores = result["scores"]  # expected dict
                        labels = list(scores.keys())
                        values = list(scores.values())
                    else:
                        # fallback if only one confidence is returned
                        labels = [result["sentiment"], "Other"]
                        values = [result["confidence"], 1 - result["confidence"]]

                    fig, ax = plt.subplots(figsize=(5, 5))
                    wedges, texts, autotexts = ax.pie(
                        values,
                        labels=labels,
                        autopct="%1.1f%%",
                        startangle=90,
                        wedgeprops=dict(width=0.4)  # makes it doughnut
                    )
                    ax.set_title("Prediction Confidence")
                    st.pyplot(fig)

                else:
                    st.error(f"Error in prediction. Status code: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.divider()

# ---------- BATCH PREDICTION ----------
st.header("Batch Prediction (CSV Upload)")
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
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()

    if st.button("Run Batch Prediction", key="batch_predict"):
        with st.spinner("Processing batch prediction..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=60
                )

                if response.status_code == 200:
                    results = pd.DataFrame(response.json())

                    st.success("Batch prediction completed successfully!")
                    st.dataframe(results)

                    # ---------- Bar chart ----------
                    st.subheader("Label Counts from Batch Prediction")

                    if "sentiment" in results.columns:
                        label_counts = results["sentiment"].value_counts()

                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.bar(label_counts.index, label_counts.values)
                        ax.set_xlabel("Sentiment Label")
                        ax.set_ylabel("Count")
                        ax.set_title("Count of Each Sentiment Label")
                        st.pyplot(fig)
                    else:
                        st.warning("No 'sentiment' column found in batch prediction results.")

                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                        key="download_results"
                    )

                else:
                    st.error(f"Error from API. Status code: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error(f"An error occurred during batch prediction: {e}")

st.divider()

# ---------- MODEL TRAINING ----------
st.header("Model Training")
st.warning("Model training can take several minutes. Please be patient.")

if st.button("Retrain Model", key="retrain"):
    progress_bar = st.protreamgress(0)
    status_text = st.empty()
    timer_text = st.empty()

    start_time = time.time()

    try:
        # Fake progress animation while waiting for backend
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