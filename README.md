# Sentiment_Analysis_for_ShopEase_Customer_Feedback

A complete end-to-end sentiment analysis pipeline built to analyze customer reviews from an e-commerce platform (ShopEase).
This project compares traditional machine learning with modern NLP approaches (BERT) while handling real-world messy, multilingual data.
The purpose of this project is to develop a comprehensive sentiment analysis system that transforms raw, unstructured customer feedback into meaningful, structured insights that can directly inform business decisions. 


 ### Project Objectives 

•	Build a scalable data processing pipeline
•	Implement advanced text preprocessing (tokenization, lemmatization, language detection)
•	Develop high accuracy sentiment classification models (Logistic Regression → BERT)
•	Generate human readable summaries of customer feedback
•	Build interactive dashboards for real time insights
•	Deploy a scalable, real time system (Streamlit/Flask)


### The goal of this project is to:

- Analyze customer reviews and classify sentiment into:
Negative (0),
Neutral (1),
Positive (2)
- Handle noisy, real-world text data (emojis, symbols, multilingual content)
- Build a scalable preprocessing pipeline
- Train and evaluate both baseline models and BERT-based models
- Deliver insights that can improve business decisions

### Dataset

The dataset contains customer reviews with the following columns:

- review_id
- product_category
- timestamp
- country
- rating
- review
- sentiment (target)

### Project Structure

Sentiment_Analysis_for_ShopEase_Customer_Feedback/

│

├── config/

│   └── constant.py          # File paths & model configs

│

├── src/

│   ├── data_ingestion.py    # Load raw data

│   ├── data_cleaning.py     # Cleaning + NLP preprocessing

│   ├── data_preprocessing.py # Train/test split + tokenization

│

├── notebooks/               # Exploratory analysis (EDA)

├── data/                    # Raw and processed data

├── models/                  # Saved models

│

├── requirements.txt

└── README.md

### Data Cleaning & Preprocessing

The pipeline handles real-world noisy text using:

- Cleaning
- Lowercasing text
- Removing:URLs, mentions (@user), numbers, punctuation, emojis, extra whitespace
- Language Handling - Detects multilingual text using langid.
Applies: English spaCy model (en_core_web_sm), Multilingual model (xx_ent_wiki_sm)
- NLP Processing: Lemmatization (spaCy), Stopword removal (NLTK)
- Feature Engineering - Converts ratings into sentiment labels: 1–2 → Negative, 3 → Neutral, 4–5 → Positive

### Data Preparation

- Train/Test split using sklearn
- Tokenization using Hugging Face Transformers
- Conversion into PyTorch datasets

### Models Used:

🔹 Baseline Model:
- Logistic Regression: High performance (~97% accuracy)
🔹 Advanced Model:
- Multilingual BERT Model: distilbert-base-multilingual-cased. Handles multilingual text effectively

### Model Performance:

- Metric	Logistic Regression	    BERT
- Accuracy	    0.97	            0.97
- Macro F1	    0.95	            0.95
- Weighted F1	0.97	            0.97

### Insight:

BERT performs similarly to the baseline, suggesting:
- Dataset is relatively clean/structured
- Simpler models can still be effective

Despite the similarity in performance between BERT model and Logistic Regression (Baseline Model), Bert Model was chosen because it excels when context is complex, sarcasm or nuance exists and long dependencies matter.

### Evaluation:

- Confusion Matrix
- Precision, Recall, F1-score
- Class imbalance handled implicitly

### Business Impact: This system can help ShopEase:

- Understand customer satisfaction trends
- Detect negative feedback early
- Improve product quality and service
- Automate large-scale review analysis
- Support data-driven decision making

### How to Run

1. Clone repo:
git clone https://github.com/your-username/Sentiment_Analysis_for_ShopEase_Customer_Feedback.git
cd Sentiment_Analysis_for_ShopEase_Customer_Feedback
2. Create virtual environment:
python -m venv shopease_env
shopease_env\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt
4. Run pipeline:
python -m src.data_preprocessing
5. Push model to MLflow:
python -m pipelines.training
6. Loading the model (backend):
uvicorn main.app:app --reload
7. Running the streamlit dashboard (frontend):
streamlit run streamlit_app.py 

### Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- spaCy
- NLTK
- Hugging Face Transformers
- PyTorch
- Streamlit
- Matplotlib, Seaborn
- Mlflow, Dagshub
- Docker

### Challenges Faced

- Handling multilingual text with inconsistent detection
- Cleaning noisy real-world reviews (emoji, symbols, whitespace)
- Performance trade-offs between speed and NLP accuracy
- Environment and dependency management

### Future Improvements

- Fine-tune BERT for better performance
- Add SHAP explainability
- Improve multilingual handling (better language detection)
- Use larger transformer models
