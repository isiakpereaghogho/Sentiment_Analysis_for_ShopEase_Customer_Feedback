import pandas as pd
import re
import spacy
import logging
import langid
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')


nltk.download('stopwords')

from src.data_ingestion import data_ingestion
from config.constant import Cleaned_Data

logging.basicConfig(level=logging.INFO)

sentiment_data = data_ingestion()


class DataCleaning:
    def __init__(self):
        self.sentiment_data = sentiment_data
        self.nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        self.nlp_multi = spacy.load("xx_ent_wiki_sm", disable=["parser", "ner", "textcat"])
        self.stop_words = set(stopwords.words("english"))

        self.sentiment_data["review"] = self.sentiment_data["review"].astype(str).str.lower()
        self.sentiment_data["review"] = self.sentiment_data["review"].str.strip()

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def possibly_multilingual(self, text):
        return bool(re.search(r"[^\x00-\x7F]", str(text)))

    def detect_language_fast(self, text):
        try:
            return langid.classify(str(text)[:200])[0]
        except Exception:
            return "unknown"

    def lemmatize_multilingual(self, text, lang):
        text = str(text)

        if lang == "en":
            doc = self.nlp_en(text)
        else:
            doc = self.nlp_multi(text)

        tokens = [token.lemma_ if token.lemma_ else token.text for token in doc]
        return " ".join(tokens).strip()

    def clean_data(self, sentiment_data):
        logging.info("Starting data cleaning...")
        try:
            sentiment_data["clean_review"] = sentiment_data["review"].apply(self.clean_text)

            sentiment_data["language"] = "en"
            mask = sentiment_data["review"].apply(self.possibly_multilingual)
            sentiment_data.loc[mask, "language"] = sentiment_data.loc[mask, "review"].apply(self.detect_language_fast)

            sentiment_data["lemma_text_review"] = sentiment_data.apply(
                lambda row: self.lemmatize_multilingual(row["clean_review"], row["language"]),
                axis=1
            )

            sentiment_data["final_review"] = sentiment_data["lemma_text_review"].apply(
                lambda x: " ".join([word for word in str(x).split() if word not in self.stop_words])
            )

            sentiment_data["label"] = sentiment_data["rating"].apply(
                lambda r: 0 if r in (1, 2) else (1 if r == 3 else 2)
            )

            sentiment_data = sentiment_data[["review", "final_review", "label"]]
            sentiment_data.to_csv(Cleaned_Data, index=False)

            logging.info("Data cleaning completed.")
            return sentiment_data

        except Exception as e:
            logging.error(f"Error occurred while cleaning data: {e}")
            raise



             
