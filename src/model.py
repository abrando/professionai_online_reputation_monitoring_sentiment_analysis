# src/model.py
"""
Model utilities.

This module is responsible for loading the pretrained sentiment model
from Hugging Face and exposing a simple pipeline.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Pretrained sentiment model for Twitter
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_sentiment_pipeline():
    """
    Load HuggingFace tokenizer/model and create a cached pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        top_k=None,
    )
