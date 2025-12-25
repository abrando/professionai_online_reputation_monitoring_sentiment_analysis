from typing import Dict, List

from .model import load_sentiment_pipeline
from .monitoring import record_prediction

# Single text prediction
def predict_single(text: str) -> Dict:
    sentiment_pipeline = load_sentiment_pipeline()

    # Force all scores so "probabilities" makes sense
    outputs = sentiment_pipeline(text, return_all_scores=True, truncation=True)[0]
    # outputs: [{"label": "...", "score": ...}, {"label": "...", ...}, ...]

    probabilities = {o["label"].lower(): float(o["score"]) for o in outputs}

    top_label = max(probabilities, key=probabilities.get)
    top_score = probabilities[top_label]

    record_prediction(label=top_label, score=top_score, text=text)

    return {"label": top_label, "score": top_score, "probabilities": probabilities}

# Batch text prediction
def predict_batch(texts: List[str]) -> List[Dict]:
    sentiment_pipeline = load_sentiment_pipeline()

    outs = sentiment_pipeline(texts, return_all_scores=True, truncation=True)
    results: List[Dict] = []

    for text, out in zip(texts, outs):
        probabilities = {o["label"].lower(): float(o["score"]) for o in out}
        top_label = max(probabilities, key=probabilities.get)
        top_score = probabilities[top_label]
        record_prediction(label=top_label, score=top_score, text=text)
        results.append({"label": top_label, "score": top_score, "probabilities": probabilities})

    return results
