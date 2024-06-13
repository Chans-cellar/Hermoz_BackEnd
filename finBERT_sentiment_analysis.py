from transformers import BertTokenizer, BertForSequenceClassification,pipeline
import torch


# Load finBERT model
finbert_tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
finbert_model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
finbert_nlp = pipeline('sentiment-analysis', model=finbert_model, tokenizer=finbert_tokenizer)

def convert_sentiment_scores(positive, neutral, negative):
    # Weights
    w_p = 1
    w_n = 0.5
    w_ng = -1

    # Compute weighted sum
    score = (w_p * positive) + (w_n * neutral) + (w_ng * negative)

    # Map to [0, 1] range
    normalized_score = (score + 1) / 2

    return normalized_score

def predict_finbert_sentiment(text):
    # Tokenize the texts
    inputs = finbert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Get the raw scores (logits) from the model
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        logits = outputs.logits

    # Convert logits to sentiment scores
    scores = torch.softmax(logits, dim=1)
    fair_score = convert_sentiment_scores(scores[0][0].item(), scores[0][2].item(), scores[0][1].item())
    return fair_score