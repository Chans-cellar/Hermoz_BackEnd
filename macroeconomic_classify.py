import joblib
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load the hybrid classifier files
clf_model_path = './Hybrid_Model/hybrid_classifier_tfidf.joblib'
tfidf_vectorizer_path = './Hybrid_Model/tfidf_vectorizer.joblib'
loaded_clf = joblib.load(clf_model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

hybrid_finbert_tokenizer_path = './Hybrid_Model/finbert_macroecon_tokenizer'
hybrid_finbert_model_path = './Hybrid_Model/finbert_macroecon_classifier'
hybrid_finbert_tokenizer = AutoTokenizer.from_pretrained(hybrid_finbert_tokenizer_path)
hybrid_finbert_model = AutoModel.from_pretrained(hybrid_finbert_model_path)


# Function to get embeddings from FinBERT
def get_embeddings(texts, hybrid_bert_model, hybrid_bert_tokenizer):
    inputs = hybrid_bert_tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = hybrid_bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


# Function to preprocess and predict a new sentence
def predict_category_by_hybrid(new_sentence):
    # Preprocess the new sentence
    embeddings = get_embeddings([new_sentence], hybrid_finbert_model, hybrid_finbert_tokenizer)
    tfidf_features = tfidf_vectorizer.transform([new_sentence]).toarray()

    # Combine FinBERT embeddings with TF-IDF features
    combined_features = np.hstack((embeddings, tfidf_features))

    # Make prediction
    prediction = loaded_clf.predict(combined_features)
    return prediction[0]