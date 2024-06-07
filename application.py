from flask import Flask, request, render_template, jsonify
from flask_cors import  CORS
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForSequenceClassification, \
    AutoTokenizer, AutoModel, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import torch
import os
import re
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
import sqlite3
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

import joblib
import numpy as np




# Load SpaCy's English model
spacy_nlp = spacy.load('en_core_web_sm')

application = Flask(__name__)
CORS(application)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# model_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Classifier Model\\macroecon_classifier'
# tokenizer_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Classifier Model\\tokenizer_model'
# tokenizer = BertTokenizer.from_pretrained(tokenizer_directory)
# model = BertForSequenceClassification.from_pretrained(model_directory)


# Load the hybrid classifier files
clf_model_path = './Hybrid Model/hybrid_classifier_tfidf.joblib'
tfidf_vectorizer_path = './Hybrid Model/tfidf_vectorizer.joblib'
loaded_clf = joblib.load(clf_model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

hybrid_finbert_tokenizer_path = './Hybrid Model/finbert_macroecon_tokenizer'
hybrid_finbert_model_path = './Hybrid Model/finbert_macroecon_classifier'
hybrid_finbert_tokenizer = AutoTokenizer.from_pretrained(hybrid_finbert_tokenizer_path)
hybrid_finbert_model = AutoModel.from_pretrained(hybrid_finbert_model_path)

# Load finBERT model
finbert_tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
finbert_model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
finbert_nlp = pipeline('sentiment-analysis', model=finbert_model, tokenizer=finbert_tokenizer)

# Load BART summarizer model and tokenizer
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_model_name)
summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)




def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    return conn


def extract_text_from_pdf_file(pdf_file: str) -> str:
    with open(pdf_file, 'rb') as pdf:
        reader = PdfReader(pdf_file)
        pdf_text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                pdf_text.append(content)
        return pdf_text

def tokenize_non_validated_text(text):
    """Tokenize text and remove stopwords"""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens


def get_top_keywords(text, n=10):
    """Extract top keywords using TF-IDF"""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_keywords_idx = tfidf_matrix[0].toarray().argsort()[0][-n:][::-1]
    top_keywords = [feature_names[idx] for idx in top_keywords_idx]
    return top_keywords


def calculate_weighted_sum(keywords):
    # Define the keyword weights
    keyword_weights = {
        'financial': 40, 'monetary': 40, 'economic': 40, 'GDP': 40, 'inflation': 40,
        'unemployment': 40, 'fiscal': 40, 'trade': 40, 'prices': 30, 'price': 30, 'market': 30,
        'rate': 20, 'rates': 20, 'imports': 20, 'export': 20, 'exports': 20, 'revenue': 20, 'expenditure': 10,
        'debt': 20,
        'growth': 10, 'cent': 10, 'rs': 10
    }
    """Calculate the weighted sum of the keywords"""
    weighted_sum = 0
    for keyword in keywords:
        if keyword in keyword_weights:
            weighted_sum += keyword_weights[keyword]
    return weighted_sum

def detect_is_economic(texts):
    # Define a threshold value
    threshold = 20
    combined_text = '. '.join([text for text in texts])

    # Tokenize text
    tokens = tokenize_non_validated_text(combined_text)

    # Keyword extraction using TF-IDF
    top_keywords_tfidf = get_top_keywords(' '.join(tokens))
    weighted_sum = calculate_weighted_sum(top_keywords_tfidf)
    return weighted_sum > threshold


def lemma_and_stopword_text(text):
    doc = spacy_nlp(text)
    # Remove stopwords and perform lemmatization
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(processed_tokens)


def tokenize_sentences(text):
    doc = spacy_nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def merge_short_sentences(sentences, min_length=4):
    merged_sentences = []
    buffer = ''
    for sentence in sentences:
        if len(sentence.split()) < min_length:
            buffer += ' ' + sentence
        else:
            if buffer:
                merged_sentences.append(buffer.strip())
                buffer = ''
            merged_sentences.append(sentence.strip())
    if buffer:
        merged_sentences.append(buffer.strip())
    return merged_sentences


def remove_headers_footers(text: str) -> str:
    patterns = [
        r'Page \d+',
        r'\bChapter \d+\b',
        r'\bTable of Contents\b',
        r'\n+'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text



def clean_text(text: str) -> str:
    text = remove_headers_footers(text)

    # Remove all non-alphanumeric characters except spaces
    text = re.sub(r'\W', ' ', text)

    # Remove all digits except those followed by "per cent" or representing years (1900-2099)
    text = re.sub(r'\b(?!\d+(?:\s+per\s+cent|))\d{1,2}\b', '', text)

    # Keep 4-digit numbers that are likely years between 1900 and 2099
    text = re.sub(r'\b(?!19\d{2}|20\d{2})\d+\b', '', text)

    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove characters at the beginning of the string
    text = re.sub(r'^\s*[a-zA-Z]\s+', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Convert text to lowercase
    text = text.lower()

    return text


def fix_common_issues(text):
    text = re.sub(r'(\w)\.([A-Z])', r'\1. \2', text)
    return text


def preprocess_text(raw_texts: str):

    clean_sentences = []
    for text in raw_texts:
        dirt_sentences = tokenize_sentences(text)
        final_sentences = merge_short_sentences(dirt_sentences)
        for sentence in final_sentences:
            if len(sentence.split()) < 4:  # Skip sentences with less than 3 words
                continue
            cleaned_text = clean_text(sentence)
            fixed_text = fix_common_issues(cleaned_text)
            # Sentiment analysis before removing stopwords and lemmatization
            no_lemma_txt = fixed_text
            preprocessed_txt = lemma_and_stopword_text(fixed_text)
            if len(preprocessed_txt.split()) < 4:  # Ensure the cleaned sentence is also meaningful
                continue
            clean_sentences.append((preprocessed_txt, no_lemma_txt))
    return clean_sentences


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




# def predict_macroecon(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     predicted_class = predictions.argmax().item()
#     labels = ['Inflation', 'International Trade', 'GDP Growth', 'Exchange Rates', 'Monetary Policy', 'Fiscal Policy',
#               'Unemployment']
#     macroecon_label = labels[predicted_class]
#     return macroecon_label


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


def store_prediction(sentence, label, sentiment_score, year, document_name):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO predictions (sentence, label, sentiment_score, year, document_name) VALUES (?, ?, ?, ?, ?)',
              (sentence, label, sentiment_score, year, document_name))
    conn.commit()
    conn.close()

def forecast_next_year(data):
    X = data['Year'].values.reshape(-1, 1)
    y = data.iloc[:, 1].values
    model = LinearRegression()
    model.fit(X, y)
    next_year = np.array([[2024]])
    forecast = model.predict(next_year)
    return forecast[0]


@application.route('/forecast', methods=['GET'])
def forecast():
    conn = get_db_connection()
    c = conn.cursor()

    # Fetch data from the database
    c.execute('SELECT year, label, AVG(sentiment_score) as avg_sentiment FROM predictions GROUP BY year, label')
    data = c.fetchall()
    conn.close()

    # Process data into a DataFrame
    df = pd.DataFrame(data, columns=['Year', 'Label', 'Avg_Sentiment'])
    df_pivot = df.pivot(index='Year', columns='Label', values='Avg_Sentiment').fillna(0)

    # Perform simple forecasting (example with linear regression)
    forecasts = {}
    changes = {}
    for label in ['Inflation', 'International Trade', 'GDP Growth', 'Exchange Rates', 'Monetary Policy',
                  'Fiscal Policy', 'Unemployment']:
        if label in df_pivot.columns:
            forecast_value = forecast_next_year(df_pivot.reset_index()[['Year', label]])
            forecasts[label] = forecast_value

            # Calculate the percentage change
            last_year_value = df_pivot[label].iloc[-1]
            percentage_change = ((forecast_value - last_year_value) / last_year_value) * 100
            changes[label] = percentage_change
        else:
            forecasts[label] = 'Data not available'
            changes[label] = 'Data not available'

    return jsonify({'forecasts': forecasts, 'changes': changes})

def get_avg_sentiments_of_last_document():
    conn = get_db_connection()
    c = conn.cursor()

    # Find the name of the last added document
    c.execute('SELECT document_name FROM predictions ORDER BY id DESC LIMIT 1')
    last_document = c.fetchone()

    if not last_document:
        conn.close()
        return {}

    last_document_name = last_document[0]

    # Get the average sentiment and count for each label in the last document
    c.execute('''
        SELECT label, AVG(sentiment_score) as avg_sentiment, COUNT(*) as count
        FROM predictions 
        WHERE document_name = ? 
        GROUP BY label
    ''', (last_document_name,))

    label_counts = c.fetchall()
    conn.close()

    return {
        'document_name': last_document_name,
        'data': {label: {'count': count, 'avg_sentiment': avg_sentiment} for label, avg_sentiment, count in label_counts}
    }


def get_avg_sentiments_by_year(selected_year):
    conn = get_db_connection()
    c = conn.cursor()

    # Get the average sentiment for each label based on the selected year
    c.execute('''
        SELECT label, AVG(sentiment_score) as avg_sentiment
        FROM predictions 
        WHERE year = ?
        GROUP BY label
    ''', (selected_year,))

    avg_sentiments = c.fetchall()
    conn.close()

    # Organize the results by label
    result = {}
    for label, avg_sentiment in avg_sentiments:
        result[label] = avg_sentiment

    return result


@application.route('/get_avg_sentiments_by_year', methods=['POST'])
def get_avg_sentiments_by_year_response():
    data = request.json
    selected_year = data.get('year')

    if not selected_year:
        return jsonify({'error': 'Year not provided'}), 400

    try:
        result = get_avg_sentiments_by_year(selected_year)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_avg_sentiments_by_factor(selected_factor):
    conn = get_db_connection()
    c = conn.cursor()

    if selected_factor == 'all':
        # Get the average sentiment for each label based on the year
        c.execute('''
            SELECT label, year, AVG(sentiment_score) as avg_sentiment
            FROM predictions 
            GROUP BY label, year
        ''')
    else:
        # Get the average sentiment for the selected factor based on the year
        c.execute('''
            SELECT year, AVG(sentiment_score) as avg_sentiment
            FROM predictions 
            WHERE label = ?
            GROUP BY year
        ''', (selected_factor,))

    avg_sentiments = c.fetchall()
    conn.close()

    # Organize the results by year and then by label if 'all' is selected
    result = {}
    if selected_factor == 'all':
        for label, year, avg_sentiment in avg_sentiments:
            if year not in result:
                result[year] = {}
            result[year][label] = avg_sentiment
    else:
        for year, avg_sentiment in avg_sentiments:
            result[year] = avg_sentiment

    return result


@application.route('/get_avg_sentiments_by_factor', methods=['POST'])
def get_avg_sentiments_by_factor_response():
    data = request.json
    selected_factor = data.get('factor')

    if not selected_factor:
        return jsonify({'error': 'Macroeconomic factor not provided'}), 400

    try:
        result = get_avg_sentiments_by_factor(selected_factor)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_avg_sentiments():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT label, AVG(sentiment_score) as avg_sentiment, COUNT(*) as count FROM predictions GROUP BY label')
    label_counts = c.fetchall()
    conn.close()
    return {label: {'count': count, 'avg_sentiment': avg_sentiment} for label, avg_sentiment, count in label_counts}


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/store_data', methods=['POST'])
def store_data():
    print('Files:', request.files)
    print('Form:', request.form)
    if 'file' not in request.files or 'year' not in request.form:
        return jsonify({'error': 'File or year not provided'})

    file = request.files['file']
    year = request.form['year']
    if file.filename == '' or year == '':
        return jsonify({'error': 'File or year not selected'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        # Determine if the document is related to economics
        raw_texts = extract_text_from_pdf_file(filepath)
        is_economic = detect_is_economic(raw_texts)
        if is_economic:
            sentences = preprocess_text(raw_texts)
            total_sentences = len(sentences)
            for i, (preprocessed_txt, no_lemma_txt) in enumerate(sentences):
                # macroecon_label = predict_macroecon(preprocessed_txt)
                macroecon_label = predict_category_by_hybrid(preprocessed_txt)
                sentiment_score = predict_finbert_sentiment(no_lemma_txt)
                store_prediction(no_lemma_txt, macroecon_label, sentiment_score, year, filename)
                progress = (i + 1) / total_sentences * 100
                print(f'Processing: {progress:.2f}%')
            return jsonify({'status': 'Data stored successfully'})
        else:
            return jsonify({'error': 'The document is not related to economics'}), 400

@application.route('/get_single_doc_data', methods=['GET'])
def get_single_doc_data():
    single_doc_sentiment = get_avg_sentiments_of_last_document()
    print(single_doc_sentiment)
    return jsonify({'single_doc_data': single_doc_sentiment})

@application.route('/get_data', methods=['GET'])
def get_data():
    label_counts = get_avg_sentiments()
    return jsonify({'results': label_counts})


@application.route('/get_summary_by_factor_year', methods=['POST'])
def get_summary_by_factor_year():
    data = request.get_json()
    year = data['year']
    factor = data['factor']

    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT sentence FROM predictions WHERE year=? AND label=?', (year, factor))
    sentences = c.fetchall()
    conn.close()

    combined_text = '. '.join([sentence[0] for sentence in sentences])

    # inputs = t5_tokenizer.encode(combined_text, return_tensors="pt", max_length=512, truncation=True)
    # summary_ids = t5_model.generate(inputs, num_beams=4, min_length=50, max_length=100, early_stopping=True)
    # summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    inputs = summarizer_tokenizer.encode("summarize: " + combined_text, return_tensors="pt", max_length=1024,
                                         truncation=True)
    summary_ids = summarizer_model.generate(inputs, num_beams=4, min_length=50, max_length=100, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({'summary': summary})


if __name__ == '__main__':
    application.run(host='0.0.0.0')
