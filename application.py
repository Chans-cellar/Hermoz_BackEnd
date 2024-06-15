from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import os

from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
import sqlite3

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

import numpy as np

from survey_data import load_and_insert_survey_data, get_survey_available_years, get_survey_data_by_current_year, \
    get_survey_data_by_future_year
from data_preprocessing import preprocess_text
from macroeconomic_classify import predict_category_by_hybrid
from finBERT_sentiment_analysis import predict_finbert_sentiment
from report_data import (store_prediction, get_avg_sentiments, get_avg_sentiments_of_last_document,
                         get_avg_sentiments_by_year, get_avg_sentiments_by_factor, get_report_available_years,
                         get_report_sentiment_class_ratios)
from summarize import get_summary_by_factor_year
from correlation_analysis import correlation_analysis

application = Flask(__name__)
CORS(application)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# model_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Classifier Model\\macroecon_classifier'
# tokenizer_directory = 'E:\\studies\\USJ FOT\\lecture\\Research\\CodeBase\\Classifier Model\\tokenizer_model'
# tokenizer = BertTokenizer.from_pretrained(tokenizer_directory)
# model = BertForSequenceClassification.from_pretrained(model_directory)


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


@application.route('/')
def home():
    return render_template('index.html')


# ----REPORT DATA-----

@application.route('/get_report_available_years', methods=['GET'])
def get_report_available_years_response():
    years = get_report_available_years()
    return jsonify({'years': years})


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


@application.route('/process_data', methods=['POST'])
def process_data():
    if 'file' not in request.files:
        return jsonify({'error': 'File not provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File not selected'})

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
            results = []
            sentiment_scores = {}
            sentiment_counts = {}
            for i, (preprocessed_txt, no_lemma_txt) in enumerate(sentences):
                macroecon_label = predict_category_by_hybrid(preprocessed_txt)
                sentiment_score = predict_finbert_sentiment(no_lemma_txt)

                # can get results as individual sentences
                results.append({
                    'sentence': no_lemma_txt,
                    'macroeconomic_factor': macroecon_label,
                    'sentiment_score': sentiment_score
                })
                if macroecon_label not in sentiment_scores:
                    sentiment_scores[macroecon_label] = 0
                    sentiment_counts[macroecon_label] = 0
                sentiment_scores[macroecon_label] += sentiment_score
                sentiment_counts[macroecon_label] += 1
                progress = (i + 1) / total_sentences * 100
                print(f'Processing: {progress:.2f}%')

            avg_sentiment_scores = {label: sentiment_scores[label] / sentiment_counts[label]
                                    for label in sentiment_scores}

            return jsonify({'results': avg_sentiment_scores})
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


@application.route('/report_sentiment_class_ratios/<year>', methods=['GET'])
def report_sentiment_class_ratios_response(year):
    try:
        ratios = get_report_sentiment_class_ratios(year)
        return jsonify({'sentiment_class_ratios': ratios})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@application.route('/get_summary_by_factor_year', methods=['POST'])
def get_summary_by_factor_year_response():
    data = request.get_json()
    year = data['year']
    factor = data['factor']

    summary = get_summary_by_factor_year(year, factor)

    return jsonify({'summary': summary})




# --SURVEY DATA--
@application.route('/insert_survey_data', methods=['POST'])
def insert_survey_data():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'File not provided'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    try:
        load_and_insert_survey_data(file_path)
        return jsonify({'status': 'Survey data inserted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@application.route('/get_survey_available_years', methods=['GET'])
def get_survey_available_years_response():
    years = get_survey_available_years()
    return jsonify({'years': years})


@application.route('/get_survey_data_current', methods=['POST'])
def get_survey_data_current_response():
    data = request.get_json()
    selected_year = data.get('year')

    if not selected_year:
        return jsonify({'error': 'Year not provided'}), 400

    try:

        result = get_survey_data_by_current_year(selected_year)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@application.route('/get_survey_data_future', methods=['POST'])
def get_survey_data_future_response():
    data = request.get_json()
    selected_year = data.get('year')

    if not selected_year:
        return jsonify({'error': 'Year not provided'}), 400

    try:

        result = get_survey_data_by_future_year(selected_year)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@application.route('/correlation_analysis/<year>', methods=['GET'])
def correlation_analysis_response(year):
    try:
        correlation = correlation_analysis(year)
        return jsonify({'correlation': correlation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
