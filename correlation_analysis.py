import sqlite3
import pandas as pd
from scipy.stats import pearsonr


def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    return conn


def get_government_data_by_year(year):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT label, AVG(sentiment_score) as avg_sentiment
        FROM predictions 
        WHERE year = ?
        GROUP BY label
    ''', (year,))
    data = c.fetchall()
    conn.close()

    return {label: avg_sentiment for label, avg_sentiment in data}


def get_survey_data_by_year(year):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT 
            inflation_current,
            fiscal_policy_current, 
            monetary_policy_current, international_trade_current, exchange_rates_current
        FROM survey_responses 
        WHERE strftime('%Y', timestamp) = ?
    ''', (year,))
    data = c.fetchall()
    conn.close()

    if not data:
        return {}

    df = pd.DataFrame(data, columns=[
        'Inflation', 'Fiscal Policy', 'Monetary Policy', 'International Trade', 'Exchange Rates'
    ])

    avg_sentiments = df.mean().to_dict()
    return {'current': avg_sentiments}


def calculate_correlation_by_factor(gov_data, survey_data):
    correlations = {}
    total_gov_values = []
    total_survey_values = []

    # print(len(gov_data))
    # print(len(survey_data['current']))

    for factor, gov_value in gov_data.items():
        if factor in survey_data['current']:
            survey_value = survey_data['current'][factor]
            total_gov_values.append(gov_value)
            total_survey_values.append(survey_value)

    print(" :gov_data " + str(total_gov_values) + ", survey_data " + str(total_survey_values))
    total_correlation, _ = pearsonr(total_gov_values, total_survey_values)

    # total_correlation, _ = pearsonr(total_gov_values, total_survey_values)
    # correlations['total'] = total_correlation
    return total_correlation


def correlation_analysis(year):
    gov_data = get_government_data_by_year(year)
    survey_data = get_survey_data_by_year(year)
    try:
        correlation = calculate_correlation_by_factor(gov_data, survey_data)
        return correlation
    except Exception as e:
        print({'error': str(e)})
