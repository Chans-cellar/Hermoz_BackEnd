import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.special import expit



def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    return conn

def fetch_data(target_year):
    conn = get_db_connection()
    query = '''
    SELECT year, label, AVG(sentiment_score) as avg_sentiment
    FROM predictions 
    WHERE year < ?
    GROUP BY year, label
    ORDER BY year
    '''
    df = pd.read_sql(query, conn, params=(target_year,))
    conn.close()
    return df


def prepare_data(df):
    df_pivot = df.pivot(index='year', columns='label', values='avg_sentiment').fillna(0)
    return df_pivot


def train_and_predict(df_pivot, target_year):
    predictions = {}
    for label in df_pivot.columns:
        X = df_pivot.index.values.reshape(-1, 1)  # Years as features
        y = df_pivot[label].values  # Sentiment scores as target

        # Train the model
        model = LinearRegression()
        model.fit(X, y)

        # Predict for the target year
        target_year_array = np.array([[target_year]])
        prediction = model.predict(target_year_array)[0]
        prediction = expit(prediction)
        predictions[label] = prediction

    return predictions


def get_predictions_for_year(target_year):
    df = fetch_data(target_year)
    df_pivot = prepare_data(df)
    predictions = train_and_predict(df_pivot, target_year)
    return predictions




