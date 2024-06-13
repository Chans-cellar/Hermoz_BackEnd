import sqlite3

def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    return conn

def get_report_available_years():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT DISTINCT year FROM predictions ORDER BY year')
    years = c.fetchall()
    conn.close()

    year_list = [year[0] for year in years]
    return year_list

def store_prediction(sentence, label, sentiment_score, year, document_name):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO predictions (sentence, label, sentiment_score, year, document_name) VALUES (?, ?, ?, ?, ?)',
              (sentence, label, sentiment_score, year, document_name))
    conn.commit()
    conn.close()


def get_avg_sentiments():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT label, AVG(sentiment_score) as avg_sentiment, COUNT(*) as count FROM predictions GROUP BY label')
    label_counts = c.fetchall()
    conn.close()
    return {label: {'count': count, 'avg_sentiment': avg_sentiment} for label, avg_sentiment, count in label_counts}

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
        'data': {label: {'count': count, 'avg_sentiment': avg_sentiment} for label, avg_sentiment, count in
                 label_counts}
    }