import sqlite3


def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT,
            label TEXT,
            sentiment_score REAL,
            year INTEGER,
            document_name TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            email TEXT,
            age TEXT,
            gender TEXT,
            education TEXT,
            employment_status TEXT,
            occupation TEXT,
            marital_status TEXT,
            income_range TEXT,
            household_size TEXT,
            inflation_current REAL,
            inflation_future REAL,
            unemployment_current REAL,
            unemployment_future REAL,
            gdp_growth_current REAL,
            gdp_growth_future REAL,
            fiscal_policy_current REAL,
            fiscal_policy_future REAL,
            monetary_policy_current REAL,
            monetary_policy_future REAL,
            international_trade_current REAL,
            international_trade_future REAL,
            exchange_rates_current REAL,
            exchange_rates_future REAL
        )
    ''')

    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
