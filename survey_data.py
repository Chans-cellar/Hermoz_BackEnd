import sqlite3
import pandas as pd

# Regular Likert scale mapping
likert_mapping = {
    'Strongly Disagree': 0.0,
    'Disagree': 0.25,
    'Neutral': 0.5,
    'Agree': 0.75,
    'Strongly Agree': 1.0
}

# Inverted Likert scale mapping
inverted_likert_mapping = {
    'Strongly Disagree': 1.0,
    'Disagree': 0.75,
    'Neutral': 0.5,
    'Agree': 0.25,
    'Strongly Agree': 0.0
}

# Define the indices for each column
column_indices = {
    'timestamp': 'Timestamp',
    'email': 'Email Address',
    'age': 'What is your age?',
    'gender': 'What is your gender?',
    'education': 'What is the highest level of education you have completed?',
    'employment_status': 'What is your current employment status?',
    'occupation': 'What is your current occupation?',
    'marital_status': 'What is your marital status?',
    'income_range': 'Which of the following income ranges best represents your annual household income before taxes?',
    'household_size': 'How many people, including yourself, live in your household?',
    'inflation_current': 'Are you buying fewer goods for the same amount of money you usually spend?',
    'inflation_future': 'Do you think the prices of everyday items will continue increasing next year?',
    'unemployment_current': 'Do you think it is challenging to find a job in our country right now?',
    'unemployment_future': 'Do you think that more job opportunities will open up next year due to the government’s economic development?',
    'gdp_growth_current': 'Even though the government talks about economic development, do you think there is actually economic development in our country?',
    'gdp_growth_future': 'Do you think that there will be economic development next year, as the government expects?',
    'fiscal_policy_current': 'Do you think that the government is spending money wisely to help the economy?',
    'fiscal_policy_future': 'Do you think that the government will spend tax revenue more effectively next year?',
    'monetary_policy_current': 'Do you think the interest rate policies set by the Central Bank are helping the economy?',
    'monetary_policy_future': 'Do you think that loan interest rates will decrease next year?',
    'international_trade_current': 'Do you believe that our country benefits from trading with other countries?',
    'international_trade_future': 'Do you expect that international trade will improve next year?',
    'exchange_rates_current': 'Do you think the value of our currency is better for our economy compared to last year?',
    'exchange_rates_future': 'Do you believe that the US dollar will increase next year?'
}

# Define the indices for questions with inverted polarity
inverted_polarity_indices = {}

# Define the appropriate columns for current and future data
current_columns = {
    'Inflation': 'inflation_current',
    'Unemployment': 'unemployment_current',
    'GDP Growth': 'gdp_growth_current',
    'Fiscal Policy': 'fiscal_policy_current',
    'Monetary Policy': 'monetary_policy_current',
    'International Trade': 'international_trade_current',
    'Exchange Rates': 'exchange_rates_current'
}

future_columns = {
    'Inflation': 'inflation_future',
    'Unemployment': 'unemployment_future',
    'GDP Growth': 'gdp_growth_future',
    'Fiscal Policy': 'fiscal_policy_future',
    'Monetary Policy': 'monetary_policy_future',
    'International Trade': 'international_trade_future',
    'Exchange Rates': 'exchange_rates_future'
}


def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    return conn


def insert_survey_data(df):
    conn = get_db_connection()
    c = conn.cursor()

    def get_value(row, idx):
        value = row.get(idx, None)
        if pd.isnull(value):
            return None  # Handle missing values
        if idx in inverted_polarity_indices:
            return inverted_likert_mapping.get(value, None)
        return likert_mapping.get(value, None)

    for index, row in df.iterrows():
        print("index " + str(index))
        print("row " + str(row))
        c.execute('''
            INSERT INTO survey_responses (
                timestamp, email, age, gender, education, employment_status, occupation, marital_status, 
                income_range, household_size, inflation_current, inflation_future, unemployment_current, 
                unemployment_future, gdp_growth_current, gdp_growth_future, fiscal_policy_current, fiscal_policy_future, 
                monetary_policy_current, monetary_policy_future, international_trade_current, international_trade_future, 
                exchange_rates_current, exchange_rates_future
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
            pd.to_datetime(row.get(column_indices['timestamp'], None)).strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(
                row.get(column_indices['timestamp'], None)) else None,
            row.get(column_indices['email'], None),
            row.get(column_indices['age'], None),
            row.get(column_indices['gender'], None), row.get(column_indices['education'], None),
            row.get(column_indices['employment_status'], None),
            row.get(column_indices['occupation'], None), row.get(column_indices['marital_status'], None),
            row.get(column_indices['income_range'], None), row.get(column_indices['household_size'], None),
            get_value(row, column_indices['inflation_current']), get_value(row, column_indices['inflation_future']),
            get_value(row, column_indices['unemployment_current']),
            get_value(row, column_indices['unemployment_future']),
            get_value(row, column_indices['gdp_growth_current']), get_value(row, column_indices['gdp_growth_future']),
            get_value(row, column_indices['fiscal_policy_current']),
            get_value(row, column_indices['fiscal_policy_future']),
            get_value(row, column_indices['monetary_policy_current']),
            get_value(row, column_indices['monetary_policy_future']),
            get_value(row, column_indices['international_trade_current']),
            get_value(row, column_indices['international_trade_future']),
            get_value(row, column_indices['exchange_rates_current']),
            get_value(row, column_indices['exchange_rates_future'])
        ))

    conn.commit()
    conn.close()


def load_and_insert_survey_data(file_path):
    print('--' + file_path)
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        # Insert the data into the database
        insert_survey_data(df)
    except Exception as e:
        print(f'Error reading Excel file: {e}')


def get_survey_available_years():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT DISTINCT strftime("%Y", timestamp) as year FROM survey_responses ORDER BY year')
    years = c.fetchall()
    conn.close()

    year_list = [year[0] for year in years]
    return year_list


def get_survey_data_by_current_year(year):
    conn = get_db_connection()
    c = conn.cursor()

    result = {}

    for factor in current_columns:
        column_name = current_columns.get(factor)
        query = f'''
                SELECT AVG({column_name}) as avg_score
                FROM survey_responses 
                WHERE strftime('%Y', timestamp) = ?
            '''
        c.execute(query, (year,))
        avg_score = c.fetchone()[0]
        result[f'{factor}_avg'] = avg_score

    conn.close()

    return result


def get_survey_data_by_future_year(year):
    conn = get_db_connection()
    c = conn.cursor()

    result = {}

    for factor in future_columns:
        column_name = future_columns.get(factor)
        query = f'''
                    SELECT AVG({column_name}) as avg_score
                    FROM survey_responses 
                    WHERE strftime('%Y', timestamp) = ?
                '''
        c.execute(query, (year,))
        avg_score = c.fetchone()[0]
        result[f'{factor}_avg'] = avg_score

    conn.close()

    return result

def categorize_survey_sentiment(score):
    if score >= 0.8:
        return 'Strongly Positive'
    elif score >= 0.6:
        return 'Mildly Positive'
    elif score >= 0.4:
        return 'Neutral'
    elif score >= 0.2:
        return 'Mildly Negative'
    else:
        return 'Strongly Negative'


# Example usage
# if __name__ == '__main__':
#     file_path = '/mnt/data/Community Economic Survey (Responses).xlsx'
#     load_and_insert_survey_data(file_path)
