from survey_data import get_survey_data_by_current_year, get_survey_data_by_future_year
from report_data import get_avg_sentiments_by_year
from forecast import get_predictions_for_year, fetch_data, prepare_data

future_columns = {
    'inflation': 'inflation_future',
    'unemployment': 'unemployment_future',
    'gdp_growth': 'gdp_growth_future',
    'fiscal_policy': 'fiscal_policy_future',
    'monetary_policy': 'monetary_policy_future',
    'international_trade': 'international_trade_future',
    'exchange_rates': 'exchange_rates_future'
}


def get_current_comparative_data(selected_year):
    report_data = get_avg_sentiments_by_year(selected_year)
    survey_data = get_survey_data_by_current_year(selected_year)

    # print(report_data, survey_data)

    comparative_data = {}

    for factor in report_data.keys():
        report_score = report_data.get(factor, None)
        survey_score = survey_data.get(factor, None)

        comparative_data[factor] = {
            'report_score': report_score,
            'survey_score': survey_score
        }

    return comparative_data


def get_report_data_changes(year, report_data):
    df = fetch_data(year)
    df_pivot = prepare_data(df)
    changes = {}
    for label in report_data:
        if label in df_pivot.columns:
            last_year_value = df_pivot[label].iloc[-1]
            forecast_value = report_data[label]
            percentage_change = ((forecast_value - last_year_value) / last_year_value) * 100
            changes[label] = percentage_change
        else:
            changes[label] = 'Data not available'

    return changes


def get_survey_data_changes(year, survey_future_data):
    current_data = get_survey_data_by_current_year(year)

    changes = {}
    for factor in survey_future_data:
        if factor in current_data:
            last_year_value = current_data[factor]
            future_year_value = survey_future_data[factor]
            percentage_change = ((future_year_value - last_year_value) / last_year_value) * 100
            changes[factor] = percentage_change
        else:
            changes[factor] = 'Data not available'

    return changes


def get_future_comparative_data(year):
    next_year = int(year) + 1
    report_data = get_predictions_for_year(next_year)
    report_changes = get_report_data_changes(next_year, report_data)

    survey_data = get_survey_data_by_future_year(str(year))
    survey_changes = get_survey_data_changes(str(year), survey_data)

    print('report data' + str(report_data), 'report data'+ str(report_changes), 'survey data'+str(survey_data), 'survey changes'+str(survey_changes))

    comparative_data = {}
    for factor in report_data.keys():
        report_score = report_data.get(factor, None)

        survey_score = survey_data.get(factor, None)

        report_change = report_changes.get(factor,None)

        survey_change = survey_changes.get(factor,None)


        comparative_data[factor] = {
            'report_score': report_score,
            'survey_score': survey_score,
            'report_change':report_change,
            'survey_change':survey_change
        }

    return comparative_data
