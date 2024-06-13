import sqlite3
from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART summarizer model and tokenizer
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_model_name)
summarizer_tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)


def get_db_connection():
    conn = sqlite3.connect('predictions.db')
    return conn


def get_summary_by_factor_year(year, factor):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT sentence FROM predictions WHERE year=? AND label=?', (year, factor))
    sentences = c.fetchall()
    conn.close()

    combined_text = '. '.join([sentence[0] for sentence in sentences])

    inputs = summarizer_tokenizer.encode("summarize: " + combined_text, return_tensors="pt", max_length=1024,
                                         truncation=True)
    summary_ids = summarizer_model.generate(inputs, num_beams=4, min_length=50, max_length=100, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
