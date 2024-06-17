import spacy
import re

# Load SpaCy's English model
spacy_nlp = spacy.load('en_core_web_sm')


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


def lemma_and_stopword_text(text):
    doc = spacy_nlp(text)
    # Remove stopwords and perform lemmatization
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(processed_tokens)


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


def preprocess_text_for_sentences(sentence: str):
    # clean_sentence = []

    # if len(sentence.split()) < 4:  # Skip sentences with less than 3 words
    #     continue
    cleaned_text = clean_text(sentence)
    fixed_text = fix_common_issues(cleaned_text)
    # Sentiment analysis before removing stopwords and lemmatization
    no_lemma_txt = fixed_text
    preprocessed_txt = lemma_and_stopword_text(fixed_text)
    # if len(preprocessed_txt.split()) < 4:  # Ensure the cleaned sentence is also meaningful
    #     continue
    clean_sentence = (preprocessed_txt, no_lemma_txt)

    return clean_sentence
