import re
import string

import nltk

mispell_dict = {
    "ain't": "is not",
    "cannot": "can not",
    "aren't": "are not",
    "can't": "can not",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "wont": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "colour": "color",
    "centre": "center",
    "favourite": "favorite",
    "travelling": "traveling",
    "counselling": "counseling",
    "theatre": "theater",
    "cancelled": "canceled",
    "labour": "labor",
    "organisation": "organization",
    "wwii": "world war 2",
    "citicise": "criticize",
    "youtu ": "youtube ",
    "Qoura": "Quora",
    "sallary": "salary",
    "Whta": "What",
    "narcisist": "narcissist",
    "howdo": "how do",
    "whatare": "what are",
    "howcan": "how can",
    "howmuch": "how much",
    "howmany": "how many",
    "whydo": "why do",
    "doI": "do I",
    "theBest": "the best",
    "howdoes": "how does",
    "Etherium": "Ethereum",
    "narcissit": "narcissist",
    "bigdata": "big data",
    "2k17": "2017",
    "2k18": "2018",
    "qouta": "quota",
    "exboyfriend": "ex boyfriend",
    "airhostess": "air hostess",
    "whst": "what",
    "watsapp": "whatsapp",
    "demonitisation": "demonetization",
    "demonitization": "demonetization",
    "demonetisation": "demonetization",
}

mispell_dict = {k.lower(): v.lower() for k, v in mispell_dict.items()}


def preprocess(text, polarity=None, subjectivity=None):
    text = _remove_amp(text)
    text = _remove_links(text)
    text = _remove_hashes(text)
    text = _remove_mentions(text)

    # text = _lowercase(text)
    text = _expand_contractions(text)
    text = _separate_punctuations(text)
    text = _remove_punctuation(text)
    # text = _remove_punctuation_all(text)
    # text = _remove_numbers(text)

    # text_tokens = _tokenize(text)
    # text_tokens = _stopword_filtering(text_tokens)
    # text_tokens = _stemming(text_tokens)
    # text = _stitch_text_tokens_together(text_tokens)

    text = _remove_multiple_spaces(text)
    text = _remove_leading_trailing_spaces(text)
    if polarity is not None and subjectivity is not None:
        text = _add_sentiment_tokens(text, polarity, subjectivity)
    text = _add_special_tokens(text)

    return text


def _remove_amp(text):
    return text.replace("&", " ")


def _remove_mentions(text):
    return re.sub(r"(@.*?)[\s]", " ", text)


def _remove_multiple_spaces(text):
    return re.sub(r"\s+", " ", text)


def _remove_links(text):
    return re.sub(r"https?:\/\/[^\s\n\r]+", " ", text)


def _remove_hashes(text):
    return re.sub(r"#", " ", text)


def _expand_contractions(text):
    return " ".join(
        [
            mispell_dict[word.lower()] if word.lower() in mispell_dict.keys() else word
            for word in text.split()
        ]
    )


def _stitch_text_tokens_together(text_tokens):
    return " ".join(text_tokens)


def _tokenize(text):
    return nltk.word_tokenize(text, language="english")


def _stopword_filtering(text_tokens):
    stop_words = nltk.corpus.stopwords.words("english")

    return [token for token in text_tokens if token not in stop_words]


def _stemming(text_tokens):
    porter = nltk.stem.porter.PorterStemmer()
    return [porter.stem(token) for token in text_tokens]


def _remove_numbers(text):
    return re.sub(r"\d+", " ", text)


def _lowercase(text):
    return text.lower()


def _separate_punctuations(text):
    # put spaces before & after punctuations to make words seprate. Like "king?" to "king", "?".
    return re.sub(r"([?!,+=—&%\'\";:¿।।।|\(\){}\[\]//])", r" \1 ", text)


def _remove_punctuation(text):
    punctuation_to_keep = "?\\!"  # keep these punctuations '?', '\', '!'
    return "".join(
        character
        for character in text
        if character not in string.punctuation or character in punctuation_to_keep
    )


def _remove_punctuation_all(text):
    return "".join(
        character for character in text if character not in string.punctuation
    )


def _remove_leading_trailing_spaces(text):
    return text.strip()


def _add_special_tokens(text):
    return "<s>" + text + "</s>"


def _add_sentiment_tokens(text, polarity, subjectivity):
    return text + " [SEP] " + str(polarity) + " [SEP] " + str(subjectivity)
