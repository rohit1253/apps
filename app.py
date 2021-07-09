import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
#from flask_sqlalchemy import SQLAlchemy
import pickle
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# A dictionary if language codes with ISO 639-1 encoding and their respective languages
languages = {'aa': 'Afar', 'ab': 'Abkhazian', 'ae': 'Avestan', 'af': 'Afrikaans', 'ak': 'Akan', 'am': 'Amharic', 'an': 'Aragonese',
 'ar': 'Arabic', 'as': 'Assamese', 'av': 'Avaric', 'ay': 'Aymara', 'az': 'Azerbaijani', 'ba': 'Bashkir', 'be': 'Belarusian',
 'bg': 'Bulgarian', 'bh': 'Bihari languages', 'bi': 'Bislama', 'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Tibetan', 'br': 'Breton',
 'bs': 'Bosnian', 'ca': 'Catalan; Valencian', 'ce': 'Chechen', 'ch': 'Chamorro', 'co': 'Corsican', 'cr': 'Cree', 'cs': 'Czech',
 'cu': 'Church Slavic; Old Slavonic; Church Slavonic; Old Bulgarian; Old Church Slavonic', 'cv': 'Chuvash', 'cy': 'Welsh',
 'da': 'Danish', 'de': 'German', 'dv': 'Divehi; Dhivehi; Maldivian', 'dz': 'Dzongkha', 'ee': 'Ewe', 'el': 'Greek, Modern (1453-)',
 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish; Castilian', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'ff': 'Fulah',
 'fi': 'Finnish', 'fj': 'Fijian', 'fo': 'Faroese', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Gaelic; Scottish Gaelic',
 'gl': 'Galician', 'gn': 'Guarani', 'gu': 'Gujarati', 'gv': 'Manx', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'ho': 'Hiri Motu',
 'hr': 'Croatian', 'ht': 'Haitian; Haitian Creole', 'hu': 'Hungarian', 'hy': 'Armenian', 'hz': 'Herero',
 'ia': 'Interlingua (International Auxiliary Language Association)', 'id': 'Indonesian', 'ie': 'Interlingue; Occidental', 'ig': 'Igbo',
 'ii': 'Sichuan Yi; Nuosu', 'ik': 'Inupiaq', 'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'ja': 'Japanese',
 'jv': 'Javanese', 'ka': 'Georgian', 'kg': 'Kongo', 'ki': 'Kikuyu; Gikuyu', 'kj': 'Kuanyama; Kwanyama', 'kk': 'Kazakh',
 'kl': 'Kalaallisut; Greenlandic', 'km': 'Central Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'kr': 'Kanuri', 'ks': 'Kashmiri',
 'ku': 'Kurdish', 'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kirghiz; Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish; Letzeburgesch',
 'lg': 'Ganda', 'li': 'Limburgan; Limburger; Limburgish', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'lu': 'Luba-Katanga',
 'lv': 'Latvian', 'mg': 'Malagasy', 'mh': 'Marshallese', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian',
 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'na': 'Nauru', 'nb': 'Bokmål, Norwegian; Norwegian Bokmål',
 'nd': 'Ndebele, North; North Ndebele', 'ne': 'Nepali', 'ng': 'Ndonga', 'nl': 'Dutch; Flemish', 'nn': 'Norwegian Nynorsk; Nynorsk, Norwegian',
 'no': 'Norwegian', 'nr': 'Ndebele, South; South Ndebele', 'nv': 'Navajo; Navaho', 'ny': 'Chichewa; Chewa; Nyanja',
 'oc': 'Occitan (post 1500)', 'oj': 'Ojibwa', 'om': 'Oromo', 'or': 'Oriya', 'os': 'Ossetian; Ossetic', 'pa': 'Panjabi; Punjabi',
 'pi': 'Pali', 'pl': 'Polish', 'ps': 'Pushto; Pashto', 'pt': 'Portuguese', 'qu': 'Quechua', 'rm': 'Romansh', 'rn': 'Rundi',
 'ro': 'Romanian; Moldavian; Moldovan', 'ru': 'Russian', 'rw': 'Kinyarwanda', 'sa': 'Sanskrit', 'sc': 'Sardinian', 'sd': 'Sindhi',
 'se': 'Northern Sami', 'sg': 'Sango', 'si': 'Sinhala; Sinhalese', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona',
 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'ss': 'Swati', 'st': 'Sotho, Southern', 'su': 'Sundanese', 'sv': 'Swedish',
 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tk': 'Turkmen', 'tl': 'Tagalog',
 'tn': 'Tswana', 'to': 'Tonga (Tonga Islands)', 'tr': 'Turkish', 'ts': 'Tsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian',
 'ug': 'Uighur; Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'vi': 'Vietnamese', 'vo': 'Volapük',
 'wa': 'Walloon', 'wo': 'Wolof', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang; Chuang', 'zh': 'Chinese', 'zu': 'Zulu'}


app = Flask(__name__)
model = pickle.load(open('iris_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus_page():
    return render_template('aboutus.html')

@app.route('/iris')
def iris_page():
    return render_template('iris.html')

@app.route('/language')
def language_page():
    return render_template('language.html')

@app.route('/sentiment')
def sentiment_page():
    return render_template('sentiment.html')

@app.route('/rubik')
def rubiks_page():
    return render_template('rubik.html')

@app.route('/stamp_collection')
def stamps_page():
    return render_template('stamp.html')

@app.route('/loan_calculator')
def loan_page():
    return render_template('loan.html')


@app.route('/iris/predict',methods=['POST'])
def predict_iris():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('iris.html', prediction_text='Iris species is : {}'.format(output))

@app.route('/language/detect',methods=['POST'])
def detect_language():
    languages_ratios = {}
    text = str([x for x in request.form.values()])
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        languages_ratios[language] = len(common_elements)
    most_rated_language = max(languages_ratios, key=languages_ratios.get).capitalize()
    return render_template('language.html', detection_text='The detected language is : {}'.format(most_rated_language))

@app.route('/sentiment/predict',methods=['POST'])
def predict_sentiment():
    text = str([x for x in request.form.values()])
    text_sentiment_score = TextBlob(text).sentiment.polarity
    if text_sentiment_score >= 0.05:
        text_sentiment = "Positive"
    elif text_sentiment_score <= -0.05:
        text_sentiment = "Negative"
    else:
        text_sentiment = "Neutral"
    return render_template('sentiment.html', sent_prediction_score_text='The sentiment score is : {}'.format(text_sentiment_score), sent_prediction_text='The sentiment is : {}'.format(text_sentiment))

@app.route('/loan_calculator/result',methods=['POST'])
def loan_result():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    p = float(final_features[0][0])
    r = float(final_features[0][1])
    n = float(final_features[0][2])
    A = round((p * ((1 + (r/100)) ** n)), 2)
    I = A - p
    emi = round((A / n), 2)
    return render_template('loan.html', principal_text='Loan Amount: \u20B9 {}'.format(int(p)), interest_rate_text='Rate of Interest: {}%'.format(r), tenure_text='Tenure: {} months'.format(int(n)), total_amount_text='Total Amount payable: \u20B9 {}'.format(A), total_interest_text='Total Interest payable: \u20B9 {}'.format(I), emi_text='EMI: \u20B9 {}'.format(emi))

if __name__ == "__main__":
    app.run(debug=True)
