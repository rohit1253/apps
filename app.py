import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
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

#SqlAlchemy Database Configuration With Mysql
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/crud'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
 
db = SQLAlchemy(app)

#Creating model table for our CRUD database
class Data(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    country = db.Column(db.String(100))
    capital = db.Column(db.String(100))
    currency = db.Column(db.String(100))
    stamp = db.Column(db.String(100)) 
 
    def __init__(self, country, capital, currency, stamp):
 
        self.country = country
        self.capital = capital
        self.currency = currency
        self.stamp = stamp

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
    all_data = Data.query.all()
    return render_template('stamp.html', stamps = all_data)


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

@app.route('/sentiment/predcit',methods=['POST'])
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

#this route is for inserting data to mysql database via html forms
@app.route('/stamp_collection/insert', methods = ['POST'])
def insert():
 
    if request.method == 'POST':
        
        country = request.form['country']
        capital = request.form['capital']
        currency = request.form['currency']
        stamp = request.form['stamp']
 
 
        my_data = Data(country, capital, currency, stamp)
        db.session.add(my_data)
        db.session.commit()
 
        flash("Entry Inserted Successfully")
 
        return redirect(url_for('stamps_page'))
 
 
#this is our update route where we are going to update our stamps data
@app.route('/stamp_collection/update', methods = ['GET', 'POST'])
def update():
 
    if request.method == 'POST':
        my_data = Data.query.get(request.form.get('id'))
        
        my_data.country = request.form['country'] 
        my_data.capital = request.form['capital']
        my_data.currency = request.form['currency']
        my_data.stamp = request.form['stamp']
 
        db.session.commit()
        flash("Data Updated Successfully")
 
        return redirect(url_for('stamps_page'))
 
 
 
 
#This route is for deleting our stamps data
@app.route('/stamp_collection/delete/<id>/', methods = ['GET', 'POST'])
def delete(id):
    my_data = Data.query.get(id)
    db.session.delete(my_data)
    db.session.commit()
    flash("Data Deleted Successfully")
 
    return redirect(url_for('stamps_page'))

if __name__ == "__main__":
    app.run(debug=True)