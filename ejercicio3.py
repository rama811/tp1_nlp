import pandas as pd
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import es_core_news_sm
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def convert_lower(df, column):
    """
    Función que recibe un dataframe y convierte la columna
    de texto a minúsculas

    """
    
    df[column] = df[column].str.lower()
    
    return df


def delete_puntuation(df, column):
    """
    Función que recibe un dataframe y 
    elimina todos los signos de puntuación del texto
    
    """

    df[column] = df[column].str.replace('[^\w\s]', '')

    return df


def remove_accents(text):
    """
    Función que elimina todos las tildes del texto

    """
    
    nfkd_form = unicodedata.normalize('NFKD', text)
    
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


def remove_stopwords(df, column, stop_words):
    """
    Función que recibe un dataframe y elimina palabras de parada del texto

    """
    
    df[column] = df[column].astype(str)
    df['Texto_Limpio'] = df[column].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    
    return df

    
def lemmatisation(df, column, nlp):
    """
    Función que recibe un dataframe, lematiza el texto y lo agrega al dataframe

    """

    lemmas = []
    for text in df[column]:
        doc = nlp(text)
        lemma_list = [token.lemma_ for token in doc]
        lemmas.append(' '.join(lemma_list))
    df['Texto_Lemat'] = lemmas
    
    return df


def word_count(text):
    """
    Función que recibe un dataframe y cuenta la cantidad de palabras de un texto

    """

    word_tokens = word_tokenize(text)

    return len(word_tokens)


def words_cloud(df, column):
    """
    Función que realiza una visualización de la frecuencia
    mediante nube de palabras

    """

    text = ' '.join(df[column])
    wordcloud = WordCloud(width = 800, height = 800,
    background_color ='white',
    min_font_size = 10).generate(text)
    
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title('General', fontsize=20)
    plt.tight_layout(pad = 0)
    plt.show()


def words_cloud_category(df, text_column, category_column, category_value):
    """
    Función que genera una nube de palabras para un valor específico en la columna de categoría
    
    """
    
    filtered_df = df[df[category_column] == category_value]
    text = ' '.join(filtered_df[text_column])

    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(category_value, fontsize=20)
    plt.tight_layout(pad=0)
    plt.show()



# Cargamos el dataframe generado en el ejercicio 2 y seteamos el nombre de la columna a trabajar
df = df = pd.read_csv('dataset_tp2.csv')
column = 'Texto'

# Convertimos todo el texto a minúsculas
df = convert_lower(df, column)

# Eliminamos todos los signos de puntuación del texto
df = delete_puntuation(df, column)

# Definimos las palabras de parada y creamos una nueva columa sin ellas
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('spanish'))
df = remove_stopwords(df, column, stop_words)

# Lematizamos el texto (sin stopwords) utilizando el paquete de spaCy para lenguaje español
nlp = es_core_news_sm.load()
df = lemmatisation(df, 'Texto_Limpio', nlp)

# Aplicamos la función para contar palabras, las agregamos al datafame e imprimimos el resultado
df['Cant_Palabras'] = df[column].apply(lambda x: word_count(x))
print(df)

# Realizamos una nube de palabras para cada categoría
for i in df['Categoría'].unique():
    words_cloud_category(df, 'Texto_Limpio', 'Categoría', i)

# Realizamos una nube de palabras general
words_cloud(df, 'Texto_Limpio')

# Exportamos el dataframe a un archivo .csv
df.to_csv("dataset_tp2_procesado.csv", index=False)

