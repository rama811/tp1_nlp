import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords

df=pd.read_csv(r"C:\Users\tomas\OneDrive\Desktop\facultad\4to cuatrimestre\Procesamiento del lenguaje natural\TP_1\dataset_tp2.csv")

categoria_mapping = {'Policiales': 0, 'Politica': 1, 'Futbol': 2, 'Tecnologia': 3}
df["labels"]=df["Categoría"].map(categoria_mapping)

#nltk.download('stopwords')
spanish_stop_words = stopwords.words('spanish')

X = df["Titulo"].str.lower()
y = df["labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(stop_words=spanish_stop_words)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


modelo_LR = LogisticRegression(max_iter=1000)
modelo_LR.fit(X_train_vectorized, y_train)

# Evaluación del modelo de Regresión Logística
y_pred_LR = modelo_LR.predict(X_test_vectorized)
acc_LR = accuracy_score(y_test, y_pred_LR)
report_LR = classification_report(y_test, y_pred_LR, zero_division=1)

print("Precisión Regresión Logística:", acc_LR)
print("Reporte de clasificación Regresión Logística:\n", report_LR)


nuevas_frases = [
    "Inteligencia artificial y biología sintética: las tecnologías que, según un experto, podrían beneficiar o destuir a la humanidad",
    "Tenía pedido de captura por tentativa de homicidio y cayó por disparos en una plaza",
    "Lo asaltaron y le robaron la moto cuando iba a trabajar",
    "El récord de Rosario Central en el Gigante de Arroyito que supera al Manchester City",
    "Fiscal impulsa una investigación contra Milei y Marra por la corrida contra el peso",
    "Massa creció en Santa Fe pero Milei fue el más votado: tercios en Diputados y un socialista",
    "Elecciones 2023: las cinco boletas que hay este domingo en el cuarto oscuro",
    "Video: el recibimiento de los hinchas de Central en el Gigante",
    "Robaron una moto a mano armada y escaparon a los tiros",
    "Usó la inteligencia artificial para mostrar cómo se verían nietos robados en dictadura: "
]


nuevas_frases = [frase.lower() for frase in nuevas_frases]


nuevas_frases_vectorizadas = vectorizer.transform(nuevas_frases)


etiquetas_predichas = modelo_LR.predict(nuevas_frases_vectorizadas)

for i, etiqueta in enumerate(etiquetas_predichas):
    categoria_predicha = [key for key, value in categoria_mapping.items() if value == etiqueta][0]
    print(f"La frase '{nuevas_frases[i]}' pertenece a la categoría: {categoria_predicha}")


