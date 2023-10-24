import pandas as pd
from Obtencion_de_datos import obtener_enlaces_noticias, web_scraping, construir_dataset
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
spanish_stop_words = stopwords.words('spanish')


urls_policiales = obtener_enlaces_noticias('https://www.rosario3.com/seccion/policiales/', 20)
urls_deporte = obtener_enlaces_noticias('https://www.rosario3.com/seccion/deportes/', 20)
urls_politica = obtener_enlaces_noticias('https://www.rosario3.com/seccion/politica/', 20)
urls_tecnologia = obtener_enlaces_noticias('https://www.rosario3.com/seccion/tecnologia/', 20)

df_policiales = construir_dataset("Policiales", urls_policiales)
df_deportes = construir_dataset("Deportes", urls_deporte)
df_politica = construir_dataset("Politica", urls_politica)
df_tecnologia = construir_dataset("Tecnologia", urls_tecnologia)

df_completo = pd.concat([df_policiales, df_deportes, df_politica, df_tecnologia], ignore_index=True)

df_completo.to_csv("dataset.csv", index=False)
