import tensorflow_text
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import pandas as pd
import matplotlib.pyplot as plt

from Funciones import generar_embeddings, similitudes_por_categoria, calcular_centro_vectores

# Cargar el modelo de embedding
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
model = hub.load(module_url)

df = pd.read_csv('dataset.csv')

df_embeddings = generar_embeddings(df, model)

# Calcular el centroide de 'Policiales'
centro_policiales = calcular_centro_vectores(df_embeddings, 'Policiales')

data = similitudes_por_categoria(df_embeddings, centro_policiales)

# Gráfico de caja
plt.boxplot(data, labels=['Policiales', 'Deportes', 'Tecnologia', 'Politica'])
plt.xlabel('Categorías')
plt.ylabel('Similitud')
plt.show()
