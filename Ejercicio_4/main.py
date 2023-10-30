import tensorflow_text
import bokeh
import bokeh.models
import bokeh.plotting
import numpy as np
import os
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import pandas as pd
import sklearn
import random
from sentence_transformers import SentenceTransformer, util
from prettytable import PrettyTable

# Cargar el modelo de embedding
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
model = hub.load(module_url)

def embed_text(input):
    return model(input)

# Cargar el conjunto de datos CSV
data = pd.read_csv('dataset.csv')

# Crear un nuevo DataFrame con las columnas 'categoria', 'titulo' y el vector asociado
df_embeddings = pd.DataFrame(columns=['categoria', 'titulo', 'vector'])

# Iterar a través de las categorías únicas en el DataFrame original
categorias_unicas = data['Categoría'].unique()

for categoria in categorias_unicas:
    # Filtrar las noticias de la categoría actual
    titulos_categoria = data[data['Categoría'] == categoria]['Titulo'].tolist()
    
    # Calcular los embeddings de los títulos de la categoría
    categoria_embeddings = embed_text(titulos_categoria)
    
    # Agregar las filas al nuevo DataFrame
    for i, titulo in enumerate(titulos_categoria):
        new_row = {'categoria': categoria, 'titulo': titulo, 'vector': categoria_embeddings[i]}
        df_embeddings = pd.concat([df_embeddings, pd.DataFrame([new_row])], ignore_index=True)



def calcular_similitud(tensor1, tensor2):
    # Calcula el producto punto entre los dos tensores
    dot_product = tf.reduce_sum(tf.multiply(tensor1, tensor2))

    # Calcula la magnitud de cada tensor
    norm1 = tf.norm(tensor1)
    norm2 = tf.norm(tensor2)

    # Calcula la similitud de coseno
    cosine_similarity = dot_product / (norm1 * norm2)

    return cosine_similarity


# Crear una tabla para mostrar los resultados
tabla = PrettyTable()
tabla.field_names = ["Categoría 1", "Título 1","Categoría 2", "Título 2", "Similitud"]

# Iterar a través de las categorías únicas en df_embeddings
for categoria in categorias_unicas:
    # Filtrar títulos de la categoría actual
    subset_df = df_embeddings[df_embeddings['categoria'] == categoria][['categoria', 'titulo', 'vector']]
    
    # Seleccionar aleatoriamente dos títulos de la misma categoría
    filas_aleatorias = subset_df.sample(n=2, random_state=random.seed())
    
    # Filtra las filas de df_embeddings que no pertenecen a la categoría actual
    filas_categorias_diferentes = df_embeddings[df_embeddings['categoria'] != categoria]

    # Selecciona una fila aleatoria de las categorías diferentes
    fila_aleatoria = filas_categorias_diferentes.sample(n=1, random_state=random.seed())
    
    
    # Calcular similitud
    similitud_categoria = calcular_similitud(filas_aleatorias.iloc[0]['vector'], filas_aleatorias.iloc[1]['vector'])
    similitud_otra_categoria = calcular_similitud(filas_aleatorias.iloc[0]['vector'], fila_aleatoria.iloc[0]['vector'])
    
    # Agregar las filas a la tabla
    tabla.add_row([categoria, filas_aleatorias.iloc[0]['titulo'], categoria, filas_aleatorias.iloc[1]['titulo'], f"{similitud_otra_categoria.numpy():.4f}"])
    tabla.add_row([categoria, filas_aleatorias.iloc[0]['titulo'], fila_aleatoria.iloc[0]['categoria'], fila_aleatoria.iloc[0]['titulo'], f"{similitud_categoria.numpy():.4f}"])

# Mostrar la tabla
print(tabla)