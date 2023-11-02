import tensorflow_text
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el modelo de embedding
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
model = hub.load(module_url)

df = pd.read_csv('dataset.csv')

def generar_embeddings(data: pd.DataFrame, model: callable) -> pd.DataFrame:
    """
    Toma un DataFrame de datos y un modelo de embedding de texto como entrada, 
    itera a través de las categorías únicas en el DataFrame de datos, calcula los 
    embeddings de los títulos en cada categoría y retorna un nuevo DataFrame que 
    almacena las categorías, los títulos y sus correspondientes embeddings.

    Parámetros:
    data (pd.DataFrame): El DataFrame de datos que contiene las categorías y los títulos.
    model (callable): El modelo de embedding de texto que se utilizará para calcular los embeddings.

    Retorna:
    pd.DataFrame: Un nuevo DataFrame que contiene las categorías, títulos y sus embeddings.
    """
    df_embeddings = pd.DataFrame(columns=['categoria', 'titulo', 'vector'])
    
    for categoria in data['Categoría'].unique():
        titulos_categoria = data[data['Categoría'] == categoria]['Titulo'].tolist()
        
        # Calcular los embeddings de los títulos de la categoría
        categoria_embeddings = model(titulos_categoria)
        
        for i, titulo in enumerate(titulos_categoria):
            new_row = {'categoria': categoria, 'titulo': titulo, 'vector': categoria_embeddings[i]}
            df_embeddings = pd.concat([df_embeddings, pd.DataFrame([new_row])], ignore_index=True)
    
    return df_embeddings


df_embeddings = generar_embeddings(df, model)

# Calcular el centroide de 'Policiales'
centro_policiales = calcular_centro_vectores(df_embeddings, 'Policiales')

data = similitudes_por_categoria(df_embeddings, centro_policiales)

# Gráfico de caja
plt.boxplot(data, labels=['Policiales', 'Deportes', 'Tecnologia', 'Politica'])
plt.xlabel('Categorías')
plt.ylabel('Similitud')
plt.show()
