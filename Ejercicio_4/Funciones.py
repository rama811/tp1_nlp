import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf

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

def calcular_similitud(tensor1: tf.Tensor, tensor2: tf.Tensor) -> float:
    """
    Calcula la similitud de coseno entre dos tensores.

    Parámetros:
    tensor1 (tf.Tensor): El primer tensor.
    tensor2 (tf.Tensor): El segundo tensor.

    Retorna:
    float: El valor de similitud de coseno entre los dos tensores.
    """
    dot_product = tf.reduce_sum(tf.multiply(tensor1, tensor2))
    norm1 = tf.norm(tensor1)
    norm2 = tf.norm(tensor2)
    cosine_similarity = dot_product / (norm1 * norm2)

    return cosine_similarity

def calcular_centro_vectores(df: pd.DataFrame, categoria: str) -> np.ndarray:
    """
    Calcula el centro de los vectores de una categoría específica en un DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos de categorías y vectores.
    categoria (str): El nombre de la categoría para la que se calculará el centro de los vectores.

    Retorna:
    np.ndarray: Un arreglo NumPy que representa el centro de los vectores de la categoría.
    """
    df_categoria = df[df['categoria'] == categoria]
    centro_categoria = np.mean(df_categoria['vector'].to_list(), axis=0)

    return centro_categoria

def similitudes_por_categoria(df, centro_vector):
    """
    Calcula las similitudes entre un vector central y los datos de todas las categorías en un DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos de categorías y vectores.
    centro_vector (np.ndarray): El vector central con el que se calcularán las similitudes.

    Retorna:
    dict: Un diccionario donde las claves son las categorías y los valores son las similitudes correspondientes.
    """

    similitudes_por_categoria = []
    
    for categoria in df['categoria'].unique():
        df_categoria = df[df['categoria'] == categoria]
        similitudes = [calcular_similitud(centro_vector, vector) for vector in df_categoria['vector'].to_list()]
        similitudes_por_categoria.append(similitudes)

    return similitudes_por_categoria



