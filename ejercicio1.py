import requests
from bs4 import BeautifulSoup
import pandas as pd

def obtener_enlaces_noticias(url_pagina: str, cantidad: int = 20) -> list[str]:
    """
    Obtiene una lista de URLs de noticias de una página web.

    Args:
        url_pagina (str): La URL de la página web que deseas analizar.
        cantidad (int, optional): La cantidad de enlaces a buscar (por defecto 10).

    Returns:
        list[str]: Una lista de URLs de noticias.
    """
    enlaces_noticias = []

    # Realiza una solicitud HTTP para obtener el contenido de la página
    response = requests.get(url_pagina)

    # Verifica si la solicitud fue exitosa
    if response.status_code != 200:
        print("Error al obtener la página:", response.status_code)
        return enlaces_noticias

    # Parsea el contenido de la página con BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encuentra los elementos HTML que contienen los enlaces a las noticias
    enlaces = soup.find_all('a', class_='cover-link')

    # Itera a través de los enlaces y obtén la URL completa
    for enlace in enlaces[:cantidad]:
        url_noticia = url_pagina.rsplit("/", 2)[0] + enlace['href']
        enlaces_noticias.append(url_noticia)

    return enlaces_noticias
        
def web_scraping(url_pagina: str) -> tuple[str, str]:
    """
    Realiza web scraping de un artículo de noticias a partir de una URL.

    Args:
        url_pagina (str): La URL de la página web del artículo.

    Returns:
        tuple[str, str]: Una tupla que contiene el título y el texto del artículo.
    """
    # Realiza una solicitud HTTP para obtener el contenido de la página
    response = requests.get(url_pagina)

    # Verifica si la solicitud fue exitosa
    if response.status_code != 200:
        print(f"Error al obtener la página {url_pagina}: {response.status_code}")
        return "", ""

    # Parsea el contenido de la página con BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encuentra el título del artículo
    titulo = soup.find_all('h1', {'class': "main-title font-900"})
    titulo = titulo[0].text

    # Encuentra el cuerpo de texto del artículo y extrae los párrafos
    cuerpo_texto = soup.find('div', {'class': "article-body"})
    parrafos = cuerpo_texto.find_all('p')

    # Une los párrafos para formar el texto del artículo
    letra = '\n'.join(par.text for par in parrafos)

    return titulo, letra

def construir_dataset(categoria: str, urls: list[str]) -> pd.DataFrame:
    """
    Construye un conjunto de datos (DataFrame) a partir de una categoría y una lista de URLs de noticias.

    Args:
        categoria (str): La categoría de las noticias.
        urls (list[str]): Una lista de URLs de noticias a analizar.

    Returns:
        pd.DataFrame: Un DataFrame que contiene las columnas "URL", "Titulo", "Texto" y "Categoría" con la información de las noticias.
    """
    data = {"URL": [], "Titulo": [], "Texto": []}

    for url in urls:
        titulo, letra = web_scraping(url)
        data["URL"].append(url)
        data["Titulo"].append(titulo)
        data["Texto"].append(letra)

    df = pd.DataFrame(data)

    # Agrega la columna de Categoría
    df["Categoría"] = categoria

    return df

urls_policiales = obtener_enlaces_noticias('https://www.rosario3.com/seccion/policiales/', 20)
urls_deporte = obtener_enlaces_noticias('https://www.rosario3.com/seccion/deportes/', 20)
urls_politica = obtener_enlaces_noticias('https://www.rosario3.com/seccion/politica/', 20)
urls_tecnologia = obtener_enlaces_noticias('https://www.rosario3.com/seccion/tecnologia/', 20)

df_policiales = construir_dataset("Policiales", urls_policiales)
df_deporte = construir_dataset("Deporte", urls_deporte)
df_politica = construir_dataset("Politica", urls_politica)
df_tecnologia = construir_dataset("Tecnologia", urls_tecnologia)

df_completo = pd.concat([df_policiales, df_deporte,df_politica,df_tecnologia], ignore_index=True)

df_completo.to_csv("dataset_tp2.csv", index=False)

