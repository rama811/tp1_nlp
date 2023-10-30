import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import textwrap

model_name = "t5-small"  # Modelo T5
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

df = pd.read_csv(r"C:\Users\tomas\OneDrive\Desktop\facultad\4to cuatrimestre\Procesamiento del lenguaje natural\TP_1\dataset_tp2.csv")
df_policiales = df[df["Categoría"] == "Policiales"]
df_futbol = df[df["Categoría"] == "Futbol"]
df_politica = df[df["Categoría"] == "Politica"]
df_tecnologia = df[df["Categoría"] == "Tecnologia"]

categorias = ["Policiales", "Futbol", "Politica", "Tecnologia"]

# Función para clasificar y resumir todas las noticias de todas las categorías
def clasificar_y_resumir_todas_categorias():
    # Combinar todas las noticias de todas las categorías
    todas_noticias = []
    for categoria in [df_policiales, df_futbol, df_politica, df_tecnologia]:
        todas_noticias.extend(categoria["Texto"].tolist())

    # Realizar un resumen para todas las noticias y acumularlos
    resumenes_parciales = []
    for noticia in todas_noticias:
        inputs = tokenizer.encode("summarize: " + noticia, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=100, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
        resumen_parcial = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        resumenes_parciales.append(resumen_parcial)

    # Unir los resúmenes parciales en un resumen general
    resumen_general = "\n".join(resumenes_parciales)

    return resumen_general

# Programa interactivo
while True:
    print("Categorías disponibles:", categorias)
    categoria_seleccionada = input("Selecciona una categoría (o 'salir' para salir): ")
    if categoria_seleccionada.lower() == "salir":
        break
    if categoria_seleccionada in categorias:
        resumen_categorias = clasificar_y_resumir_todas_categorias()
        print(textwrap.fill(resumen_categorias, width=80))
    else:
        print("Categoría no válida.")
