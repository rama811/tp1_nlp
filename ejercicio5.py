import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import textwrap

model_name = "t5-small"  # Modelo T5
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

df = pd.read_csv(r"C:\Users\tomas\OneDrive\Desktop\facultad\4to cuatrimestre\Procesamiento del lenguaje natural\TP_1\dataset_tp2.csv")  # Reemplaza con tu ruta

categorias = df["Categoría"].unique()

def resumir_noticias(noticias, tokenizer, model):
    resumenes = []
    for noticia in noticias:
        inputs = tokenizer.encode("summarize: " + noticia, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=False)
        resumen = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        resumenes.append(resumen)
    return resumenes

while True:
    print("Categorías disponibles:", categorias)
    categoria_seleccionada = input("Selecciona una categoría (o 'salir' para salir): ")
    if categoria_seleccionada.lower() == "salir":
        break
    if categoria_seleccionada in categorias:
        df_categoria = df[df["Categoría"] == categoria_seleccionada]
        noticias_categoria = df_categoria["Texto"].tolist()
        resumenes_noticias = resumir_noticias(noticias_categoria, tokenizer, model)
        
        print(f"Resúmenes de la categoría {categoria_seleccionada}:")
        for i, resumen in enumerate(resumenes_noticias, start=1):
            print(f"Resumen de noticia {i} - {resumen}")
    else:
        print("Categoría no válida.")
