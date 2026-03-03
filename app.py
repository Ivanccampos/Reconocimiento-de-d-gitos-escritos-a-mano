import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
import os
import random

# Configuracion de la pagina
st.set_page_config(page_title="Juego de Numeros", layout="centered")

# --- ESTILO CSS PARA COMIC SANS Y LETRAS DE COLORES ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
    
    .comic-font {
        font-family: 'Comic Sans MS', 'Comic Neue', cursive;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        line-height: 1.2;
        margin-bottom: 20px;
    }
    
    .letter {
        display: inline-block;
        padding: 0 2px;
    }

    .comic-text {
        font-family: 'Comic Sans MS', 'Comic Neue', cursive;
        font-size: 20px;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# Funcion para crear titulos con letras de colores aleatorios
def titulo_colores(texto):
    colores = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFF333', '#33FFF3', '#FF8333', '#FF3383']
    html_final = '<div class="comic-font">'
    for letra in texto:
        if letra == " ":
            html_final += '&nbsp;'
        else:
            color_elegido = random.choice(colores)
            html_final += f'<span class="letter" style="color:{color_elegido};">{letra}</span>'
    html_final += '</div>'
    return html_final

# 1. Cargar el modelo ONNX
@st.cache_resource
def cargar_modelo():
    return ort.InferenceSession("modelo_digitos.onnx")

try:
    session = cargar_modelo()
except Exception:
    st.error("No se encontro el archivo modelo_digitos.onnx")
    st.stop()

# --- VENTANA EMERGENTE CON EL GIF RESTAURADO ---
@st.dialog("ADIVINANZA")
def ventana_resultado(prediccion, confianza, probabilidades):
    # Titulo de colores dentro de la ventana
    st.markdown(titulo_colores(f"NUMERO {prediccion}"), unsafe_allow_html=True)
    
    col_izq, col_der = st.columns([1, 1.2])
    
    with col_izq:
        # Buscamos el GIF correspondiente al numero predicho
        ruta_gif = f"Gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
        elif os.path.exists(f"{prediccion}.gif"):
            st.image(f"{prediccion}.gif", use_container_width=True)
        else:
            st.write("✨") # Respaldo visual si falta el GIF
    
    with col_der:
        st.markdown(f"<p class='comic-text'>CONFIANZA: {confianza:.1f}%</p>", unsafe_allow_html=True)
        st.progress(int(confianza))
        
    st.write("---")
    st.markdown("<p class='comic-text'>PUNTUACION DE LOS NUMEROS</p>", unsafe_allow_html=True)
    
    # Grafica de barras colorida
    datos_grafica = pd.DataFrame({
        'Numero': [str(i) for i in range(10)],
        'Valor': probabilidades
    })

    grafica = alt.Chart(datos_grafica).mark_bar(cornerRadius=8).encode(
        x=alt.X('Numero', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Valor', axis=None),
        color=alt.condition(
            alt.datum.Numero == str(prediccion),
            alt.value('#FF4B4B'), # Rojo para el ganador
            alt.value('#4B8BFF')  # Azul para los demas
        )
    ).properties(height=200)

    st.altair_chart(grafica, use_container_width=True)
    
    if st.button("VOLVER A JUGAR"):
        st.rerun()

# --- INTERFAZ PRINCIPAL ---
st.markdown(titulo_colores("ADIVINA EL NUMERO"), unsafe_allow_html=True)

st.markdown("<p class='comic-text' style='text-align:center;'>Dibuja un numero grande en el cuadro</p>", unsafe_allow_html=True)

# Lienzo de dibujo
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=25,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas_infantil",
)

if canvas_result.image_data is not None:
    img_data = canvas_result.image_data.astype('uint8')
    img_pil = Image.fromarray(img_data).convert('L')
    
    st.write("") 
    if st.button("¿QUE NUMERO ES?"):
        # Solo procesar si hay dibujo en el lienzo
        if np.any(np.array(img_pil) > 20):
            # Preprocesamiento para el modelo CNN
            img_res = img_pil.resize((28, 28), Image.LANCZOS)
            img_norm = np.array(img_res).astype('float32') / 255.0
            img_final = img_norm.reshape(1, 28, 28, 1)

            # Inferencia con ONNX Runtime
            input_node = session.get_inputs()[0].name
            output_node = session.get_outputs()[0].name
            pred_raw = session.run([output_node], {input_node: img_final})[0][0]
            
            num_final = np.argmax(pred_raw)
            prob_final = float(pred_raw[num_final] * 100)

            # Mostrar resultado en ventana emergente (Dialog)
            ventana_resultado(num_final, prob_final, pred_raw)
        else:
            st.warning("Dibuja algo primero")

