import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
import os
import random

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Juego de Numeros", layout="wide")

# 2. ESTILO CSS RETRO (Fondo de mosaico y Comic Sans)
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/diagmonds-light.png');
        background-repeat: repeat;
        background-attachment: fixed;
    }
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Comic Sans MS', 'Comic Neue', cursive !important;
    }

    /* Animación de colores para el título */
    @keyframes color-change {
        0% { color: #FF5733; }
        25% { color: #33FF57; }
        50% { color: #3357FF; }
        75% { color: #F333FF; }
        100% { color: #FF5733; }
    }

    .comic-font {
        font-size: 55px;
        font-weight: bold;
        text-align: center;
        text-shadow: 3px 3px 0px #000;
        margin-bottom: 10px;
    }
    
    .animated-letter {
        display: inline-block;
        animation: color-change 2s infinite;
    }

    /* Estilo del Canvas con bordes 3D retro */
    [data-testid="stCanvas"] {
        border: 4px solid;
        border-color: #ffffff #808080 #808080 #ffffff !important;
        box-shadow: 6px 6px 0px #000;
        margin: 0 auto;
    }

    /* Centrar el botón */
    .stButton > button {
        display: block;
        margin: 20px auto;
        background-color: #ffeb3b;
        border: 2px solid black;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def titulo_animado(texto):
    html_title = '<div class="comic-font">'
    for i, char in enumerate(texto):
        if char == " ":
            html_title += '&nbsp;'
        else:
            delay = i * 0.1
            html_title += f'<span class="animated-letter" style="animation-delay: {delay}s;">{char}</span>'
    html_title += '</div>'
    return html_title

# 3. CARGAR MODELO ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

try:
    session = load_model()
except:
    st.error("Archivo modelo_digitos.onnx no encontrado")

# --- VENTANA DE RESULTADO ---
@st.dialog("¡MIRA!")
def mostrar_resultado(prediccion, confianza, probabilidades):
    st.markdown(titulo_animado(f"NUMERO {prediccion}"), unsafe_allow_html=True)
    
    col_gif, col_txt = st.columns([1, 1.2])
    with col_gif:
        # Busca el GIF del resultado (0.gif, 1.gif...)
        ruta_gif = f"Gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
        else:
            st.write("🌈")

    with col_txt:
        st.write(f"### CONFIANZA: {confianza:.1f}%")
        st.progress(int(confianza))
    
    if st.button("VOLVER A JUGAR"):
        st.rerun()

# --- INTERFAZ PRINCIPAL ---
st.markdown(titulo_animado("ADIVINA EL NUMERO"), unsafe_allow_html=True)

# 4. DISPOSICIÓN DE IMÁGENES (.PNG) Y CANVAS
# Usamos 3 columnas para que las imágenes rodeen al lienzo sin moverlo
col_izq, col_centro, col_der = st.columns([1, 2, 1])

with col_izq:
    # Imagen del Pollo
    if os.path.exists("Gifs/pollo.png"):
        st.image("Gifs/pollo.png", use_container_width=True)
    st.write(" ") # Espacio
    # Imagen de Barrio Sésamo
    if os.path.exists("Gifs/brsm.png"):
        st.image("Gifs/brsm.png", use_container_width=True)

with col_der:
    # GIF de Pocoyó desde enlace directo
    st.image("https://media.tenor.com/On7_2777698AAAAC/pocoyo-dance.gif", use_container_width=True)
    st.write(" ") # Espacio
    # Tu imagen decorativa adicional
    if os.path.exists("Gifs/image_992305.png"):
        st.image("Gifs/image_992305.png", use_container_width=True)

with col_centro:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=25,
        stroke_color="white",
        background_color="black",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    st.write("<p style='text-align:center;'>Dibuja aquí arriba</p>", unsafe_allow_html=True)
    
    if st.button("¿QUE NUMERO ES?"):
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        if np.any(np.array(img) > 20):
            # Preprocesar para el modelo (28x28)
            img_28 = img.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img_28).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # Inferencia
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            result = session.run([output_name], {input_name: img_array})[0][0]
            
            prediccion = np.argmax(result)
            confianza = float(result[prediccion] * 100)

            mostrar_resultado(prediccion, confianza, result)
        else:
            st.warning("¡Dibuja algo primero!")
