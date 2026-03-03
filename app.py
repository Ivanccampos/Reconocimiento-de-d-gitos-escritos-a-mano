import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
import os

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Juego de Numeros", layout="wide")

# 2. CSS CON GRID OSCURO Y ALTO CONTRASTE
st.markdown("""
    <style>
    /* Fondo con Grid/Mosaico en tonalidad oscura */
    .stApp {
        background-color: #1a1c23 !important; /* Gris casi negro */
        background-image: url('https://www.transparenttextures.com/patterns/diagmonds-light.png');
        background-repeat: repeat;
        background-attachment: fixed;
    }
    
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
    
    /* Forzamos texto claro para que se lea bien sobre el fondo oscuro */
    html, body, [class*="st-"], p, h1, h2, h3, span, label {
        font-family: 'Comic Sans MS', 'Comic Neue', cursive !important;
        color: #ffffff !important;
    }

    /* Título Animado */
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
        margin-bottom: 20px;
    }
    .animated-letter {
        display: inline-block;
        animation: color-change 2s infinite;
    }

    /* CENTRADO DEL CANVAS */
    div.stColumn > div > div > div > div:has(canvas) {
        display: flex !important;
        justify-content: center !important;
        margin: 0 auto !important;
        width: 350px !important;
        background-color: #000000 !important;
        border: 4px solid #444 !important;
        border-radius: 10px;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
    }

    [data-testid="stCanvas"] {
        background-color: #000000 !important;
    }

    /* ICONOS DE HERRAMIENTAS ULTRA VISIBLES */
    .stCanvasToolbar {
        justify-content: center !important;
        background-color: #333 !important;
        border-radius: 8px !important;
        padding: 5px !important;
    }
    .stCanvasToolbar button {
        background-color: #FFFF00 !important; /* Amarillo */
        border: 2px solid #000 !important;
        margin: 5px !important;
    }
    .stCanvasToolbar button svg {
        fill: #000 !important;
    }

    /* BOTÓN PRINCIPAL NEÓN */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    .stButton > button {
        display: block !important;
        margin: 30px auto !important;
        background-color: #ff0000 !important;
        color: white !important;
        font-size: 24px !important;
        border: 3px solid #fff !important;
        border-radius: 10px !important;
        padding: 15px 40px !important;
        animation: pulse-red 2s infinite;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

def titulo_animado(texto):
    html_title = f'<div class="comic-font">'
    for i, char in enumerate(texto):
        if char == " ":
            html_title += '&nbsp;'
        else:
            delay = i * 0.1
            html_title += f'<span class="animated-letter" style="animation-delay: {delay}s;">{char}</span>'
    html_title += '</div>'
    return html_title

# 3. CARGAR MODELO
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

try:
    session = load_model()
except Exception as e:
    st.error(f"Error: {e}")

# --- DIÁLOGO DE RESULTADO ---
@st.dialog("¡RESULTADO!")
def mostrar_resultado(prediccion, confianza, probabilidades):
    st.markdown(titulo_animado(f"NUMERO {prediccion}"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ruta_gif = f"Gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif)
    with col2:
        st.write(f"### Confianza: {confianza:.1f}%")
        st.progress(int(confianza))
    
    st.write("---")
    # Gráfica de probabilidades
    chart_data = pd.DataFrame({'Número': [str(i) for i in range(10)], 'Probabilidad': probabilidades})
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Número', axis=alt.Axis(labelAngle=0, labelColor='white')),
        y=alt.Y('Probabilidad', axis=None),
        color=alt.condition(alt.datum.Número == str(prediccion), alt.value('#FF4B4B'), alt.value('#4B8BFF'))
    ).properties(height=150).configure_view(strokeOpacity=0)
    st.altair_chart(chart, use_container_width=True)
    
    if st.button("INTENTAR OTRA VEZ"):
        st.rerun()

# --- ESTRUCTURA ---
st.markdown(titulo_animado("ADIVINO TU NUMERO"), unsafe_allow_html=True)

col_izq, col_centro, col_der = st.columns([1, 1.2, 1])

with col_izq:
    if os.path.exists("Gifs/pollo.png"):
        st.image("Gifs/pollo.png")

with col_der:
    if os.path.exists("Gifs/brsm.png"):
        st.image("Gifs/brsm.png")

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
        display_toolbar=True,
    )
    
    st.markdown("<p style='text-align:center; font-weight:bold; font-size:18px;'>Escribe un número aquí arriba</p>", unsafe_allow_html=True)
    
    if st.button("¿QUE NUMERO ES?"):
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        if np.any(np.array(img) > 20):
            img_28 = img.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img_28).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            res = session.run(None, {session.get_inputs()[0].name: img_array})[0][0]
            pred = np.argmax(res)
            mostrar_resultado(pred, float(res[pred]*100), res)
        else:
            st.warning("¡Dibuja algo primero!")
