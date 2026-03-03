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

# 2. CSS DE ALTO CONTRASTE (IGNORA EL MODO DEL NAVEGADOR)
st.markdown("""
    <style>
    /* Forzamos el fondo de toda la pantalla a un gris claro neutro */
    .stApp {
        background-color: #E5E7EB !important;
        background-image: url('https://www.transparenttextures.com/patterns/diagmonds-light.png');
    }
    
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
    
    /* Forzamos color de texto negro en toda la app */
    html, body, [class*="st-"], p, h1, h2, h3 {
        font-family: 'Comic Sans MS', 'Comic Neue', cursive !important;
        color: #1F2937 !important;
    }

    /* Título Animado con sombra para legibilidad */
    @keyframes color-change {
        0% { color: #FF5733; }
        25% { color: #22C55E; }
        50% { color: #3B82F6; }
        75% { color: #D946EF; }
        100% { color: #FF5733; }
    }
    .comic-font {
        font-size: 55px;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 0px #FFFFFF;
        margin-bottom: 20px;
    }
    .animated-letter {
        display: inline-block;
        animation: color-change 2s infinite;
    }

    /* CONTENEDOR DEL CANVAS: Centrado y con fondo oscuro fijo */
    div.stColumn > div > div > div > div:has(canvas) {
        display: flex !important;
        justify-content: center !important;
        margin: 0 auto !important;
        width: 370px !important; /* Un poco más ancho que el canvas para el padding */
        background-color: #111827 !important; /* Azul oscuro casi negro */
        padding: 10px !important;
        border-radius: 15px !important;
        border: 4px solid #374151 !important;
    }

    [data-testid="stCanvas"] {
        background-color: #000000 !important;
        border: 2px solid #4B5563 !important;
    }

    /* BARRA DE HERRAMIENTAS: Siempre visible */
    .stCanvasToolbar {
        justify-content: center !important;
        background-color: #1F2937 !important;
        padding: 5px !important;
        border-radius: 8px !important;
    }
    
    .stCanvasToolbar button {
        background-color: #FACC15 !important; /* Amarillo brillante */
        border: 2px solid #000 !important;
        margin: 5px !important;
        color: #000 !important;
    }
    
    .stCanvasToolbar button svg {
        fill: #000 !important;
    }

    /* BOTÓN DE PREDICCIÓN */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .stButton > button {
        display: block !important;
        margin: 30px auto !important;
        background-color: #EF4444 !important; /* Rojo */
        color: white !important;
        font-size: 24px !important;
        border: 4px solid #000 !important;
        border-radius: 10px !important;
        box-shadow: 5px 5px 0px #000 !important;
        animation: pulse 2s infinite;
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

# 3. MODELO Y LÓGICA
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

try:
    session = load_model()
except Exception as e:
    st.error(f"Error: {e}")

@st.dialog("¡RESULTADO!")
def mostrar_resultado(prediccion, confianza, probabilidades):
    st.markdown(titulo_animado(f"ES EL {prediccion}"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(f"Gifs/{prediccion}.gif"):
            st.image(f"Gifs/{prediccion}.gif")
    with col2:
        st.metric("Confianza", f"{confianza:.1f}%")
        st.progress(int(confianza))
    
    chart_data = pd.DataFrame({'N': [str(i) for i in range(10)], 'P': probabilidades})
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='N', y='P', color=alt.condition(alt.datum.N == str(prediccion), alt.value('#EF4444'), alt.value('#3B82F6'))
    ).properties(height=150)
    st.altair_chart(chart, use_container_width=True)
    if st.button("VOLVER"):
        st.rerun()

# --- INTERFAZ ---
st.markdown(titulo_animado("ADIVINO TU NUMERO"), unsafe_allow_html=True)

col_izq, col_centro, col_der = st.columns([1, 1.5, 1])

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
    
    st.markdown("<p style='text-align:center; font-weight:bold;'>Dibuja el número aquí arriba</p>", unsafe_allow_html=True)
    
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
