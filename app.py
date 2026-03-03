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

# 2. ESTILO CSS COMPATIBLE CON MODO CLARO Y OSCURO
st.markdown("""
    <style>
    /* Forzamos que el fondo de la app sea siempre el mismo */
    .stApp {
        background-color: #f0f2f6; /* Gris muy claro para modo claro */
        background-image: url('https://www.transparenttextures.com/patterns/diagmonds-light.png');
        background-repeat: repeat;
        background-attachment: fixed;
    }
    
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Comic Sans MS', 'Comic Neue', cursive !important;
        color: #31333F; /* Color de texto oscuro para lectura fácil */
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
        text-shadow: 2px 2px 0px #ddd;
        margin-bottom: 20px;
    }
    
    .animated-letter {
        display: inline-block;
        animation: color-change 2s infinite;
    }

    /* CENTRADO Y CONTENEDOR DEL CANVAS */
    div.stColumn > div > div > div > div:has(canvas) {
        display: flex !important;
        justify-content: center !important;
        margin: 0 auto !important;
        width: 350px !important;
        background-color: #262730; /* Fondo oscuro para el área del canvas */
        padding: 10px;
        border-radius: 10px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
    }

    [data-testid="stCanvas"] {
        border: 4px solid #ffffff !important;
        background-color: black !important;
    }

    /* HERRAMIENTAS ULTRA-VISIBLES */
    .stCanvasToolbar {
        justify-content: center !important;
        background-color: #333 !important; /* Fondo oscuro para la barra */
        padding: 8px !important;
        border-radius: 8px !important;
        margin-top: 5px !important;
    }
    
    .stCanvasToolbar button {
        background-color: #FFD700 !important; /* Oro/Amarillo para resaltar */
        border: 2px solid #000 !important;
        margin: 0 5px !important;
        width: 40px !important;
        height: 40px !important;
    }
    
    .stCanvasToolbar button svg {
        fill: #000 !important; /* Iconos negros sobre fondo amarillo */
        width: 20px !important;
        height: 20px !important;
    }

    /* Botón Principal */
    @keyframes border-flicker {
        0% { border-color: #FF0000; }
        50% { border-color: #00FF00; }
        100% { border-color: #FF0000; }
    }

    .stButton > button {
        display: block !important;
        margin: 30px auto !important;
        background-color: #000 !important;
        color: #fff !important;
        font-size: 22px !important;
        border: 4px solid #FF0000 !important;
        animation: border-flicker 1s infinite;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ... (Funciones de carga de modelo y título animado se mantienen igual) ...

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

@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

try:
    session = load_model()
except Exception as e:
    st.error(f"Error: {e}")

# --- DIÁLOGO DE RESULTADO ---
@st.dialog("¡TE HE PILLADO!")
def mostrar_resultado(prediccion, confianza, probabilidades):
    st.markdown(titulo_animado(f"NUMERO {prediccion}"), unsafe_allow_html=True)
    col_gif, col_txt = st.columns([1, 1])
    with col_gif:
        ruta_gif = f"Gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
    with col_txt:
        st.write(f"### CONFIANZA: {confianza:.1f}%")
        st.progress(int(confianza))
    st.write("---")
    chart_data = pd.DataFrame({'Num': [str(i) for i in range(10)], 'Prob': probabilidades})
    grafica = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Num', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Prob', axis=None),
        color=alt.condition(alt.datum.Num == str(prediccion), alt.value('#FF4B4B'), alt.value('#4B8BFF'))
    ).properties(height=150)
    st.altair_chart(grafica, use_container_width=True)
    if st.button("INTENTAR DE NUEVO"):
        st.rerun()

# --- ESTRUCTURA PRINCIPAL ---
st.markdown(titulo_animado("ADIVINO TU NUMERO"), unsafe_allow_html=True)

col_izq, col_centro, col_der = st.columns([1, 1.2, 1])

with col_izq:
    if os.path.exists("Gifs/pollo.png"):
        st.image("Gifs/pollo.png", use_container_width=True)

with col_der:
    if os.path.exists("Gifs/brsm.png"):
        st.image("Gifs/brsm.png", use_container_width=True)

with col_centro:
    # Envolvemos el canvas en un fondo oscuro para que resalte en el tema claro
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
    
    st.markdown("<p style='text-align:center; font-weight:bold;'>Escribe un número bien grande</p>", unsafe_allow_html=True)
    
    if st.button("¿QUE NUMERO ES?"):
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        if np.any(np.array(img) > 20):
            img_28 = img.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img_28).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            res = session.run(None, {session.get_inputs()[0].name: img_array})[0][0]
            pred = np.argmax(res)
            conf = float(res[pred] * 100)
            mostrar_resultado(pred, conf, res)
        else:
            st.warning("¡Dibuja algo primero!")
