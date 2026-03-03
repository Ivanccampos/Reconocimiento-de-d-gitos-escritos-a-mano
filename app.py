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

# 2. ESTILO CSS "VINTAGE" Y MEJORAS DE VISIBILIDAD
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

    /* CENTRADO ABSOLUTO DEL CANVAS */
    div.stColumn > div > div > div > div:has(canvas) {
        display: flex !important;
        justify-content: center !important;
        margin: 0 auto !important;
        width: 350px !important;
    }

    [data-testid="stCanvas"] {
        border: 6px solid;
        border-color: #ffffff #808080 #808080 #ffffff !important;
        box-shadow: 10px 10px 0px #000;
        background-color: black !important;
    }

    /* ICONO DE BORRAR Y HERRAMIENTAS ULTRA VISIBLES */
    .stCanvasToolbar {
        justify-content: center !important;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 5px;
        border-radius: 10px;
    }
    
    /* Estilo para los botones de la papelera y deshacer */
    .stCanvasToolbar button {
        background-color: #FFFF00 !important; /* Amarillo fluorescente */
        border: 2px solid black !important;
        border-radius: 5px !important;
        margin: 5px !important;
        width: 45px !important;
        height: 45px !important;
        transition: transform 0.2s;
    }
    
    .stCanvasToolbar button:hover {
        transform: scale(1.2);
        background-color: #FF00FF !important; /* Cambia a rosa al pasar el ratón */
    }

    /* Forzar que los iconos internos sean negros y grandes */
    .stCanvasToolbar button svg {
        fill: #000000 !important;
        color: #000000 !important;
        width: 25px !important;
        height: 25px !important;
    }

    /* Botón Principal Neón */
    @keyframes border-flicker {
        0% { border-color: #FF0000; box-shadow: 0 0 5px #FF0000; }
        50% { border-color: #00FF00; box-shadow: 0 0 20px #00FF00; }
        100% { border-color: #FF0000; box-shadow: 0 0 5px #FF0000; }
    }

    .stButton > button {
        display: block !important;
        margin: 40px auto !important;
        background-color: #000 !important;
        color: #fff !important;
        font-size: 24px !important;
        padding: 15px 40px !important;
        border: 5px solid #FF0000 !important;
        animation: border-flicker 1s infinite;
        border-radius: 0px !important;
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

# 3. CARGAR MODELO
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
    
    if st.button("BORRAR Y JUGAR"):
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
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=28,
        stroke_color="white",
        background_color="black",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )
    
    st.markdown("<p style='text-align:center; font-weight:bold; color:white;'>Escribe un número bien grande</p>", unsafe_allow_html=True)
    
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
            st.warning("¡Primero dibuja algo!")
