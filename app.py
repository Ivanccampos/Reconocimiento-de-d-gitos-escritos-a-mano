import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
import os
import random

# Configuración de la página
st.set_page_config(page_title="Juego de Numeros", layout="centered")

# --- ESTILO CSS RETRO Y ANIMACIONES ---
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

    @keyframes color-change {
        0% { color: #FF5733; }
        25% { color: #33FF57; }
        50% { color: #3357FF; }
        75% { color: #F333FF; }
        100% { color: #FF5733; }
    }

    .comic-font {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 0px #000;
        margin-bottom: 20px;
    }
    
    .animated-letter {
        display: inline-block;
        animation: color-change 2s infinite;
    }

    [data-testid="stCanvas"] {
        border: 4px solid;
        border-color: #ffffff #808080 #808080 #ffffff !important;
        box-shadow: 6px 6px 0px #000;
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

# Cargar modelo
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

session = load_model()

# --- VENTANA DE RESULTADO ---
@st.dialog("RESULTADO")
def mostrar_resultado(prediccion, confianza, probabilidades):
    st.markdown(titulo_animado(f"NUMERO {prediccion}"), unsafe_allow_html=True)
    col_gif, col_txt = st.columns([1, 1.2])
    with col_gif:
        ruta_gif = f"Gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
    with col_der:
        st.write(f"### CONFIANZA: {confianza:.1f}%")
        st.progress(int(confianza))
    
    st.write("---")
    chart_data = pd.DataFrame({'Numero': [str(i) for i in range(10)], 'Puntaje': probabilidades})
    grafica = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Numero', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Puntaje', axis=None),
        color=alt.condition(alt.datum.Numero == str(prediccion), alt.value('#FF4B4B'), alt.value('#4B8BFF'))
    ).properties(height=150)
    st.altair_chart(grafica, use_container_width=True)
    if st.button("VOLVER A JUGAR"):
        st.rerun()

# --- INTERFAZ PRINCIPAL ---
st.markdown(titulo_animado("ADIVINA EL NUMERO"), unsafe_allow_html=True)

# Fila superior de imágenes y Pocoyo
col_a, col_b, col_c = st.columns(3)
with col_a:
    if os.path.exists("Gifs/0.gif"):
        st.image("Gifs/0.gif", caption="¡Hola!", use_container_width=True)
with col_b:
    # Pocoyo Dance GIF de Tenor
    st.image("https://media.tenor.com/pocoyo-dance.gif", use_container_width=True)
with col_c:
    if os.path.exists("Gifs/image_992305.png"):
        st.image("Gifs/image_992305.png", use_container_width=True)

st.write("<p style='text-align:center;'>Dibuja un numero grande abajo</p>", unsafe_allow_html=True)

# Centrado del Canvas
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=25,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    if st.button("¿QUE NUMERO ES?"):
        if np.any(np.array(img) > 20):
            img_28 = img.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img_28).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            res = session.run(None, {session.get_inputs()[0].name: img_array})[0][0]
            pred = np.argmax(res)
            conf = float(res[pred] * 100)
            mostrar_resultado(pred, conf, res)
