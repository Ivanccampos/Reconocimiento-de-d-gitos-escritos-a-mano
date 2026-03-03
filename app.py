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

# --- ESTILO CSS RETRO Y CENTRADO ---
st.markdown("""
    <style>
    /* Fondo de mosaico repetitivo tipo early internet */
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/diagmonds-light.png');
        background-repeat: repeat;
        background-attachment: fixed;
    }

    /* Fuente Comic Sans */
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Comic Sans MS', 'Comic Neue', cursive !important;
    }

    .comic-font {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 0px #000;
        margin-bottom: 20px;
    }

    /* Bordes estilo Windows 95 para el lienzo */
    [data-testid="stCanvas"] {
        border: 3px solid;
        border-color: #ffffff #808080 #808080 #ffffff !important;
        box-shadow: 4px 4px 0px #000;
    }
    
    /* Centrar botones */
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

def titulo_colores(texto):
    colores = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFF333', '#33FFF3', '#FF8333']
    html_title = '<div class="comic-font">'
    for char in texto:
        if char == " ":
            html_title += '&nbsp;'
        else:
            color = random.choice(colores)
            html_title += f'<span style="color:{color};">{char}</span>'
    html_title += '</div>'
    return html_title

# 1. Cargar el modelo ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

session = load_model()

# --- VENTANA DE RESULTADO ---
@st.dialog("RESULTADO")
def mostrar_resultado(prediccion, confianza, probabilidades):
    st.markdown(titulo_colores(f"NUMERO {prediccion}"), unsafe_allow_html=True)
    
    col_gif, col_txt = st.columns([1, 1.5])
    with col_gif:
        ruta_gif = f"Gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
        else:
            st.write("✨")

    with col_txt:
        st.write(f"CONFIANZA: {confianza:.1f}%")
        st.progress(int(confianza))
    
    st.write("---")
    chart_data = pd.DataFrame({
        'Numero': [str(i) for i in range(10)],
        'Puntaje': probabilidades
    })

    grafica = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Numero', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Puntaje', axis=None),
        color=alt.condition(
            alt.datum.Numero == str(prediccion),
            alt.value('#FF4B4B'), 
            alt.value('#4B8BFF')  
        )
    ).properties(height=200)

    st.altair_chart(grafica, use_container_width=True)
    
    if st.button("VOLVER A JUGAR"):
        st.rerun()

# --- INTERFAZ PRINCIPAL ---
st.markdown(titulo_colores("ADIVINA EL NUMERO"), unsafe_allow_html=True)
st.write("<p style='text-align:center;'>Dibuja un numero grande en el cuadro negro</p>", unsafe_allow_html=True)

# 2. Centrado del Canvas usando columnas
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=25,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

# 3. Procesamiento y Predicción
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    st.write("") # Espacio estético
    if st.button("¿QUE NUMERO ES?"):
        if np.any(np.array(img) > 20):
            # Preprocesar
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
            st.warning("Dibuja algo primero")
