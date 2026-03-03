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

# --- ESTILO CSS PARA LETRAS DE COLORES Y COMIC SANS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&display=swap');
    
    .comic-font {
        font-family: 'Comic Sans MS', 'Comic Neue', cursive;
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .letter { display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

def colored_title(text):
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFF333', '#33FFF3', '#FF8333']
    html_title = '<div class="comic-font">'
    for char in text:
        if char == " ":
            html_title += '&nbsp;'
        else:
            color = random.choice(colors)
            html_title += f'<span class="letter" style="color:{color};">{char}</span>'
    html_title += '</div>'
    return html_title

# 1. Cargar el modelo
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digits.onnx")

try:
    session = load_model()
except:
    st.error("Falta el archivo modelo_digits.onnx")

# --- VENTANA DE RESULTADO ---
@st.dialog("RESULTADO")
def mostrar_resultado(prediccion, confianza, probabilidades):
    if confianza > 80:
        st.balloons()
    
    # Título colorido dentro del diálogo
    st.markdown(colored_title(f"NUMERO {prediccion}"), unsafe_allow_html=True)
    
    col_gif, col_txt = st.columns([1, 1.5])
    
    with col_gif:
        ruta_gif = f"gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
        else:
            st.write("---")

    with col_txt:
        st.write(f"CONFIANZA: {confianza:.1f}%")
        st.progress(int(confianza))
    
    st.write("GRAFICA DE PUNTOS")
    
    chart_data = pd.DataFrame({
        'Numero': [str(i) for i in range(10)],
        'Puntaje': probabilidades
    })

    grafica = alt.Chart(chart_data).mark_bar(cornerRadius=10).encode(
        x=alt.X('Numero', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Puntaje'),
        color=alt.condition(
            alt.datum.Numero == str(prediccion),
            alt.value('#FF4B4B'), 
            alt.value('#4BFF4B')  
        )
    ).properties(height=250)

    st.altair_chart(grafica, use_container_width=True)
    
    if st.button("VOLVER A JUGAR"):
        st.rerun()

# --- CUERPO PRINCIPAL ---
st.markdown(colored_title("ADIVINA EL NUMERO"), unsafe_allow_html=True)

st.write("Dibuja un numero muy grande en el cuadro negro")

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
    
    if st.button("ADIVINAR"):
        if np.any(np.array(img) > 30):
            img_28 = img.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img_28).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            result = session.run([output_name], {input_name: img_array})[0][0]
            
            prediccion = np.argmax(result)
            confianza = float(result[prediccion] * 100)

            mostrar_resultado(prediccion, confianza, result)
        else:
            st.write("DIBUJA ALGO PRIMERO")
