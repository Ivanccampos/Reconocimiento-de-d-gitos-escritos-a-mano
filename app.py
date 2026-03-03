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

# 2. ESTILO CSS RETRO COMPLETO
st.markdown("""
    <style>
    /* Fondo de mosaico */
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
        margin-bottom: 10px;
    }
    
    .animated-letter {
        display: inline-block;
        animation: color-change 2s infinite;
    }

    /* CENTRADO DEL CANVAS Y ELIMINACIÓN DE CUADRO NEGRO SOBRANTE */
    [data-testid="stCanvas"] {
        display: table !important; /* Ajusta el ancho al contenido (350px) */
        margin: 0 auto !important; /* Centra el componente */
        border: 5px solid;
        border-color: #ffffff #808080 #808080 #ffffff !important;
        box-shadow: 8px 8px 0px #000;
    }

    /* ICONOS DE LA HERRAMIENTA EN MODO CLARO/OSCURO */
    /* Invertimos el color de los iconos de la papelera/deshacer para que se vean */
    .stCanvasToolbar button {
        background-color: #444 !important;
        border-radius: 5px;
        margin: 2px;
    }
    .stCanvasToolbar button svg {
        fill: white !important;
        color: white !important;
    }

    /* BOTÓN CON BORDE PARPADEANTE NEÓN */
    @keyframes border-flicker {
        0% { border-color: #FF0000; box-shadow: 0 0 5px #FF0000; }
        33% { border-color: #00FF00; box-shadow: 0 0 15px #00FF00; }
        66% { border-color: #0000FF; box-shadow: 0 0 5px #0000FF; }
        100% { border-color: #FF0000; box-shadow: 0 0 15px #FF0000; }
    }

    .stButton > button {
        display: block;
        margin: 20px auto;
        background-color: #000 !important;
        color: #fff !important;
        font-weight: bold;
        font-size: 22px !important;
        padding: 15px 30px !important;
        border: 4px solid #FF0000 !important;
        animation: border-flicker 1.5s infinite;
        text-transform: uppercase;
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
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# --- VENTANA DE RESULTADO ---
@st.dialog("¡MIRA EL RESULTADO!")
def mostrar_resultado(prediccion, confianza, probabilidades):
    st.markdown(titulo_animado(f"NUMERO {prediccion}"), unsafe_allow_html=True)
    
    col_gif, col_txt = st.columns([1, 1.2])
    with col_gif:
        ruta_gif = f"Gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
        else:
            st.write("🌈")

    with col_txt:
        st.write(f"### CONFIANZA: {confianza:.1f}%")
        st.progress(int(confianza))
    
    st.write("---")
    chart_data = pd.DataFrame({
        'Número': [str(i) for i in range(10)],
        'Probabilidad': probabilidades
    })

    grafica = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Número', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Probabilidad', axis=None),
        color=alt.condition(
            alt.datum.Número == str(prediccion),
            alt.value('#FF4B4B'), 
            alt.value('#4B8BFF')
        )
    ).properties(height=150)

    st.altair_chart(grafica, use_container_width=True)
    
    if st.button("INTENTAR DE NUEVO"):
        st.rerun()

# --- INTERFAZ PRINCIPAL ---
st.markdown(titulo_animado("ADIVINO TU NUMERO"), unsafe_allow_html=True)

# 4. DISPOSICIÓN: [POLLO] [CANVAS CENTRADO] [BARRIO SÉSAMO]
col_izq, col_centro, col_der = st.columns([1, 2, 1])

with col_izq:
    if os.path.exists("Gifs/pollo.png"):
        st.image("Gifs/pollo.png", use_container_width=True)

with col_der:
    if os.path.exists("Gifs/brsm.png"):
        st.image("Gifs/brsm.png", use_container_width=True)

with col_centro:
    # Contenedor para el canvas
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
    
    st.write("<p style='text-align:center; font-weight:bold;'>Dibuja un numero grande arriba</p>", unsafe_allow_html=True)
    
    if st.button("¿QUE NUMERO ES?"):
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        if np.any(np.array(img) > 20):
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
            st.warning("¡Dibuja algo primero!")
