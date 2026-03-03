import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
import os

# Configuración de la página con un toque divertido
st.set_page_config(page_title="🎨 Mi Primer Robot Inteligente", layout="centered")

# 1. Cargar el cerebro del Robot (Modelo ONNX)
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

session = load_model()

# --- ¡LA VENTANA MÁGICA DE RESULTADOS! ---
@st.dialog("✨ ¡MIRA LO QUE ENCONTRÉ! ✨")
def mostrar_resultado(prediccion, confianza, probabilidades):
    # Si el robot está muy seguro, ¡celebramos con globos!
    if confianza > 80:
        st.balloons()
    
    col_gif, col_txt = st.columns([1, 1.5])
    
    with col_gif:
        # Buscamos a tu nuevo amigo animado
        ruta_gif = f"gifs/{prediccion}.gif"
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
        else:
            st.write("🌈") # Un arcoíris de respaldo

    with col_txt:
        st.write(f"## ¡HOLA! Soy el número **{prediccion}**")
        st.write(f"¡Estoy un **{confianza:.1f}%** seguro de que me dibujaste!")
        st.progress(int(confianza))
    
    st.write("---")
    st.write("📊 **¿Qué otros números pensó mi cerebro?**")
    
    # Gráfica de colores para niños
    chart_data = pd.DataFrame({
        'Número': [str(i) for i in range(10)],
        'Puntaje': probabilidades
    })

    grafica = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
        x=alt.X('Número', axis=alt.Axis(labelAngle=0, title="¿Qué número es?")),
        y=alt.Y('Puntaje', axis=alt.Axis(title="Nivel de magia")),
        color=alt.condition(
            alt.datum.Número == str(prediccion),
            alt.value('#FF4B4B'), # Rojo divertido para el ganador
            alt.value('#FFD700')  # Dorado para los demás
        )
    ).properties(height=250)

    st.altair_chart(grafica, use_container_width=True)
    
    if st.button("✨ ¡QUIERO DIBUJAR OTRO! ✨"):
        st.rerun()

# --- PANTALLA PRINCIPAL ---
st.title("🤖 ¡Hola! Soy tu Robot Dibujante")
st.write("### 🎨 ¡Dibuja un número aquí abajo y trataré de adivinarlo!")

# Instrucciones divertidas
st.info("Usa tu ratón o tu dedo para dibujar un número **GRANDE** y bien clarito.")

# El pizarrón mágico (Lienzo)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=25, # Un pincel un poco más gordito
    stroke_color="#FFFFFF",
    background_color="#262730", # Fondo oscuro estilo pizarra
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Botón mágico para activar al Robot
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    st.write("") # Espacio
    if st.button("🚀 ¡ADIVINA MI NÚMERO, ROBOT!"):
        # ¿Hay algún dibujo?
        if np.any(np.array(img) > 30):
            with st.spinner('👀 El Robot está mirando tu dibujo...'):
                # Procesar imagen
                img_28 = img.resize((28, 28), Image.LANCZOS)
                img_array = np.array(img_28).astype('float32') / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                # El Robot piensa...
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                result = session.run([output_name], {input_name: img_array})[0][0]
                
                prediccion = np.argmax(result)
                confianza = float(result[prediccion] * 100)

                # ¡Mostrar la sorpresa!
                mostrar_resultado(prediccion, confianza, result)
        else:
            st.warning("¡Ups! El pizarrón está vacío. ¡Dibuja algo divertido! 🖍️")

st.markdown("---")
st.caption("Hecho con ❤️ para pequeños científicos")
