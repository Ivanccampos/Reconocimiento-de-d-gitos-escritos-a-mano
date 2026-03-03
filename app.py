import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt

st.set_page_config(page_title="Reconocedor de Dígitos CNN", layout="centered")

# 1. Cargar el modelo ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

session = load_model()

# --- FUNCIÓN PARA LA VENTANA EMERGENTE (DIALOG) ---
@st.dialog("Resultado del Reconocimiento")
def mostrar_resultado(prediccion, confianza, probabilidades):
    st.write(f"### ¡Es un número {prediccion}!")
    st.progress(int(confianza))
    st.write(f"Confianza: **{confianza:.2f}%**")
    
    st.write("---")
    st.write("Análisis de probabilidades:")
    
    # Datos para la gráfica
    chart_data = pd.DataFrame({
        'Dígito': [str(i) for i in range(10)],
        'Confianza': probabilidades
    })

    # Gráfica con Altair
    grafica = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Dígito', axis=alt.Axis(labelAngle=0, title="Dígito")),
        y=alt.Y('Confianza', axis=alt.Axis(title="Nivel de Confianza")),
        color=alt.condition(
            alt.datum.Dígito == str(prediccion),
            alt.value('orange'),
            alt.value('steelblue')
        )
    ).properties(height=300)

    st.altair_chart(grafica, use_container_width=True)
    
    if st.button("Cerrar"):
        st.rerun()

# --- INTERFAZ PRINCIPAL ---
st.title("🔢 Reconocedor de Dígitos con CNN")
st.write("Dibuja un número grande y centrado en el cuadro negro.")

# Configuración del Lienzo
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Lógica del botón de análisis
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    if st.button("Analizar Dibujo"):
        # Aseguramos que el bloque de abajo esté bien indentado
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

            # Llamar a la ventana emergente
            mostrar_resultado(prediccion, confianza, result)
        else:
            st.warning("Por favor, dibuja algo primero.")
