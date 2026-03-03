import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
import os

# Configuración de la página
st.set_page_config(page_title="Reconocedor de Dígitos CNN", layout="centered")

# 1. Cargar el modelo ONNX de forma eficiente
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

session = load_model()

# --- FUNCIÓN PARA LA VENTANA EMERGENTE (DIALOG) ---
@st.dialog("Resultado del Reconocimiento")
def mostrar_resultado(prediccion, confianza, probabilidades):
    # Crear dos columnas: una para el GIF animado y otra para el texto del resultado
    col_gif, col_txt = st.columns([1, 1.5])
    
    with col_gif:
        # Intentamos cargar el GIF desde la carpeta 'gifs'
        # El nombre del archivo debe ser 0.gif, 1.gif, etc.
        ruta_gif = f"gifs/{prediccion}.gif"
        
        if os.path.exists(ruta_gif):
            st.image(ruta_gif, use_container_width=True)
        else:
            # Si el GIF no está en una carpeta, lo busca en la raíz
            ruta_raiz = f"{prediccion}.gif"
            if os.path.exists(ruta_raiz):
                st.image(ruta_raiz, use_container_width=True)
            else:
                st.warning(f"No se encontró {prediccion}.gif")

    with col_txt:
        st.write(f"## ¡Es un {prediccion}!")
        st.write(f"**Confianza:** {confianza:.2f}%")
        st.progress(int(confianza))
    
    st.write("---")
    st.write("### Análisis de Probabilidades")
    
    # Preparar datos para la gráfica de Altair
    chart_data = pd.DataFrame({
        'Dígito': [str(i) for i in range(10)],
        'Confianza': probabilidades
    })

    # Gráfica de barras con el ganador resaltado en naranja
    grafica = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Dígito', axis=alt.Axis(labelAngle=0, title="Número")),
        y=alt.Y('Confianza', axis=alt.Axis(title="Probabilidad")),
        color=alt.condition(
            alt.datum.Dígito == str(prediccion),
            alt.value('orange'), # Color del número detectado
            alt.value('steelblue') # Color del resto
        )
    ).properties(height=250)

    st.altair_chart(grafica, use_container_width=True)
    
    if st.button("Cerrar y Limpiar"):
        st.rerun()

# --- INTERFAZ PRINCIPAL ---
st.title("🔢 Reconocedor de Dígitos con CNN")
st.write("Dibuja un número grande y centrado en el cuadro negro.")

# Configuración del Lienzo (Canvas)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=22, # Grosor ideal para MNIST
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Lógica del botón de análisis
if canvas_result.image_data is not None:
    # Convertir datos del canvas a imagen de escala de grises
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    if st.button("Analizar Dibujo"):
        # Verificar si el lienzo contiene dibujo (píxeles blancos)
        if np.any(np.array(img) > 30):
            # 1. Preprocesar (Redimensionar a 28x28 y normalizar)
            img_28 = img.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img_28).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # 2. Inferencia con ONNX
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            result = session.run([output_name], {input_name: img_array})[0][0]
            
            # 3. Obtener resultados
            prediccion = np.argmax(result)
            confianza = float(result[prediccion] * 100)

            # 4. Lanzar ventana emergente con el GIF
            mostrar_resultado(prediccion, confianza, result)
        else:
            st.warning("El lienzo está vacío. Por favor, dibuja un número.")
