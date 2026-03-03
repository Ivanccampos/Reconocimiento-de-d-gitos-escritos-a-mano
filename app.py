import os
import sys
import subprocess

# --- BLOQUE DE SEGURIDAD PARA DEPENDENCIAS ---
# Esto asegura que la app funcione incluso si Streamlit Cloud ignora el requirements.txt
try:
    import streamlit_drawable_canvas
    import onnxruntime as ort
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "streamlit-drawable-canvas==0.9.3", 
                          "onnxruntime==1.17.1", 
                          "numpy<2.0.0", "Pillow", "matplotlib"])
    import streamlit_drawable_canvas
    import onnxruntime as ort

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="IA Reconocedora de Números", layout="centered")

st.title("🔢 Reconocedor de Dígitos (CNN + ONNX)")
st.markdown("""
Dibuja un número **grande y centrado** en el recuadro negro. 
El sistema usará una Red Neuronal Convolucional para identificarlo.
""")

# 1. Cargar el modelo ONNX de forma eficiente
@st.cache_resource
def load_model():
    # Asegúrate de que 'modelo_digitos.onnx' esté en la misma carpeta que este script
    return ort.InferenceSession("modelo_digitos.onnx")

try:
    session = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_digitos.onnx' esté en el repositorio.")
    st.stop()

# 2. Configuración del área de dibujo
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Lienzo de dibujo")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20, # Grosor para que la IA lo vea bien
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Limpiar Lienzo"):
        st.rerun()

# 3. Procesamiento y Predicción
if canvas_result.image_data is not None:
    # Convertir el dibujo a imagen PIL (Escala de grises)
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    # Comprobar si el usuario ha dibujado algo (si no todo es negro)
    if np.any(np.array(img) > 0):
        # Preprocesar para el modelo (28x28 píxeles)
        img_28 = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_28).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Inferencia con ONNX
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img_array})[0][0]
        
        prediccion = np.argmax(result)
        confianza = result[prediccion] * 100

        with col2:
            st.subheader("Predicción")
            st.metric(label="Número detectado", value=str(prediccion), delta=f"{confianza:.1f}% confianza")
            
            # --- GRÁFICO DE BARRAS CON EJE X VERTICAL ---
            fig, ax = plt.subplots(figsize=(5, 4))
            etiquetas = [str(i) for i in range(10)]
            colores = ['skyblue' if i != prediccion else 'orange' for i in range(10)]
            
            ax.bar(etiquetas, result, color=colores)
            ax.set_ylim(0, 1.1) # Un poco más de 1 para que quepan las etiquetas
            ax.set_ylabel("Probabilidad")
            
            # Rotar etiquetas del eje X a vertical
            plt.xticks(rotation=90) 
            
            # Estilo estético
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
    else:
        with col2:
            st.info("Dibuja algo para ver la predicción.")

st.markdown("---")
st.caption("Desarrollado con TensorFlow, ONNX y Streamlit.")
