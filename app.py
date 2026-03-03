import sys
import subprocess
import streamlit as st

# --- BLOQUE DE SEGURIDAD PARA DEPENDENCIAS (Versión Ultra-Segura) ---
# Forzamos la instalación ANTES de importar cualquier librería conflictiva
def install_dependencies():
    try:
        import streamlit_drawable_canvas
        import onnxruntime
        import matplotlib
        import PIL
    except ImportError:
        # Si falta alguna, instalamos todas de golpe
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "streamlit-drawable-canvas==0.9.3", 
            "onnxruntime==1.17.1", 
            "numpy<2.0.0", 
            "Pillow", 
            "matplotlib"
        ])
        st.rerun()

install_dependencies()

# --- AHORA SÍ, IMPORTAMOS TODO ---
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="IA Reconocedora de Números", layout="centered")

st.title("🔢 Reconocedor de Dígitos (CNN + ONNX)")
st.markdown("""
Dibuja un número **grande y centrado** en el recuadro negro. 
""")

# 1. Cargar el modelo ONNX
@st.cache_resource
def load_model():
    # El archivo debe estar en la raíz de tu GitHub
    return ort.InferenceSession("modelo_digitos.onnx")

try:
    session = load_model()
except Exception as e:
    st.error(f"Archivo 'modelo_digitos.onnx' no encontrado. Súbelo a GitHub.")
    st.stop()

# 2. Interfaz de usuario (Columnas)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Lienzo")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=22,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Limpiar"):
        st.rerun()

# 3. Lógica de Predicción
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    # Solo predecir si hay algo dibujado (píxeles blancos)
    if np.any(np.array(img) > 50):
        img_28 = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_28).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img_array})[0][0]
        
        prediccion = np.argmax(result)
        confianza = result[prediccion] * 100

        with col2:
            st.subheader("Resultado")
            st.metric(label="Número", value=str(prediccion), delta=f"{confianza:.1f}%")
            
            # Gráfico de barras con etiquetas verticales
            fig, ax = plt.subplots(figsize=(5, 4))
            etiquetas = [str(i) for i in range(10)]
            barras = ax.bar(etiquetas, result, color='skyblue')
            barras[prediccion].set_color('orange') # Resaltar el ganador
            
            ax.set_ylim(0, 1)
            plt.xticks(rotation=90) # Etiquetas verticales
            
            # Limpiar bordes del gráfico
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
    else:
        with col2:
            st.info("Dibuja para ver el análisis.")

st.markdown("---")
st.caption("Arquitectura CNN optimizada | Formato ONNX")
