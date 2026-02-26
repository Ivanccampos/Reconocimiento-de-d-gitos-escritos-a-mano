import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image

st.set_page_config(page_title="Reconocedor de Dígitos CNN", layout="centered")

st.title("🔢 Reconocedor de Dígitos con CNN")
st.write("Dibuja un número del 0 al 9 en el cuadro de abajo.")

# 1. Cargar el modelo ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession("modelo_digitos.onnx")

session = load_model()

# 2. Configuración del Lienzo (Canvas)
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

# 3. Procesamiento y Predicción
if canvas_result.image_data is not None:
    # Obtener la imagen del canvas
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    if st.button("Predecir"):
        # Preprocesar para el modelo (28x28)
        img_28 = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_28).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Inferencia con ONNX
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img_array})[0]
        
        prediccion = np.argmax(result)
        confianza = np.max(result) * 100

        # Mostrar resultados
        st.subheader(f"Resultado: {prediccion}")
        st.progress(int(confianza))
        st.write(f"Confianza: {confianza:.2f}%")
        
        # Mostrar gráfica de barras
        st.bar_chart(result[0])