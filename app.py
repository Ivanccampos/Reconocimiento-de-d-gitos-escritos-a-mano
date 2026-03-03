import streamlit as st
from streamlit_drawable_canvas import st_canvas
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt

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
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    if st.button("Predecir"):
        # Preprocesar para el modelo (28x28)
        img_28 = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_28).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Inferencia con ONNX
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img_array})[0][0]
        
        prediccion = np.argmax(result)
        confianza = np.max(result) * 100

        # Mostrar resultados principales
        st.subheader(f"Resultado: {prediccion}")
        st.progress(int(confianza))
        st.write(f"Confianza: {confianza:.2f}%")
        
        # --- NUEVA GRÁFICA CON EJE X VERTICAL ---
        st.write("Probabilidades por dígito:")
        
        # Preparamos los datos para la gráfica
        chart_data = pd.DataFrame({
            'Dígito': [str(i) for i in range(10)],
            'Confianza': result
        })

        # Creamos la gráfica con Altair (etiquetas a 90 grados)
        grafica = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Dígito', axis=alt.Axis(labelAngle=0)), # Cambia a 90 si prefieres vertical total
            y='Confianza',
            color=alt.condition(
                alt.datum.Dígito == str(prediccion),
                alt.value('orange'),     # Color para el ganador
                alt.value('steelblue')   # Color para los demás
            )
        ).properties(height=300)

        st.altair_chart(grafica, use_container_width=True)
