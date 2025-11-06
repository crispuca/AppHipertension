import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import random

from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="Predicción de Hipertensión – ENS 2025", page_icon="🩺", layout="wide")

PALETTE = ["#5B8E7D","#D96C6C","#6C78D9","#E6B655"]

# ------------------ CARGA DEL MODELO ------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load("models/best_model_elasticnet.joblib")

modelo = cargar_modelo()

st.title("🩺 Predicción de Hipertensión – Modelo Final (ElasticNet Logistic Regression)")
st.caption("Aplicación interactiva basada en el modelo entrenado durante el análisis ENS 2025.")

# ------------------ MODO EXPLORACIÓN ------------------
st.header("📊 Exploración de datos y relaciones")
uploaded_file = st.file_uploader("Subí un archivo CSV con las variables del modelo para explorar predicciones", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    # Calcular probabilidades
    y_prob = modelo.predict_proba(df)[:, 1]
    df["Probabilidad_Hipertension"] = y_prob
    df["Prediccion"] = np.where(y_prob >= 0.5, "Hipertenso", "No Hipertenso")

    st.success("✅ Predicciones generadas con el modelo final.")

    # Mostrar resumen
    st.metric("Promedio de probabilidad predicha", f"{df['Probabilidad_Hipertension'].mean():.2%}")

    # Gráfico interactivo 1 - Edad vs Probabilidad
    if "edad" in df.columns and "actividad_fisica" in df.columns:
        chart_edad = (
            alt.Chart(df)
            .mark_circle(size=60, opacity=0.5)
            .encode(
                x="edad:Q",
                y="Probabilidad_Hipertension:Q",
                color=alt.Color("actividad_fisica:N", title="Actividad Física", scale=alt.Scale(range=["#D96C6C","#5B8E7D"])),
                tooltip=["edad","actividad_fisica","Probabilidad_Hipertension"]
            )
            .properties(title="Edad vs Probabilidad de Hipertensión", height=350)
            .interactive()
        )
        st.altair_chart(chart_edad, use_container_width=True)

    # Gráfico interactivo 2 - Nivel socioeconómico vs hábito de fumar
    if {"nivel_socioeconomico","habito_fumar"}.issubset(df.columns):
        chart_socio_fumar = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("nivel_socioeconomico:N", title="Nivel socioeconómico"),
                y=alt.Y("mean(Probabilidad_Hipertension):Q", title="Promedio de probabilidad"),
                color=alt.Color("habito_fumar:N", title="Hábito de fumar", scale=alt.Scale(domain=["Sí","No"], range=["#D96C6C","#5B8E7D"])),
                tooltip=["nivel_socioeconomico","habito_fumar","mean(Probabilidad_Hipertension)"]
            )
            .properties(title="Nivel socioeconómico y hábito de fumar", height=350)
        )
        st.altair_chart(chart_socio_fumar, use_container_width=True)

    # Gráfico interactivo 3 - Sueño y Diabetes
    if {"satifaccion_calidad_sueño","diabetes"}.issubset(df.columns):
        chart_sueño_diabetes = (
            alt.Chart(df)
            .mark_boxplot(size=30)
            .encode(
                x="satifaccion_calidad_sueño:N",
                y="Probabilidad_Hipertension:Q",
                color=alt.Color("diabetes:N", title="Diabetes", scale=alt.Scale(domain=["Sí","No"], range=["#D96C6C","#5B8E7D"])),
                tooltip=["satifaccion_calidad_sueño","diabetes","Probabilidad_Hipertension"]
            )
            .properties(title="Sueño y diabetes en el riesgo estimado", height=350)
        )
        st.altair_chart(chart_sueño_diabetes, use_container_width=True)

# ------------------ MODO PREDICCIÓN ------------------
st.header("🧮 Predicción individual")
st.markdown("Completá los campos para obtener una predicción:")

col1, col2, col3 = st.columns(3)

with col1:
    edad = st.number_input("Edad", 0, 120, 45)
    mayor_60 = int(edad >= 60)

    # 🔹 Calcular grupo de edad automáticamente según el valor de edad
    if edad < 30:
        grupo_edad = "<30"
    elif edad < 50:
        grupo_edad = "30-50"
    elif edad < 70:
        grupo_edad = "50-70"
    else:
        grupo_edad = ">70"

    # Mostrar el grupo asignado (solo informativo, no editable)
    st.info(f"📊 Grupo de edad asignado automáticamente: **{grupo_edad}**")

    sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
    nivel_educacion = st.selectbox(
        "Nivel educativo", ["Primaria", "Secundaria", "Terciaria/Universitaria", "Posgrado"]
    )

with col2:
    nivel_socioeconomico = st.selectbox("Nivel socioeconómico", ["Bajo", "Medio", "Alto"])
    actividad_fisica = st.selectbox("Actividad física", ["Sí", "No"])
    habito_fumar = st.selectbox("Hábito de fumar", ["Sí", "No"])
    consume_alcohol_bin = st.selectbox("Consumo de alcohol", ["Consume", "No consume"])
    sueño_simple = st.selectbox("Calidad de sueño", ["Satisfecho", "Insatisfecho"])

with col3:
    diabetes = st.selectbox("Diabetes", ["Sí", "No"])
    diabetes_mayor60 = int((diabetes == "Sí") and (mayor_60 == 1))
    ocupacion = st.selectbox(
        "Ocupación",
        ["Activo laboral", "Estudiante", "Rentista", "Desocupado", "Jubilado", "Otra Situación"]
    )
    satifaccion_calidad_sueño = st.selectbox(
        "Satisfacción con el sueño", ["Muy Bien", "Bien", "Regular", "Mal", "Muy Mal"]
    )
    nro_comidas_dia = st.slider("Número de comidas al día", 1, 7, 3)

if st.button("🔮 Predecir riesgo"):
    # 🔹 Crear el DataFrame con todas las variables requeridas
    input_df = pd.DataFrame([{
        "edad": edad,
        "mayor_60": mayor_60,
        "grupo_edad": grupo_edad,  # ← ahora calculado automáticamente
        "sexo": sexo,
        "nivel_educacion": nivel_educacion,
        "nivel_socioeconomico": nivel_socioeconomico,
        "actividad_fisica": actividad_fisica,
        "habito_fumar": habito_fumar,
        "consume_alcohol_bin": consume_alcohol_bin,
        "sueño_simple": sueño_simple,
        "diabetes": diabetes,
        "diabetes_mayor60": diabetes_mayor60,
        "ocupacion": ocupacion,
        "satifaccion_calidad_sueño": satifaccion_calidad_sueño,
        "nro_comidas_dia": nro_comidas_dia
    }])

    # 🔹 Predicción con el modelo
    prob = modelo.predict_proba(input_df)[0, 1]
    pred = "Hipertenso" if prob >= 0.5 else "No hipertenso"

    # 🔹 Mostrar resultados
    st.metric("Probabilidad estimada", f"{prob:.2%}")
    st.success(f"Predicción: {pred}")
    
        # ---------------- RECOMENDACIONES Y VISUALIZACIONES ----------------
    if prob > 0.4:
        st.warning("⚠️ Tu probabilidad estimada supera el 40%. A continuación se muestran recomendaciones preventivas:")

        # --- Visualización Altair: impacto de hábitos saludables ---
        data_consejos = pd.DataFrame({
            "Hábito": ["Actividad física regular", "No fumar", "Buena calidad de sueño", "Dieta equilibrada", "Chequeos médicos anuales"],
            "Reducción de riesgo (%)": [30, 25, 15, 20, 10]
        })

        chart_consejos = (
            alt.Chart(data_consejos)
            .mark_bar()
            .encode(
                x=alt.X("Reducción de riesgo (%):Q", title="Reducción estimada del riesgo de hipertensión"),
                y=alt.Y("Hábito:N", sort='-x'),
                color=alt.Color("Hábito:N", scale=alt.Scale(scheme="tealblues")),
                tooltip=["Hábito", "Reducción de riesgo (%)"]
            )
            .properties(title="Hábitos saludables que ayudan a reducir el riesgo", height=300)
        )
        st.altair_chart(chart_consejos, use_container_width=True)

        # --- Recursos externos útiles ---
        st.markdown("### 🌐 Recursos recomendados para cuidar tu salud cardiovascular:")
        st.markdown(
            """
            - 🩺 Consejos para la hipertensión :  
              [https://www.caeme.org.ar/hipertension-10-consejos-para-cuidar-la-presion-arterial/](https://www.caeme.org.ar/hipertension-10-consejos-para-cuidar-la-presion-arterial/)
            - ❤️ Fundación Cardiológica Argentina – Guías sobre prevención:  
              [https://www.fundacioncardiologica.org/](https://www.fundacioncardiologica.org/)
            - 🌎 Organización Mundial de la Salud – Información general sobre hipertensión:  
              [https://www.who.int/es/news-room/fact-sheets/detail/hypertension](https://www.who.int/es/news-room/fact-sheets/detail/hypertension)
            - 🥗 Sociedad Argentina de Hipertensión Arterial – Recomendaciones alimentarias:  
              [https://saha.org.ar/](https://saha.org.ar/)
            """
        )


        # ------------------ MENSAJES DINÁMICOS ------------------
        mensajes_saludables = [
            "💪 Cada pequeño cambio cuenta. Caminar 30 minutos al día puede marcar la diferencia.",
            "🍎 Recordá mantener una dieta balanceada: frutas, verduras y menos sal.",
            "🧘‍♀️ Dormir bien es tan importante como hacer ejercicio. ¡Dale prioridad al descanso!",
            "🚶‍♂️ Moverte más no siempre significa ir al gimnasio: subí escaleras o salí a pasear.",
            "❤️ Cuidar tu corazón es una inversión en tu futuro. ¡Empezá hoy!",
            "🥗 Reducí el consumo de sodio y bebidas azucaradas. Tu presión te lo va a agradecer.",
            "🩺 Medite tu presión arterial al menos una vez al año, aunque te sientas bien.",
            "💧 Tomá suficiente agua y evitá el exceso de café o alcohol.",
            "😌 Controlar el estrés también protege tu salud cardiovascular.",
            "👟 La constancia vale más que la intensidad: moverte un poco todos los días ya es ganar."
        ]

        mensajes_informativos = [
            "📊 Más del 30% de los adultos tiene hipertensión sin saberlo. ¡Chequeate regularmente!",
            "🧠 La hipertensión no siempre da síntomas, pero puede afectar corazón, riñones y cerebro.",
            "❤️ Una presión arterial ideal suele estar por debajo de 120/80 mmHg.",
            "🧍‍♀️ Mantener un peso saludable ayuda a reducir la presión arterial naturalmente.",
            "🩸 El exceso de sal es uno de los principales factores de riesgo de hipertensión."
        ]

        mensajes_motivacionales = [
            "🌟 Cada paso cuenta hacia una vida más sana.",
            "🔥 Tu salud está en tus manos: cuidate con pequeños hábitos diarios.",
            "💖 Nunca es tarde para empezar a mejorar tu bienestar.",
            "⚡ Las decisiones saludables de hoy son tu energía de mañana.",
            "🌈 Cuidarte no es un lujo, es una forma de quererte."
        ]

        # Selecciona uno aleatorio de cada categoría
        mensaje_diario = random.choice(mensajes_saludables + mensajes_informativos + mensajes_motivacionales)

        # Mostrarlo al inicio de la app
        st.info(mensaje_diario)
