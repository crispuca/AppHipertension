import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import random

st.set_page_config(page_title="Predicción de Hipertensión ", page_icon="🩺", layout="wide")

PALETTE = ["#5B8E7D","#D96C6C","#6C78D9","#E6B655"]

# ------------------ CARGA DEL MODELO ------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load("models/best_model_elasticnet.joblib")

modelo = cargar_modelo()

st.title("Predicción de Hipertensión")
st.caption("Aplicación interactiva basada en el modelo entrenado durante el análisis del portal de datos abiertos de Chile")

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

    st.success("Predicciones generadas con el modelo final.")

    # Mostrar resumen
    st.metric("Promedio de probabilidad predicha", f"{df['Probabilidad_Hipertension'].mean():.2%}")

    # --- Gráfico 1: Edad vs Probabilidad ---
    if {"edad", "actividad_fisica"}.issubset(df.columns):
        chart_edad = (
            alt.Chart(df)
            .mark_circle(size=60, opacity=0.6)
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

    # --- Gráfico 2: Enfermedades crónicas ---
    if {"enfermedad_renal","enfermedad_cardiaca","danio_higado"}.issubset(df.columns):
        df_long = df.melt(
            id_vars=["Probabilidad_Hipertension"],
            value_vars=["enfermedad_renal","enfermedad_cardiaca","danio_higado"],
            var_name="Condición",
            value_name="Presencia"
        )
        chart_enfermedades = (
            alt.Chart(df_long)
            .mark_boxplot(size=35)
            .encode(
                x=alt.X("Condición:N", title="Condición médica"),
                y=alt.Y("Probabilidad_Hipertension:Q", title="Probabilidad de Hipertensión"),
                color=alt.Color("Presencia:N", scale=alt.Scale(domain=["Sí","No"], range=["#D96C6C","#5B8E7D"])),
                tooltip=["Condición","Presencia","Probabilidad_Hipertension"]
            )
            .properties(title="Influencia de condiciones médicas en el riesgo estimado", height=350)
        )
        st.altair_chart(chart_enfermedades, use_container_width=True)

    # --- Gráfico 3: Nivel socioeconómico y fumar ---
    if {"nivel_socioeconomico","habito_fumar"}.issubset(df.columns):
        chart_socio = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x="nivel_socioeconomico:N",
                y="mean(Probabilidad_Hipertension):Q",
                color=alt.Color("habito_fumar:N", title="Hábito de fumar", scale=alt.Scale(domain=["Sí","No"], range=["#D96C6C","#5B8E7D"])),
                tooltip=["nivel_socioeconomico","habito_fumar","mean(Probabilidad_Hipertension)"]
            )
            .properties(title="Nivel socioeconómico y hábito de fumar", height=350)
        )
        st.altair_chart(chart_socio, use_container_width=True)


# ------------------ MODO PREDICCIÓN ------------------
st.header("Predicción individual")
st.markdown("Completá los campos para obtener una predicción:")

col1, col2, col3 = st.columns(3)

with col1:
    edad = st.number_input("Edad", 0, 120, 45)
    mayor_60 = int(edad >= 60)
    grupo_edad = "<30" if edad < 30 else "30-50" if edad < 50 else "50-70" if edad < 70 else ">70"
    st.info(f"📊 Grupo de edad asignado automáticamente: **{grupo_edad}**")

    sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
    nivel_educacion = st.selectbox("Nivel educativo", ["Primaria", "Secundaria", "Terciaria/Universitaria", "Posgrado"])

with col2:
    nivel_socioeconomico = st.selectbox("Nivel socioeconómico", ["Buena", "Regular", "Mala"])
    actividad_fisica = st.selectbox("Actividad física", ["Sí", "No"])
    habito_fumar = st.selectbox("Hábito de fumar", ["Sí", "No"])
    consume_alcohol_bin = st.selectbox("Consumo de alcohol", ["Consume", "No consume"])

with col3:
    diabetes = st.selectbox("Diabetes", ["Sí", "No"])
    diabetes_mayor60 = int((diabetes == "Sí") and (mayor_60 == 1))
    ocupacion_simplificada = st.selectbox("Ocupación", ["Activo laboral", "Estudiante", "Rentista", "Desocupado", "Jubilado", "Otra Situación"])
    satifaccion_calidad_sueño = st.selectbox("Satisfacción con el sueño", ["Satisfecho", "Regular", "Insatisfecho", "Muy Insatisfecho"])
    nro_comidas_dia = st.slider("Número de comidas al día", 1, 7, 3)
    enfermedad_renal = st.selectbox("Enfermedad renal", ["Sí", "No"])
    enfermedad_cardiaca = st.selectbox("Enfermedad cardíaca", ["Sí", "No"])
    danio_higado = st.selectbox("Daño hepático", ["Sí", "No"])

if st.button("🔮 Predecir riesgo"):
    # --- Crear DataFrame con todas las variables ---
    input_df = pd.DataFrame([{
        "edad": edad, "mayor_60": mayor_60, "grupo_edad": grupo_edad,
        "sexo": sexo, "nivel_educacion": nivel_educacion,
        "nivel_socioeconomico": nivel_socioeconomico,
        "actividad_fisica": actividad_fisica, "habito_fumar": habito_fumar,
        "consume_alcohol_bin": consume_alcohol_bin, 
        "diabetes": diabetes, "diabetes_mayor60": diabetes_mayor60,
        "ocupacion_simplificada": ocupacion_simplificada, "satifaccion_calidad_sueño": satifaccion_calidad_sueño,
        "nro_comidas_dia": nro_comidas_dia, "enfermedad_renal": enfermedad_renal,
        "enfermedad_cardiaca": enfermedad_cardiaca, "danio_higado": danio_higado
    }])

    prob = modelo.predict_proba(input_df)[0, 1]
    pred = "Hipertenso" if prob >= 0.5 else "No hipertenso"

    st.metric("Probabilidad estimada", f"{prob:.2%}")
    st.success(f"Predicción: {pred}")

    # ---------------- RECOMENDACIONES ----------------
    if prob >= 0.5:
        st.warning("⚠️ Tu probabilidad estimada supera el 50%. A continuación se muestran recomendaciones preventivas:")

        data_consejos = pd.DataFrame({
            "Hábito": ["Actividad física regular", "No fumar", "Buena calidad de sueño", "Dieta equilibrada", "Chequeos médicos anuales"],
            "Reducción de riesgo (%)": [30, 25, 15, 20, 10]
        })

        chart_consejos = (
            alt.Chart(data_consejos)
            .mark_bar()
            .encode(
                x=alt.X("Reducción de riesgo (%):Q", title="Reducción estimada del riesgo"),
                y=alt.Y("Hábito:N", sort='-x'),
                color=alt.Color("Hábito:N", scale=alt.Scale(scheme="tealblues")),
                tooltip=["Hábito","Reducción de riesgo (%)"]
            )
            .properties(title="Hábitos saludables que ayudan a reducir el riesgo", height=300)
        )
        st.altair_chart(chart_consejos, use_container_width=True)

        st.markdown("### 🌐 Recursos recomendados:")
        st.markdown("""
        - 🩺 [Consejos para la hipertensión – CAEME](https://www.caeme.org.ar/hipertension-10-consejos-para-cuidar-la-presion-arterial/)
        - ❤️ [Fundación Cardiológica Argentina](https://www.fundacioncardiologica.org/)
        - 🌎 [OMS – Información sobre hipertensión](https://www.who.int/es/news-room/fact-sheets/detail/hypertension)
        - 🥗 [Sociedad Argentina de Hipertensión Arterial](https://saha.org.ar/)
        """)

        mensajes = [
            "💪 Caminar 30 minutos al día puede reducir la presión arterial significativamente.",
            "🍎 Evitá comidas ultraprocesadas y reducí el consumo de sal.",
            "🧘 Dormir bien (6–8h) es esencial para mantener la presión controlada.",
            "🚭 Si fumás, dejarlo puede reducir tu riesgo en un 25% en pocos meses.",
            "💉 Controlá tu presión regularmente aunque te sientas bien.",
        ]
        st.info(random.choice(mensajes))
