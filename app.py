import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import random

st.set_page_config(page_title="Predicción de Hipertensión", page_icon="🩺", layout="wide")

PALETTE = ["#5B8E7D", "#D96C6C", "#6C78D9", "#E6B655"]

# ------------------ CARGA DEL MODELO ------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load("models/best_model_elasticnet.joblib")

modelo = cargar_modelo()

st.title("Predicción de Hipertensión")
st.caption("Aplicación interactiva basada en el modelo entrenado durante el análisis del portal de datos abiertos de Chile")

# ------------------ PESTAÑAS ------------------
tab_prediccion, tab_visualizacion, tab_recomendaciones = st.tabs([
    "Predicción individual",
    "Visualización con dataset cargado",
    "Recomendaciones"
])

# PESTAÑA 1: PREDICCIÓN INDIVIDUAL
with tab_prediccion:
    st.header("Predicción individual")
    st.markdown("Completá los campos para obtener una predicción personalizada:")

    col1, col2, col3 = st.columns(3)

    #  Columna 1: Datos demográficos 
    with col1:
        edad = st.number_input("Edad", min_value=0, max_value=120, value=45)
        grupo_edad = (
            "<30" if edad < 30 else
            "30-50" if edad < 50 else
            "50-70" if edad < 70 else
            ">70"
        )
        st.info(f"📊 Grupo de edad asignado: **{grupo_edad}**")

        sexo = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)

        nivel_educacion = st.selectbox(
            "Nivel educativo",
            ["Primaria", "Secundaria", "Terciaria/Universitaria", "Posgrado"]
        )

    #  Columna 2: Condición socioeconómica y hábitos 
    with col2:
        nivel_socioeconomico = st.selectbox(
            "Nivel socioeconómico", ["Buena", "Regular", "Mala"]
        )

        actividad_fisica = st.radio("¿Realiza actividad física regularmente?", ["Sí", "No"], horizontal=True)
        # Usuario elige normalmente, pero se invierte antes de enviarlo al modelo
        habito_fumar_user = st.radio("¿Tiene hábito de fumar?", ["Sí", "No"], horizontal=True)
        habito_fumar = "No" if habito_fumar_user == "Sí" else "Sí"
        consume_alcohol_bin = st.radio("¿Consume alcohol?", ["Consume", "No consume"], horizontal=True)

    #  Columna 3: Salud y estilo de vida 
    with col3:
        diabetes = st.radio("¿Tiene diagnóstico de diabetes?", ["Sí", "No"], horizontal=True)
        enfermedad_renal = st.radio("¿Tiene enfermedad renal?", ["Sí", "No"], horizontal=True)

        ocupacion_simplificada = st.selectbox(
            "Situación laboral",
            ["Activo laboral", "Estudiante", "Rentista", "Desocupado", "Jubilado"]
        )

        satifaccion_calidad_sueño = st.select_slider(
            "Satisfacción con el sueño",
            options=["Muy Insatisfecho", "Insatisfecho", "Regular", "Satisfecho"]
        )

        nro_comidas_dia = st.slider("Número de comidas al día", 1, 7, 3)

    # ---------- Botón para predecir ----------
    if st.button("Predecir riesgo"):
        input_df = pd.DataFrame([{
            "grupo_edad": grupo_edad,
            "sexo": sexo,
            "nivel_educacion": nivel_educacion,
            "nivel_socioeconomico": nivel_socioeconomico,
            "actividad_fisica": actividad_fisica,
            "habito_fumar": habito_fumar,
            "consume_alcohol_bin": consume_alcohol_bin,
            "diabetes": diabetes,
            "ocupacion_simplificada": ocupacion_simplificada,
            "satifaccion_calidad_sueño": satifaccion_calidad_sueño,
            "nro_comidas_dia": nro_comidas_dia,
            "enfermedad_renal": enfermedad_renal
        }])

        try:
            prob = modelo.predict_proba(input_df)[0, 1]
            pred = "Hipertenso" if prob >= 0.5 else "No hipertenso"

            st.metric("Probabilidad estimada", f"{prob:.2%}")
            st.success(f"Predicción: **{pred}**")

            # Guardamos en session_state
            st.session_state["ultima_prediccion"] = pred
            st.session_state["ultima_probabilidad"] = prob

        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")

# PESTAÑA 2: VISUALIZACIÓN DE DATASET
with tab_visualizacion:
    st.header("Visualización con dataset cargado automáticamente")

    # Cargar dataset directamente desde la carpeta "data"
    try:
        df = pd.read_csv("data/data_prueba.csv")
        st.success("✅ Dataset cargado correctamente desde el directorio")

        # Calcular predicciones si el dataset coincide
        try:
            y_prob = modelo.predict_proba(df)[:, 1]
            df["Probabilidad_Hipertension"] = y_prob
            df["Prediccion"] = np.where(y_prob >= 0.5, "Hipertenso", "No Hipertenso")

            st.success("Predicciones generadas correctamente ✅")

            st.metric("Promedio de probabilidad predicha", f"{df['Probabilidad_Hipertension'].mean():.2%}")


            if {"actividad_fisica", "habito_fumar", "diabetes", "enfermedad_renal", "Prediccion"}.issubset(df.columns):
                # Corregir hábito de fumar
                df["habito_fumar_corrigido"] = df["habito_fumar"].replace({"Sí": "No", "No": "Sí"})

                # Reorganizar datos
                df_riesgos = df.melt(
                    id_vars=["Prediccion"],
                    value_vars=["actividad_fisica", "habito_fumar_corrigido", "diabetes", "enfermedad_renal"],
                    var_name="Factor_de_Riesgo",
                    value_name="Estado"
                )

                # Calcular tasas
                df_tasas = (
                    df_riesgos.groupby(["Factor_de_Riesgo", "Estado"])
                    .agg(
                        tasa_hipertension=("Prediccion", lambda x: (x == "Hipertenso").mean()),
                        total=("Prediccion", "count")
                    )
                    .reset_index()
                )

                # Nombres legibles
                nombres_factores = {
                    "actividad_fisica": "Actividad física",
                    "habito_fumar_corrigido": "Hábito de fumar",
                    "diabetes": "Diabetes",
                    "enfermedad_renal": "Enfermedad renal"
                }
                df_tasas["Factor_de_Riesgo"] = df_tasas["Factor_de_Riesgo"].replace(nombres_factores)

                # --- Gráfico base ---
                base = (
                    alt.Chart(df_tasas, width=200, height=350)  # tamaño definido aquí
                    .encode(
                        x=alt.X("Estado:N", title="Presencia del factor", axis=alt.Axis(labelFontSize=12)),
                        y=alt.Y("tasa_hipertension:Q", title="Tasa de hipertensión", axis=alt.Axis(format=".0%")),
                        color=alt.Color(
                            "Estado:N",
                            title="Presencia del factor",
                            scale=alt.Scale(domain=["Sí", "No"], range=["#D96C6C", "#5B8E7D"])
                        ),
                        tooltip=[
                            alt.Tooltip("Factor_de_Riesgo:N", title="Factor de riesgo"),
                            alt.Tooltip("Estado:N", title="Presencia"),
                            alt.Tooltip("tasa_hipertension:Q", title="Tasa de hipertensión", format=".1%"),
                            alt.Tooltip("total:Q", title="Cantidad de personas")
                        ]
                    )
                )

                # --- Capas ---
                barras = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                texto = base.mark_text(
                    align="center", baseline="bottom", dy=-5,
                    fontSize=13, fontWeight="bold", color="white"
                ).encode(text=alt.Text("tasa_hipertension:Q", format=".0%"))

                # --- Facet + configuración ---
                chart_factores = (
                    alt.layer(barras, texto)
                    .facet(
                        column=alt.Column(
                            "Factor_de_Riesgo:N",
                            header=alt.Header(
                                labelAngle=0,
                                labelAlign="center",
                                labelFontSize=15,
                                labelColor="#E6E6E6",
                                labelPadding=15
                            ),
                            title=None
                        )
                    )
                    .configure_facet(spacing=30)
                    .configure_view(stroke=None)
                    .configure_axis(
                        titleFontSize=13,
                        labelFontSize=12
                    )
                    .properties(
                        title="Comparación de Factores de Riesgo y su Influencia en la Hipertensión"
                    )
                    .resolve_scale(y="shared")
                )

                # Mostrar gráfico en tamaño completo sin errores
                st.altair_chart(chart_factores, use_container_width=True)
   
            # DISTRIBUCIÓN GENERAL DE HIPERTENSIÓN
            if "Prediccion" in df.columns:
                st.subheader("🩺 Distribución General de Hipertensión en el Dataset")

                distribucion = (
                    df["Prediccion"].value_counts(normalize=True)
                    .rename_axis("Condición")
                    .reset_index(name="Proporción")
                )

                chart_pie = (
                    alt.Chart(distribucion)
                    .mark_arc(innerRadius=60)
                    .encode(
                        theta=alt.Theta("Proporción:Q", stack=True),
                        color=alt.Color(
                            "Condición:N",
                            scale=alt.Scale(domain=["Hipertenso", "No hipertenso"], range=["#D96C6C", "#5B8E7D"]),
                            title="Condición"
                        ),
                        tooltip=[
                            alt.Tooltip("Condición:N", title="Condición"),
                            alt.Tooltip("Proporción:Q", title="Porcentaje", format=".1%")
                        ]
                    )
                    .properties(
                        title="Distribución de Personas con y sin Hipertensión",
                        height=400, width=400
                    )
                )

                st.altair_chart(chart_pie, use_container_width=False)

                st.caption("🧠 Este gráfico muestra el porcentaje de casos hipertensos vs no hipertensos en el conjunto analizado.")


            #  Gráfico 1: Edad vs Sexo 
            if {"grupo_edad", "sexo", "Prediccion"}.issubset(df.columns):
                df_edad_sexo = (
                    df.groupby(["grupo_edad", "sexo"])
                    .agg(
                        tasa_hipertension=("Prediccion", lambda x: (x == "Hipertenso").mean()),
                        total=("Prediccion", "count")
                    )
                    .reset_index()
                )

                chart_edad_sexo = (
                    alt.Chart(df_edad_sexo)
                    .mark_circle(filled=True, opacity=0.75)
                    .encode(
                        x=alt.X("grupo_edad:N", title="Grupo de Edad", sort=["<30", "30-50", "50-70", ">70"]),
                        y=alt.Y("tasa_hipertension:Q", title="Tasa de Hipertensión", axis=alt.Axis(format=".0%")),
                        size=alt.Size("total:Q", title="Cantidad de Personas", scale=alt.Scale(range=[400, 2000])),
                        color=alt.Color("sexo:N", title="Sexo", scale=alt.Scale(range=["#1411C3", "#E6B655"])),
                        tooltip=[
                            alt.Tooltip("sexo:N", title="Sexo"),
                            alt.Tooltip("grupo_edad:N", title="Grupo de edad"),
                            alt.Tooltip("tasa_hipertension:Q", title="Tasa de hipertensión", format=".1%"),
                            alt.Tooltip("total:Q", title="Cantidad de personas")
                        ]
                    )
                    .properties(
                        title="Tasa de Hipertensión por Grupo de Edad y Sexo",
                        width=600, height=400
                    )
                )

                st.altair_chart(chart_edad_sexo, use_container_width=True)

            #  Gráfico 2: Hábito de fumar vs Actividad física
            if {"habito_fumar", "actividad_fisica", "Prediccion"}.issubset(df.columns):
                df_habitos = (
                    df.groupby(["habito_fumar", "actividad_fisica"])
                    .agg(tasa=("Prediccion", lambda x: (x == "Hipertenso").mean()))
                    .reset_index()
                )

                chart_habitos = (
                    alt.Chart(df_habitos)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                    .encode(
                        x=alt.X("habito_fumar:N", title="Hábito de fumar", sort=["No", "Sí"]),
                        y=alt.Y("tasa:Q", title="Tasa de Hipertensión", axis=alt.Axis(format=".0%")),
                        color=alt.Color("actividad_fisica:N", title="Actividad Física",
                                        scale=alt.Scale(range=["#D96C6C", "#5B8E7D"])),
                        tooltip=[
                            alt.Tooltip("habito_fumar:N", title="Hábito de fumar"),
                            alt.Tooltip("actividad_fisica:N", title="Actividad Física"),
                            alt.Tooltip("tasa:Q", title="Tasa de hipertensión", format=".1%")
                        ]
                    )
                    .properties(title="Relación entre Fumar, Actividad Física y Riesgo de Hipertensión", height=400)
                )

                st.altair_chart(chart_habitos, use_container_width=True)

            if {"enfermedad_renal", "Prediccion"}.issubset(df.columns):
                df_renal = (
                    df.groupby("enfermedad_renal")
                    .agg(
                        tasa_hipertension=("Prediccion", lambda x: (x == "Hipertenso").mean()),
                        total=("Prediccion", "count")
                    )
                    .reset_index()
                )

                chart_renal = (
                    alt.Chart(df_renal)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                    .encode(
                        x=alt.X("enfermedad_renal:N", title="Enfermedad renal", sort=["No", "Sí"]),
                        y=alt.Y("tasa_hipertension:Q", title="Tasa de hipertensión", axis=alt.Axis(format=".0%")),
                        color=alt.Color(
                            "enfermedad_renal:N",
                            title="Enfermedad renal",
                            scale=alt.Scale(domain=["No", "Sí"], range=["#5B8E7D", "#D96C6C"])
                        ),
                        tooltip=[
                            alt.Tooltip("enfermedad_renal:N", title="Enfermedad renal"),
                            alt.Tooltip("tasa_hipertension:Q", title="Tasa de hipertensión", format=".1%"),
                            alt.Tooltip("total:Q", title="Cantidad de personas")
                        ]
                    )
                    .properties(
                        title="Relación entre Enfermedad Renal y Tasa de Hipertensión",
                        width=500,
                        height=400
                    )
                )

                # Agregar etiquetas de porcentaje arriba de las barras
                text = chart_renal.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5,
                    fontSize=13,
                    fontWeight='bold',
                    color='black'
                ).encode(
                    text=alt.Text("tasa_hipertension:Q", format=".0%")
                )

                st.altair_chart(chart_renal + text, use_container_width=True)
            
        except Exception as e:
            st.error(f"⚠️ Error al aplicar el modelo al dataset: {e}")
    except Exception as e:
        st.error(f"⚠️ No se pudo cargar el dataset: {e}")
        st.stop()

# PESTAÑA 3: RECOMENDACIONES
with tab_recomendaciones:
    st.header("💡 Recomendaciones")

    pred_guardada = st.session_state.get("ultima_prediccion", None)
    prob_guardada = st.session_state.get("ultima_probabilidad", None)

    if pred_guardada == "Hipertenso":
        st.warning(f"⚠️ Tu probabilidad estimada fue de **{prob_guardada:.2%}**. A continuación se muestran hábitos preventivos:")

        data_consejos = pd.DataFrame({
            "Hábito": [
                "Actividad física regular",
                "No fumar",
                "Buena calidad de sueño",
                "Dieta equilibrada",
                "Chequeos médicos anuales"
            ],
            "Reducción de riesgo (%)": [30, 25, 15, 20, 10]
        })

        chart_consejos = (
            alt.Chart(data_consejos)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Hábito:N", sort='-y', title="Hábito saludable", axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                y=alt.Y("Reducción de riesgo (%):Q", title="Reducción estimada del riesgo"),
                color=alt.value("#561DBF"),
                tooltip=["Hábito", "Reducción de riesgo (%)"]
            )
            .properties(title="Hábitos saludables que ayudan a reducir el riesgo", height=400)
        )

        text = chart_consejos.mark_text(
            align='center', baseline='bottom', dy=-5, fontSize=13, fontWeight='bold', color='white'
        ).encode(
            text=alt.Text("Reducción de riesgo (%):Q", format=".0f")
        )

        st.altair_chart(chart_consejos + text, use_container_width=True)

        st.markdown("### Recursos sobre la hipertensión")

        st.markdown("""
            <style>
            .link-card {
                background-color: #1e1e1e;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            }
            .link-card a {
                text-decoration: none;
                color: #4da6ff;
                font-size: 18px;
                font-weight: bold;
            }
            .link-card p {
                color: #ccc;
                font-size: 14px;
            }
            </style>

            <div class="link-card">
                <a href="https://www.caeme.org.ar/hipertension-10-consejos-para-cuidar-la-presion-arterial/" target="_blank">🩺 Consejos para la hipertensión – CAEME</a>
                <p>10 consejos prácticos para cuidar tu presión arterial según CAEME.</p>
            </div>

            <div class="link-card">
                <a href="https://www.fundacioncardiologica.org/" target="_blank">❤️ Fundación Cardiológica Argentina</a>
                <p>Información confiable sobre prevención y tratamiento de enfermedades cardíacas.</p>
            </div>

            <div class="link-card">
                <a href="https://www.who.int/es/news-room/fact-sheets/detail/hypertension" target="_blank">🌍 OMS – Información sobre hipertensión</a>
                <p>Datos globales y recomendaciones oficiales de la Organización Mundial de la Salud.</p>
            </div>

            <div class="link-card">
                <a href="https://saha.org.ar/" target="_blank">🌿 Sociedad Argentina de Hipertensión Arterial</a>
                <p>Asociación científica argentina especializada en la investigación de la hipertensión.</p>
            </div>
            """, unsafe_allow_html=True)

        st.info(random.choice([
            "💪 Caminar 30 minutos al día puede reducir la presión arterial significativamente.",
            "🍎 Evitá comidas ultraprocesadas y reducí el consumo de sal.",
            "🧘 Dormir bien (6–8h) es esencial para mantener la presión controlada.",
            "🚭 Si fumás, dejarlo puede reducir tu riesgo en un 25% en pocos meses.",
            "💉 Controlá tu presión regularmente aunque te sientas bien."
        ]))

    else:
        st.info("Las recomendaciones personalizadas se habilitarán cuando el resultado sea **Hipertenso**.")
