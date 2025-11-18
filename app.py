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
tab_prediccion, tab_visualizacion, tab_recomendaciones, tab_informeProyecto= st.tabs([
    "Predicción individual",
    "Visualización con dataset cargado",
    "Recomendaciones",
    "informe del Proyecto"
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

            st.metric("⚠️ Tu probabilidad estimada fue de", f"{prob:.2%}")
            st.success(f"Predicción: **{pred}**")

            # Guardamos en session_state
            st.session_state["ultima_prediccion"] = pred
            st.session_state["ultima_probabilidad"] = prob

        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")
    
        if pred == "Hipertenso":
            st.warning("A continuación se muestran hábitos preventivos para posible hipertension")

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
            st.info("Ya que no es probable que seas hipertenso, igual te dejamos consejos a tener en cuenta")
            st.info(
                "Revisiones de rutina, manten una dieta baja en sodio, evita los alimentos ultraprocesados y regula tu estres y sueño"
            )



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
                        order=alt.Order("Proporción:Q", sort="descending"),
                        color=alt.Color(
                            "Condición:N",
                            scale=alt.Scale(domain=["Hipertenso", "No Hipertenso"], range=["#D96C6C", "#5B8E7D"]),
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

            #Grafico de factores de riesgo, habito de fumar, enfermedad renal y diabetes
            if {"habito_fumar", "diabetes", "enfermedad_renal", "Prediccion"}.issubset(df.columns):
                # Corregir hábito de fumar
                df["habito_fumar_corrigido"] = df["habito_fumar"].replace({"Sí": "No", "No": "Sí"})

                # Reorganizar datos
                df_riesgos = df.melt(
                    id_vars=["Prediccion"],
                    value_vars=["habito_fumar_corrigido", "diabetes", "enfermedad_renal"],
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
                        title=alt.TitleParams(
                            text="Comparación de Factores de Riesgo y su Influencia en la Hipertensión",
                            # Este es el campo correcto para el subtítulo
                            subtitle="En este gráfico mostraremos cómo la actividad física, la diabetes, la enfermedad renal y el hábito de fumar influyen en la población a la hora de predecir si es probable que sea hipertenso o no"
                        )                    
                    )
                    .resolve_scale(y="shared")
                )

                # Mostrar gráfico en tamaño completo sin errores
                st.altair_chart(chart_factores, use_container_width=True)


            #Grafico de la influencia de la actividad fisica en la tasa de hipertension
            if {"actividad_fisica", "Prediccion"}.issubset(df.columns):
                df_renal = (
                    df.groupby("actividad_fisica")
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
                        x=alt.X("actividad_fisica:N", title="actividad_fisica", sort=["Sí", "No"]),
                        y=alt.Y("tasa_hipertension:Q", title="Tasa de hipertensión", axis=alt.Axis(format=".0%")),
                        color=alt.Color(
                            "actividad_fisica:N",
                            title="Actividad Fisica",
                            scale=alt.Scale(domain=["Sí", "No"], range=["#5B8E7D","#D96C6C"])
                        ),
                        tooltip=[
                            alt.Tooltip("actividad_fisica:N", title="Actividad Fisica"),
                            alt.Tooltip("tasa_hipertension:Q", title="Tasa de hipertensión", format=".1%"),
                            alt.Tooltip("total:Q", title="Cantidad de personas")
                        ]
                    )
                    .properties(
                        title="Relación entre Actividad Fisica y Tasa de Hipertensión",
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
            

            # Gráfico de Edad vs Sexo, grafico de burbujas 
            if {"grupo_edad", "sexo", "Prediccion"}.issubset(df.columns):
                df_edad_sexo = (
                    df.groupby(["grupo_edad", "sexo"])
                    .agg(
                        tasa_hipertension=("Prediccion", lambda x: (x == "Hipertenso").mean()),
                        total=("Prediccion", "count")
                    )
                    .reset_index()
                )

                TURQUESA_OSCURO = "#005757" #Turquesa oscuro

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
                    .configure_legend(
                    # Controla el color del símbolo (los círculos en la leyenda de Tamaño)
                        symbolFillColor=TURQUESA_OSCURO
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
                        # X: Divide la gráfica en los dos grupos principales
                        x=alt.X("actividad_fisica:N", title="Actividad Física", axis=None), 
                        # Y: La altura de la barra es la tasa de hipertensión
                        y=alt.Y("tasa:Q", title="Tasa de Hipertensión", axis=alt.Axis(format=".0%")), 
                        # Color: Usa el hábito de fumar para distinguir el color 
                        color=alt.Color("actividad_fisica:N", title="Actividad Física", scale=alt.Scale(range=["#D96C6C", "#5B8E7D"])), 
                        # Column: Crea dos paneles separados por el hábito de fumar
                        column=alt.Column(
                            "habito_fumar:N", 
                            title="Hábito de Fumar",
                            # 💡 ALINEACIÓN Y ORIENTACIÓN DEFINIDA DIRECTAMENTE EN EL ENCABEZADO
                             header=alt.Header(titleOrient="bottom", titleAlign="center")
                        ),
                    )
                    .properties(title="Tasa de Hipertensión por Hábito de Fumar y Actividad Física", height=400)
                )

                st.altair_chart(chart_habitos, use_container_width=True)


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


# PESTAÑA 4: Informe de Proyecto
with tab_informeProyecto:
    st.header("💡 Informe del Proyecto")
    
    st.subheader("Nombre del Proyecto:")
    st.title("Modelo de Predicción de Hipertensión en Personas según Hábitos y Nivel de Vida")
    
    st.markdown("""
    ---
    ## 🎯 Objetivos y Utilidad
    
    Este proyecto fue desarrollado con el objetivo principal de **proveer una herramienta de detección temprana** del riesgo de hipertensión arterial (HTA) utilizando datos de estilo de vida, demográficos y de salud (factores de riesgo).
    
    ### ¿Por qué es útil?
    
    * **Prevención Temprana:** Permite identificar a individuos con alto riesgo de HTA antes de que la enfermedad se manifieste o genere complicaciones severas.
    * **Personalización de Intervenciones:** Al conocer los factores específicos que elevan el riesgo, las autoridades sanitarias y los profesionales pueden dirigir campañas de prevención más efectivas y personalizadas.
    * **Optimización de Recursos:** Enfocar recursos de seguimiento y diagnóstico en la población de mayor riesgo.
    
    ---
    """)

    st.header("📊 Factores Fundamentales en la Predicción")
    st.subheader("Importancia de las Variables")

    st.markdown("""
    En esta seccion vamos a ver un ranking dentro de nuestro modelo de como influye cada variable para la prediccion de hipertension, este
    ranking no explica que las que esten abajo son las que descartamos, sino muestra la influencia de las mismas
    """)
    # Muestra la imagen de Importancia de Variables
    st.image(
        "data/assets/variables_influyentes.png", 
        caption="Importancia de las variables por Peso Absoluto (magnitud del efecto)",
        use_column_width=True
    )
    
    st.markdown("""
    El gráfico anterior muestra el **Peso Absoluto** o la **Magnitud del Efecto** que cada factor tiene en el resultado de la predicción, destacando las que tienen mayor influencia.
    
    ### 🥇 Variables de Mayor Impacto (Predictores Clave)

    Las variables con el mayor "Peso absoluto" ejercen la **mayor influencia** en la probabilidad de que un individuo sea clasificado como hipertenso o no hipertenso.
    
    1.  **`num_edad` (Edad numérica):** Con el peso más alto (alrededor de 1.1), la **edad es el factor predictivo fundamental**. Esto es consistente con el conocimiento médico, ya que el riesgo de hipertensión aumenta significativamente con la edad.
    2.  **`cat_ocupacion_Rentista` (Ocupación: Rentista):** El segundo factor más relevante (alrededor de 0.95), lo que sugiere que esta categoría ocupacional (a menudo asociada con mayor edad o menor actividad física laboral) tiene un impacto muy alto.
    
    ### 🥈 Factores de Salud y Sueño
    
    Los siguientes factores refuerzan la relevancia del estado de salud y los hábitos:
    
    * **`cat_sueño_simple_Muy Bien`:** Un peso alto indica que una **excelente calidad de sueño** es un factor protector.
    * **`cat_diabetes_Sí` / `cat_diabetes_No`:** El estado de diabetes es un predictor muy fuerte debido a la conocida comorbilidad entre ambas condiciones.
    * **`cat_sueño_simple_Muy Mal`:** Una pésima calidad de sueño también figura como un factor importante, lo que subraya la necesidad de considerar la salud del sueño en la evaluación de riesgo.

    La gráfica confirma que, si bien la **edad** es el predictor dominante, el modelo captura la compleja interacción de **condiciones sociolaborales** y **hábitos de salud** para una predicción más robusta.

    ---
    ## 🧠 El Modelo: Elastic Net (Regresión Logística)
    
    Elegimos la **Regresión Logística con regularización Elastic Net** por ser una opción que ofrece un equilibrio excepcional entre el poder predictivo y la interpretabilidad de los resultados.
    
    ### Justificación basada en el ROC-AUC
    """)
    st.image(
    "data/assets/curva_roc.png", 
    caption="Curva ROC de ejemplo y valor AUC para evaluar el modelo.", 
    use_column_width=True
    )
    st.markdown("""
    La métrica principal utilizada para seleccionar este modelo fue el **Área bajo la Curva ROC (ROC-AUC)**.
    
    ***¿Qué es el ROC-AUC?** Es una métrica de rendimiento que evalúa la capacidad de un modelo para distinguir entre las clases positivas (hipertenso) y negativas (no hipertenso). Un valor de **1.0** representa una predicción perfecta, mientras que **0.5** indica una predicción aleatoria.
    * **¿Por qué Elastic Net?** El modelo Elastic Net alcanzó un alto valor de ROC-AUC (**[0,823]**), demostrando una gran capacidad predictiva. Además, la regularización Elastic Net nos permite:
        * **Seleccionar Variables Clave (Lasso/L1):** Ceros o minimiza el impacto de variables menos relevantes, ayudando a que el modelo se enfoque en los factores de riesgo más importantes.
        * **Manejar Colinealidad (Ridge/L2):** Mejora la estabilidad del modelo, previniendo el sobreajuste (*overfitting*) al manejar la posible correlación entre múltiples factores de riesgo (ej: la edad y otros hábitos de salud).
    
    
    
    ---
    ## 🚀 Desafíos y Futuras Aplicaciones
    
    ### Desafíos de la Aplicación
    
    1.  **Dependencia de la Calidad de los Datos:** La precisión del modelo está limitada por la calidad, sesgos y representatividad de los datos originales del portal de datos abiertos de Chile.
    2.  **Generalización:** El modelo está optimizado para la población de Chile. Su aplicación directa a otras poblaciones con hábitos y sistemas de salud muy diferentes podría requerir un ajuste o reentrenamiento.
    3.  **No es un Diagnóstico:** Es fundamental recordar que la aplicación provee una **estimación de riesgo** y no reemplaza la consulta ni el diagnóstico clínico de un médico.
    
    ### Objetivos Futuros
    
    * **Integración Clínica:** Desarrollar una API que pueda ser consumida por sistemas de información de salud para facilitar la evaluación de riesgo en consultas médicas.
    * **Actualización Continua:** Integrar un proceso de actualización periódica del modelo con datos más recientes para mantener la relevancia y precisión predictiva.
    * **Análisis de Sensibilidad:** Realizar un análisis más profundo de la sensibilidad del modelo ante cambios pequeños en factores de estilo de vida para dar recomendaciones más detalladas.
    """)