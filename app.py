import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ==========================================================
# CONFIGURACIÓN DE LA APP
# ==========================================================

st.set_page_config(
    page_title="Exploración de Datos", page_icon="📊")

st.sidebar.image("img/inEDA26.png", caption="Dr. Jesus Alvarado-Huayhuaz")

st.title("Ciencia de Datos")
st.write("Analiza, limpia y visualiza tus datos.")

# ==========================================================
# SESSION STATE
# ==========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

# ==========================================================
# SIDEBAR - NAVEGACIÓN
# ==========================================================

st.sidebar.title("Menú")

etapa = st.sidebar.radio(
    "Seleccione una etapa",
    [
        "Carga de datos",
        "Exploratory Data Analysis",
        "Limpieza y procesamiento",
        "Visualización"
    ]
)

# ==========================================================
# ETAPA 0: CARGA DE DATOS
# ==========================================================

if etapa == "Carga de datos":

    st.header("Carga de archivo CSV")

    archivo = st.file_uploader("Suba un archivo CSV", type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo)

        st.session_state.df = df
        st.session_state.processed_df = df.copy()

        st.success("Archivo cargado correctamente")

        st.write("Vista previa")
        st.dataframe(df.head())

# ==========================================================
# ETAPA 1: EDA
# ==========================================================

elif etapa == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    if st.session_state.df is None:
        st.warning("Primero cargue un archivo CSV")
    else:

        df = st.session_state.df

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dimensiones del dataset")
            st.write(df.shape)

        with col2:
            st.subheader("Tipos de datos")
            st.write(df.dtypes)

        st.subheader("Columnas numéricas")
        numeric_cols = df.select_dtypes(include=np.number).columns
        st.write(list(numeric_cols))

        st.subheader("Columnas categóricas")
        cat_cols = df.select_dtypes(include="object").columns
        st.write(list(cat_cols))

        st.subheader("Valores faltantes")
        st.dataframe(df.isnull().sum())

        st.subheader("Estadísticas descriptivas")
        st.dataframe(df.describe())

        # OUTLIERS

        st.subheader("Detección de outliers (IQR)")

        outliers = {}

        for col in numeric_cols:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            count = df[(df[col] < lower) | (df[col] > upper)].shape[0]

            outliers[col] = count

        st.write(outliers)

        st.subheader("Histogramas automáticos")

        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

# ==========================================================
# ETAPA 2: LIMPIEZA
# ==========================================================

elif etapa == "Limpieza y procesamiento":

    st.header("Limpieza y procesamiento de datos")

    if st.session_state.processed_df is None:
        st.warning("Primero cargue un dataset")
    else:

        df = st.session_state.processed_df

        st.subheader("DataFrame actual")
        st.dataframe(df.head())

        # ELIMINAR COLUMNAS

        st.subheader("Eliminar columnas")

        cols_delete = st.multiselect(
            "Seleccione columnas",
            df.columns
        )

        if st.button("Eliminar columnas"):
            df = df.drop(columns=cols_delete)
            st.session_state.processed_df = df
            st.success("Columnas eliminadas")

        # ELIMINAR NaN

        st.subheader("Eliminar NaN")

        if st.button("Eliminar filas con NaN"):
            df = df.dropna()
            st.session_state.processed_df = df
            st.success("Filas eliminadas")

        if st.button("Eliminar columnas con NaN"):
            df = df.dropna(axis=1)
            st.session_state.processed_df = df
            st.success("Columnas eliminadas")

        # DUPLICADOS

        if st.button("Eliminar duplicados"):
            df = df.drop_duplicates()
            st.session_state.processed_df = df
            st.success("Duplicados eliminados")

        # ONE HOT

        st.subheader("One Hot Encoding")

        cat_cols = df.select_dtypes(include="object").columns

        col_encode = st.selectbox(
            "Columna categórica",
            [""] + list(cat_cols)
        )

        if st.button("Aplicar One Hot Encoding") and col_encode != "":
            df = pd.get_dummies(df, columns=[col_encode])
            st.session_state.processed_df = df
            st.success("Encoding aplicado")

        # NORMALIZACIÓN

        st.subheader("Normalización")

        numeric_cols = df.select_dtypes(include=np.number).columns

        col_norm = st.selectbox(
            "Columna numérica",
            [""] + list(numeric_cols)
        )

        if st.button("Normalizar") and col_norm != "":
            df[col_norm] = (df[col_norm] - df[col_norm].min()) / (
                df[col_norm].max() - df[col_norm].min()
            )
            st.session_state.processed_df = df
            st.success("Normalización aplicada")

        # ESTANDARIZACIÓN

        if st.button("Estandarizar") and col_norm != "":
            df[col_norm] = (df[col_norm] - df[col_norm].mean()) / df[col_norm].std()
            st.session_state.processed_df = df
            st.success("Estandarización aplicada")

        # CÓDIGO PERSONALIZADO
        
        # ==========================================================
        # CÓDIGO PERSONALIZADO CON VISUALIZACIÓN DE RESULTADOS
        # ==========================================================
        
        st.subheader("Ejecutar código Python")
        
        st.write("Puede usar `df` como el DataFrame actual.")
        
        codigo = st.text_area(
            "Escriba código Python",
            height=150,
            placeholder='Ejemplo:\ndf["suma"] = df["A"] + df["B"]\ndf.head()'
        )
        
        if st.button("Ejecutar código"):
        
            try:
        
                # entorno seguro de ejecución
                local_env = {"df": df, "pd": pd, "np": np}
        
                resultado = None
        
                try:
                    # intentar evaluarlo como expresión
                    resultado = eval(codigo, {}, local_env)
        
                except:
                    # si no es expresión, ejecutarlo como bloque
                    exec(codigo, {}, local_env)
        
                # actualizar dataframe por si fue modificado
                df = local_env["df"]
                st.session_state.processed_df = df
        
                st.success("Código ejecutado correctamente")
        
                # mostrar resultado si existe
                if resultado is not None:
        
                    st.subheader("Resultado")
        
                    if isinstance(resultado, pd.DataFrame):
                        st.dataframe(resultado)
        
                    elif isinstance(resultado, pd.Series):
                        st.dataframe(resultado)
        
                    else:
                        st.write(resultado)
        
                # mostrar dataframe actualizado
                st.subheader("DataFrame actualizado")
                st.dataframe(df.head())
        
            except Exception as e:
                st.error(f"Error en el código: {e}")

        # DESCARGA CSV

        st.subheader("Descargar dataset procesado")

        nombre = st.text_input("Nombre del archivo")

        if nombre != "":

            csv = df.to_csv(index=False).encode()

            st.download_button(
                "Descargar CSV",
                csv,
                file_name=f"{nombre}.csv",
                mime="text/csv"
            )

# ==========================================================
# ETAPA 3: VISUALIZACIÓN
# ==========================================================

elif etapa == "Visualización":

    st.header("Visualización de resultados")

    if st.session_state.processed_df is None:
        st.warning("Primero cargue datos")
    else:

        df = st.session_state.processed_df

        plot_type = st.selectbox(
            "Tipo de gráfico",
            [
                "Histogram",
                "Boxplot",
                "Violin",
                "Scatter",
                "Pairplot",
                "Correlation heatmap"
            ]
        )

        numeric_cols = df.select_dtypes(include=np.number).columns

        fig = None

        if plot_type == "Histogram":

            col = st.selectbox("Columna", numeric_cols)

            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)

        elif plot_type == "Boxplot":

            col = st.selectbox("Columna", numeric_cols)

            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)

        elif plot_type == "Violin":

            col = st.selectbox("Columna", numeric_cols)

            fig, ax = plt.subplots()
            sns.violinplot(y=df[col], ax=ax)

        elif plot_type == "Scatter":

            x = st.selectbox("X", numeric_cols)
            y = st.selectbox("Y", numeric_cols)

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x, y=y, ax=ax)

        elif plot_type == "Pairplot":

            fig = sns.pairplot(df[numeric_cols])
            st.pyplot(fig)

        elif plot_type == "Correlation heatmap":

            corr = df.corr()

            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

        if fig is not None:
            st.pyplot(fig)

            buffer = BytesIO()
            fig.savefig(buffer, format="png")

            st.download_button(
                "Descargar gráfico",
                buffer.getvalue(),
                file_name="grafico.png",
                mime="image/png"
            )

        # CÓDIGO PERSONALIZADO

        st.subheader("Código de visualización personalizado")

        code_plot = st.text_area(
            "Escriba código usando df, plt, sns"
        )

        if st.button("Ejecutar gráfico personalizado"):

            try:

                exec(code_plot)

                fig = plt.gcf()

                st.pyplot(fig)

                buffer = BytesIO()
                fig.savefig(buffer, format="png")

                st.download_button(
                    "Descargar gráfico",
                    buffer.getvalue(),
                    file_name="custom_plot.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(e)
