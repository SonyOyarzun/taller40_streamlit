import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA

# --- Diccionario de interpretación de clústeres
interpretaciones = {
    0: "Súper compradores premium: Altas ventas, duración prolongada, maduros",
    1: "Jóvenes impulsivos insatisfechos: Baja edad, devoluciones altas, duración corta",
    2: "Impulsivos digitales: Exploración larga, comportamiento errático",
    3: "Neutrales prudentes: Bajas devoluciones (PCA1), edad variable, ventas medias, comportamiento mesurado",
    4: "Exploradores comprometidos: Edad media, duración más alta, devoluciones bajas"
}

# --- Pantalla de carga
@st.cache_data(show_spinner=False)
def load_data():
    with st.spinner("Cargando datos..."):
        df = pd.read_csv("Datos_Website_Logs.csv", sep=";", dayfirst=True)
        df.rename(columns={'accessed_Ffom': 'accessed_from'}, inplace=True)
        df['accessed_from'] = df['accessed_from'].replace('SafFRi', 'Safari')
        df = df.dropna(subset=['accessed_date'])
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['salesM$'] = pd.to_numeric(df['salesM$'], errors='coerce')
        df['age'] = df.groupby(['country', 'gender'])['age'].transform(lambda x: x.fillna(x.median()))
        df = df[df['salesM$'] < 100000]
        df['accessed_date'] = pd.to_datetime(df['accessed_date'], format='%d-%m-%Y %H:%M', errors='coerce')
        df['hour_of_day'] = df['accessed_date'].dt.hour
        df['net_sales'] = df['salesM$'] - df['returned_amountM$']
        return df

@st.cache_resource(show_spinner=False)
def load_scaler_kmeans():
    with st.spinner("Cargando modelos..."):
        scaler = joblib.load("scaler_kmeans.pkl")
        kmeans = joblib.load("kmeans_model.pkl")
        return scaler, kmeans

# --- Cargar
st.set_page_config(page_title="Análisis Clustering", layout="wide")
df = load_data()
scaler, kmeans = load_scaler_kmeans()
features = ['duration_(secs)', 'age', 'salesM$', 'returned_amountM$']

# --- Interfaz
st.title("🛒 Clustering de Usuarios + PCA")
seccion = st.sidebar.radio("Ir a:", ["📊 Vista de Datos", "🔵 Clustering + PCA", "✍️ Clasificar Nuevo Dato"])

if seccion == "📊 Vista de Datos":
    st.header("📊 Vista Previa del Dataset Limpio")
    st.dataframe(df.head())

elif seccion == "🔵 Clustering + PCA":
    st.header("🔵 Visualización de Clústeres KMeans + PCA")

    # Filtros
    pais = st.sidebar.multiselect("País:", df["country"].dropna().unique(), default=df["country"].dropna().unique())
    genero = st.sidebar.multiselect("Género:", df["gender"].dropna().unique(), default=df["gender"].dropna().unique())

    df_filtrado = df[(df["country"].isin(pais)) & (df["gender"].isin(genero))]
    X = df_filtrado[features].dropna()
    df_filtrado = df_filtrado.loc[X.index]
    X_scaled = scaler.transform(X)
    clusters = kmeans.predict(X_scaled)
    df_filtrado = df_filtrado.copy()
    df_filtrado["cluster"] = clusters

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    df_pca["PCA1"] = df_pca["PCA1"]
    df_pca["cluster"] = clusters

    # Gráfico PCA
    st.subheader("🌌 Visualización PCA por Clúster")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    legend_labels = [f"{c} - {interpretaciones[c].split(':')[0]}" for c in sorted(df_pca['cluster'].unique())]
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="cluster", palette="Set2", alpha=0.7, s=60, ax=ax1)
    ax1.set_xlim(-5, 3)
    ax1.grid(True)
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles, legend_labels, title="Clúster")
    st.pyplot(fig1)

    # Cargas de PCA
    st.subheader("📊 Contribución de Variables en PCA")
    cargas = pd.DataFrame(pca.components_.T, columns=["PCA1", "PCA2"], index=features)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    cargas.plot(kind="bar", ax=ax2)
    ax2.set_title("Cargas de Variables")
    ax2.set_ylabel("Contribución")
    ax2.grid(True)
    st.pyplot(fig2)

    st.dataframe(cargas.style.format(precision=3))

    # Perfil de clústeres
    st.subheader("🧠 Interpretación de Clústeres")
    cluster_info = pd.DataFrame({
        "Clúster": [0, 1, 2, 3, 4],
        "Color PCA": ["Verde oscuro", "Naranja", "Azul", "Amarillo", "Verde claro"],
        "Perfil sugerido": [
            "💰 Súper compradores premium",
            "🔶 Jóvenes impulsivos insatisfechos",
            "🔵 Impulsivos digitales",
            "👵 Neutrales prudentes",
            "🕵️‍♂️ Exploradores comprometidos"
        ],
        "Características principales": [
            "Altas ventas, duración prolongada, maduros",
            "Baja edad, devoluciones altas, duración corta",
            "Exploración larga, comportamiento errático",
            "Bajas devoluciones (PCA1), edad variable, ventas medias, comportamiento mesurado",
            "Edad media, duración más alta, devoluciones bajas"
        ]
    })
    st.dataframe(cluster_info)

elif seccion == "✍️ Clasificar Nuevo Dato":
    st.header("✍️ Clasificar nuevo dato")

    duracion = st.number_input("Duración conexión (secs)", min_value=0)
    edad = st.number_input("Edad", min_value=0)
    venta = st.number_input("Ventas M$", min_value=0.0, format="%.2f")
    devolucion = st.number_input("Monto devuelto M$", min_value=0.0, format="%.2f")

    if st.button("Clasificar"):
        try:
            nuevo = np.array([[duracion, edad, venta, devolucion]])
            nuevo_scaled = scaler.transform(nuevo)
            cluster_predicho = kmeans.predict(nuevo_scaled)[0]
            st.success(f"✅ Pertenece al clúster {cluster_predicho} — {interpretaciones[cluster_predicho]}")
        except Exception as e:
            st.error(f"❌ Error al clasificar: {e}")
