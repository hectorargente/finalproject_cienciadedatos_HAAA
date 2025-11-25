# ========= IMPORTS =========
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import pydeck as pdk
import os
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from shapely import wkb
from scipy.spatial import Voronoi
import json

logging.basicConfig(level=logging.INFO)


def _find_col(cols, candidates):
    """Regresa el primer nombre de columna que exista en `cols`."""
    for c in candidates:
        if c in cols:
            return c
    return None

# IMPORTANTE: layout ancho
st.set_page_config(
    page_title="Modelo Go-to-Market Geoespacial",
    layout="wide"
)

# ===== VALIDACIÓN DE ARCHIVOS REQUERIDOS =====
REQUIRED_FILES = {
    "GEO_PARQUET_PATH": "georef-united-states-of-america-zc-point.parquet",
    "ZIP_DIM_PATH": "ZIP_Locale_Detail(ZIP_DETAIL).csv"
}

for file_key, file_path in REQUIRED_FILES.items():
    if not os.path.exists(file_path):
        st.error(f"❌ Archivo requerido no encontrado: {file_path}")
        st.stop()

# ===== CONFIGURACIÓN API CENSUS =====
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY") or "a185f4932d47ac5c7a92ba4187bf5b98b056994b"
CENSUS_YEAR = "2022"
CENSUS_DATASET = "acs/acs5"

CENSUS_VARS = [
    "NAME",
    "B03001_004E",  # Población mexicana
    "B01003_001E",  # Población total
    "B19013_001E"   # Ingreso medio del hogar
]

CENSUS_BASE_URL = f"https://api.census.gov/data/{CENSUS_YEAR}/{CENSUS_DATASET}"

# ===== FUNCIÓN PARA OBTENER DATOS DEL CENSUS =====
def fetch_census_data():
    """
    Llama a la API del U.S. Census (ACS 5-year) y regresa un DataFrame
    con población mexicana, población total e ingreso medio por ZIP.
    Además, muestra la respuesta cruda en caso de error para depurar.
    """
    params = {
        "get": ",".join(CENSUS_VARS),
        "for": "zip code tabulation area:*",
        "key": CENSUS_API_KEY
    }

    # Hacemos la petición
    response = requests.get(CENSUS_BASE_URL, params=params)

    # Intentamos parsear como JSON
    try:
        data = response.json()
    except Exception as e:
        st.error(f"No se pudo interpretar la respuesta como JSON: {e}")
        st.code(response.text[:500], language="text")
        raise

    # data[0] = encabezados, data[1:] = filas
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)
    return df

# ===== FUNCIÓN PARA CARGAR DATOS =====
@st.cache_data
def load_data():
    try:
        df = fetch_census_data()
        return df
    except Exception as e:
        st.error(f"Error al cargar datos del Census: {e}")
        return pd.DataFrame()

# ===== FUNCIÓN PARA CARGAR PUNTOS GEO (PARQUET) =====
@st.cache_data
def load_geo_points():
    """
    Carga el archivo Parquet con los puntos de ZIP codes (lat/lon)
    Convierte WKB geometry a coordenadas lat/lon
    """
    GEO_PARQUET_PATH = "georef-united-states-of-america-zc-point.parquet"

    gdf = pd.read_parquet(GEO_PARQUET_PATH)

    # Normalizar ZIP a 5 dígitos
    gdf["zip_code"] = gdf["zip_code"].astype(str).str.zfill(5)

    # Procesar geo_point_2d (WKB - Well-Known Binary format)
    if "geo_point_2d" in gdf.columns:
        def extract_coords(wkb_data):
            try:
                geom = wkb.loads(wkb_data)
                return pd.Series({"lat": geom.y, "lon": geom.x})
            except:
                return pd.Series({"lat": np.nan, "lon": np.nan})
        
        coords = gdf["geo_point_2d"].apply(extract_coords)
        gdf["lat"] = coords["lat"]
        gdf["lon"] = coords["lon"]
    else:
        raise ValueError("No se encontró la columna 'geo_point_2d' en el parquet")

    # Eliminar filas sin coordenadas válidas
    gdf = gdf.dropna(subset=["lat", "lon"])

    return gdf[["zip_code", "lat", "lon"]]

# ====== CARGA Y TRANSFORMACIONES GLOBALES ======
df = load_data()   # 1) cargamos solo una vez

# 2) Renombrar columnas para que sean más legibles
df = df.rename(columns={
    "B03001_004E": "pop_mexicana",
    "B01003_001E": "pop_total",
    "B19013_001E": "ingreso_medio",
    "zip code tabulation area": "zip_code"
})

# 3) Convertir a numérico
for col in ["pop_mexicana", "pop_total", "ingreso_medio"]:
    before_count = df[col].notna().sum()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    after_count = df[col].notna().sum()
    lost_records = before_count - after_count
    if lost_records > 0:
        logging.warning(f"Columna '{col}': {lost_records} registros convertidos a NaN durante conversión a numérico")

# 3b) Limpiar outliers del Census Bureau (códigos de error: valores negativos enormes)
logging.info("=== LIMPIEZA DE OUTLIERS ===")
before_clean = len(df)

# Eliminar negativos y valores del Census Bureau muy altos/bajos
df = df[
    (df["pop_mexicana"].isna() | (df["pop_mexicana"] >= 0)) &
    (df["pop_total"].isna() | (df["pop_total"] >= 0)) &
    (df["ingreso_medio"].isna() | ((df["ingreso_medio"] >= 1000) & (df["ingreso_medio"] <= 500000)))
].copy()

after_clean = len(df)
if before_clean > after_clean:
    logging.warning(f"Limpieza outliers Census: {before_clean - after_clean} registros removidos")

# Remover registros donde población mexicana > población total (error lógico)
before_logic = len(df)
df = df[df["pop_mexicana"] <= df["pop_total"]].copy()
after_logic = len(df)
if before_logic > after_logic:
    logging.warning(f"Limpieza lógica (pop_mexicana > pop_total): {before_logic - after_logic} registros removidos")

# 4) Crear porcentaje de población mexicana
df["pct_mexicana"] = (df["pop_mexicana"] / df["pop_total"]).fillna(0) * 100

# ====== CARGA DE DIMENSIÓN DE ZIPs (CIUDAD / ESTADO) ======
ZIP_DIM_PATH = "ZIP_Locale_Detail(ZIP_DETAIL).csv"

# 1) Leer CSV original
zip_dim_raw = pd.read_csv(ZIP_DIM_PATH, sep=";", dtype=str)

zip_dim_raw.columns = zip_dim_raw.columns.str.strip()

# 2) Normalizar ZIP a 5 dígitos
zip_dim_raw["zip_code"] = (
    zip_dim_raw["DELIVERY ZIPCODE"]
        .astype(str)
        .str.zfill(5)
)

# 3) Agrupar "CALIFORNIA 1", "CALIFORNIA 2"... en sólo "CALIFORNIA"
zip_dim_raw["state_group"] = (
    zip_dim_raw["DISTRICT NAME"]
        .str.replace(r"\s+\d+$", "", regex=True)
        .str.strip()
)

# 4) Quedarse con columnas importantes
zip_dim = (
    zip_dim_raw[[
        "zip_code",
        "PHYSICAL CITY",
        "PHYSICAL STATE",
        "state_group"
    ]]
    .drop_duplicates(subset=["zip_code"])
)

# ===== UNIÓN CON LA DIMENSIÓN DE ZIPs =====

# Aseguramos que zip_code tiene 5 dígitos en ambas bases
df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
zip_dim["zip_code"] = zip_dim["zip_code"].astype(str).str.zfill(5)

# Unimos por zip_code
before_merge = len(df)
df_merged = df.merge(zip_dim, on="zip_code", how="left")
after_merge = len(df_merged)

# Normalizamos nombres de ciudad/estado para usarlos en el modelo
df_merged = df_merged.rename(columns={
    "PHYSICAL CITY": "PHYSICAL_CITY",
    "PHYSICAL STATE": "PHYSICAL_STATE",
})

unmatched = (df_merged[["PHYSICAL_CITY", "PHYSICAL_STATE"]].isna().any(axis=1)).sum()
if unmatched > 0:
    logging.warning(f"Merge con ZIP_DIM: {unmatched} registros sin información de ciudad/estado")

# Limpieza adicional de outliers en df_merged
before_outlier_clean = len(df_merged)
df_merged = df_merged[
    (df_merged["pop_mexicana"].isna() | (df_merged["pop_mexicana"] >= 0)) &
    (df_merged["pop_total"].isna() | (df_merged["pop_total"] >= 0)) &
    (df_merged["ingreso_medio"].isna() | ((df_merged["ingreso_medio"] >= 1000) & (df_merged["ingreso_medio"] <= 500000))) &
    (df_merged["pop_mexicana"] <= df_merged["pop_total"])
].copy()
after_outlier_clean = len(df_merged)
if before_outlier_clean > after_outlier_clean:
    logging.warning(f"Limpieza de outliers en df_merged: {before_outlier_clean - after_outlier_clean} registros removidos")

# df_merged ahora incluye:
# - pop_mexicana
# - pop_total
# - ingreso_medio
# - pct_mexicana
# - zip_code
# - PHYSICAL CITY (ciudad)
# - PHYSICAL STATE (estado)
# - state_group (agrupación limpia)

@st.cache_data
def load_zip_dimension():
    ZIP_DIM_PATH = "ZIP_Locale_Detail(ZIP_DETAIL).csv"

    # Cargar archivo
    zip_dim_raw = pd.read_csv(ZIP_DIM_PATH, sep=";")

    # Normalizar ZIP a 5 dígitos
    if "DELIVERY ZIPCODE" in zip_dim_raw.columns:
        zip_dim_raw["zip_code"] = (
            zip_dim_raw["DELIVERY ZIPCODE"]
            .astype(str)
            .str.zfill(5)
        )
    else:
        st.error("❌ La columna 'DELIVERY ZIPCODE' NO existe en el archivo.")
        st.stop()

    # Crear agrupador de estado limpio
    zip_dim_raw["state_group"] = (
        zip_dim_raw["DISTRICT NAME"]
        .str.replace(r"\s+\d+$", "", regex=True)
        .str.strip()
    )

    # Seleccionamos columnas útiles
    zip_dim = zip_dim_raw[
        [
            "zip_code",
            "PHYSICAL CITY",
            "PHYSICAL STATE",
            "state_group",
        ]
    ].drop_duplicates(subset=["zip_code"])

    return zip_dim

# Función auxiliar para K-Means

def run_kmeans_auto(df_input, features, k_range=range(2, 9)):
    """
    Corre K-Means con diferentes k y selecciona el mejor usando Silhouette.
    Regresa:
      - df_clustered: df_input + columna 'cluster'
      - best_k: número óptimo de clusters
      - best_score: score de Silhouette
    """
    df_model = df_input.dropna(subset=features).copy()
    if df_model.empty:
        return df_input.assign(cluster=None), None, None

    X = df_model[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = None
    best_score = -1
    best_labels = None

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels

    df_model["cluster"] = best_labels
    # Volvemos a unir con el df original para conservar todos los ZIP
    df_output = df_input.merge(
        df_model[["zip_code", "cluster"]],
        on="zip_code",
        how="left"
    )
    return df_output, best_k, best_score

# ========= TÍTULO PRINCIPAL =========
st.title("Modelo Go-to-Market Geoespacial")
st.write("Dashboard inicial - Proyecto de Ciencia de Datos")

# Mostrar info de limpieza realizada
if after_outlier_clean < before_outlier_clean:
    with st.expander("ℹ️ Datos: Limpieza aplicada"):
        st.info(
            f"""
**Limpieza de datos aplicada:**
- ❌ Removidos **{before_outlier_clean - after_outlier_clean}** registros con outliers
  - Valores negativos en población o ingreso
  - Ingresos menores a $1,000 o mayores a $500,000
  - Población mexicana mayor a población total
  - Códigos de error del Census Bureau (-666,666,666)

**Registros finales:** {after_outlier_clean:,} ZIP codes
            """
        )

# ========= PESTAÑAS =========
tab_intro, tab_eda, tab_model, tab_geo = st.tabs(
    ["Introducción", "EDA Descriptiva", "Modelo K-Means", "Mapa Geoespacial"]
)

# ========= TAB 1: INTRODUCCIÓN =========
with tab_intro:
    st.subheader("Bienvenido 👋")
    st.write(
        """
        Esta aplicación será el prototipo de un modelo de Go-to-Market
        para el mercado mexicano en Estados Unidos.

        En las otras pestañas verás:
        - EDA descriptiva de zonas
        - Un modelo de clustering (K-Means)
        - Un mapa geoespacial con los resultados
        """
    )

# ========= TAB 2: EDA =========
with tab_eda:
    st.markdown("## Exploración Descriptiva del Dataset (EDA)")

    # 1) Cargamos datos "crudos" desde la API
    df = load_data()

    # 2) Renombrar columnas para que sean más legibles
    df = df.rename(columns={
        "B03001_004E": "pop_mexicana",
        "B01003_001E": "pop_total",
        "B19013_001E": "ingreso_medio",
        "zip code tabulation area": "zip_code"
    })

    # 3) Convertir a numérico (vienen como strings)
    for col in ["pop_mexicana", "pop_total", "ingreso_medio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) Crear porcentaje de población mexicana
    df["pct_mexicana"] = (df["pop_mexicana"] / df["pop_total"]).fillna(0) * 100

    # 5) Cargar dimensión de ZIPs (ciudad / estado) y unir
    zip_dim = load_zip_dimension()
    df = df.merge(zip_dim, on="zip_code", how="left")

    # 6) Limpiar datos extremos para que las gráficas sean legibles
    df_clean = df.copy()

    # Ingreso medio razonable: entre 0 y 250k
    df_clean = df_clean[
        (df_clean["ingreso_medio"] >= 0) &
        (df_clean["ingreso_medio"] <= 250000)
    ]

    # % mexicana entre 0 y 100
    df_clean = df_clean[
        (df_clean["pct_mexicana"] >= 0) &
        (df_clean["pct_mexicana"] <= 100)
    ]

    # 7) Layout: columna de filtros + columna principal
    col_filters, col_main = st.columns([1, 3])

    # ====== Filtros (siempre visibles a la izquierda) ======
    with col_filters:
        st.markdown("### 🧩 Filtros")

        # Rango de ingreso medio según los datos limpios
        ing_min = float(df_clean["ingreso_medio"].min())
        ing_max = float(df_clean["ingreso_medio"].max())

        pct_min = float(df_clean["pct_mexicana"].min())
        pct_max = float(df_clean["pct_mexicana"].max())

        zip_search = st.text_input("Buscar ZIP (ejemplo: 00601)")

        # Filtro por estado / región (state_group)
        state_options = sorted(df_clean["state_group"].dropna().unique())
        selected_states = st.multiselect(
            "Filtrar por estado / región",
            options=state_options
        )

        # Filtro por ciudad (dependiendo de estados seleccionados)
        if selected_states:
            city_options = sorted(
                df_clean[df_clean["state_group"].isin(selected_states)]["PHYSICAL CITY"]
                .dropna()
                .unique()
            )
        else:
            city_options = sorted(
                df_clean["PHYSICAL CITY"].dropna().unique()
            )

        selected_cities = st.multiselect(
            "Filtrar por ciudad",
            options=city_options
        )

        ingreso_range = st.slider(
            "Filtrar por ingreso medio",
            min_value=ing_min,
            max_value=ing_max,
            value=(ing_min, ing_max),
            step=1000.0,
        )

        pct_range = st.slider(
            "Filtrar por % mexicana",
            min_value=pct_min,
            max_value=pct_max,
            value=(pct_min, pct_max),
            step=1.0,
        )

    # ====== Contenido principal ======
    with col_main:
        st.markdown("### 📌 Primeros 10 registros (datos limpios)")
        st.dataframe(df_clean.head(10))

        # Aplicamos filtros para construir df_plot
        df_plot = df_clean.copy()

        # Filtro por texto de ZIP (contiene)
        if zip_search:
            df_plot = df_plot[df_plot["zip_code"].astype(str).str.contains(zip_search)]

        # Filtro por estados seleccionados
        if selected_states:
            df_plot = df_plot[df_plot["state_group"].isin(selected_states)]

        # Filtro por ciudades seleccionadas
        if selected_cities:
            df_plot = df_plot[df_plot["PHYSICAL CITY"].isin(selected_cities)]

        # Filtros numéricos
        df_plot = df_plot[
            (df_plot["ingreso_medio"] >= ingreso_range[0]) &
            (df_plot["ingreso_medio"] <= ingreso_range[1]) &
            (df_plot["pct_mexicana"] >= pct_range[0]) &
            (df_plot["pct_mexicana"] <= pct_range[1])
        ]

        st.markdown("### 📊 Datos filtrados")
        st.dataframe(df_plot)

        st.markdown("---")
        st.markdown("## 📈 Visualizaciones Estadísticas")

        # ============ HISTOGRAMAS ============
        st.markdown("#### Histogramas por variable")

        col1, col2 = st.columns(2)
        
        with col1:
            fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

            axes[0].hist(df_plot["pop_total"].dropna(), bins=30)
            axes[0].set_title("Población total", fontsize=10)

            axes[1].hist(df_plot["pop_mexicana"].dropna(), bins=30)
            axes[1].set_title("Población mexicana", fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(3, 2.5))
            ax.hist(df_plot["pct_mexicana"].dropna(), bins=30)
            ax.set_title("% Población mexicana", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        # ============ BOXPLOTS ============
        st.markdown("#### Boxplots (detección de outliers)")

        col1, col2 = st.columns(2)
        
        with col1:
            fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

            axes[0].boxplot(df_plot["ingreso_medio"].dropna())
            axes[0].set_title("Ingreso medio", fontsize=10)

            axes[1].boxplot(df_plot["pop_total"].dropna())
            axes[1].set_title("Población total", fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(3, 2.5))
            ax.boxplot(df_plot["pct_mexicana"].dropna())
            ax.set_title("% mexicana", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        # ============ RELACIÓN ENTRE VARIABLES (BURBUJAS) ============
        st.markdown("#### Relación entre variables (bubble chart)")

        if not df_plot.empty:
            col_chart, _ = st.columns([2, 1])
            with col_chart:
                x = df_plot["ingreso_medio"]
                y = df_plot["pct_mexicana"]
                size_raw = df_plot["pop_mexicana"]  # tamaño de la burbuja

                # Normalizar tamaño para que se vea bien
                size_norm = 300 * (size_raw / size_raw.max())

                fig, ax = plt.subplots(figsize=(5, 3.5))
                sc = ax.scatter(
                    x,
                    y,
                    s=size_norm,
                    alpha=0.5,
                    edgecolors="w",
                    linewidths=0.5
                )

                ax.set_xlabel("Ingreso medio (USD)")
                ax.set_ylabel("% población mexicana")
                ax.set_title(
                    "Bubble chart: ingreso vs % mexicana\n"
                    "(tamaño = población mexicana)"
                )

                # Marcar el ZIP con mayor % mexicana dentro de los filtrados
                idx_max_pct = df_plot["pct_mexicana"].idxmax()
                punto_max = df_plot.loc[idx_max_pct]

                ax.scatter(
                    punto_max["ingreso_medio"],
                    punto_max["pct_mexicana"],
                    s=400,
                    color="red",
                    edgecolors="black",
                    label=f"ZIP top % mexicana: {punto_max['zip_code']}"
                )
                ax.legend()

                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No hay datos para mostrar en la gráfica con los filtros actuales.")

        # ============ TOP MERCADOS ============
        st.markdown("## 🏙️ Top mercados por ingreso y % mexicana")

        if not df_plot.empty:
            top_markets = (
                df_plot.sort_values(
                    ["pct_mexicana", "ingreso_medio", "pop_total"],
                    ascending=[False, False, False]
                )
                .head(10)
                [["zip_code", "PHYSICAL CITY", "state_group",
                  "ingreso_medio", "pct_mexicana", "pop_total"]]
            )
            st.dataframe(top_markets)

        # ============ CONCLUSIONES AUTOMÁTICAS ============
        st.markdown("## 🧠 Conclusiones")

        if not df_plot.empty:
            zip_max_ingreso = df_plot.loc[df_plot["ingreso_medio"].idxmax()]
            zip_max_pct = df_plot.loc[df_plot["pct_mexicana"].idxmax()]
            zip_min_pct = df_plot.loc[df_plot["pct_mexicana"].idxmin()]

            st.markdown(
                f"""
                **Insights principales (sobre datos filtrados):**

                - El ZIP con mayor ingreso medio es **{zip_max_ingreso['zip_code']}** "
                  en **{zip_max_ingreso['PHYSICAL CITY']} ({zip_max_ingreso['state_group']})** "
                  con aproximadamente **${zip_max_ingreso['ingreso_medio']:,.0f}**.
                - El ZIP con mayor % de población mexicana es **{zip_max_pct['zip_code']}** "
                  en **{zip_max_pct['PHYSICAL CITY']} ({zip_max_pct['state_group']})** "
                  con **{zip_max_pct['pct_mexicana']:.2f}%** de población mexicana.
                - El ZIP con menor % de población mexicana es **{zip_min_pct['zip_code']}** "
                  en **{zip_min_pct['PHYSICAL CITY']} ({zip_min_pct['state_group']})**.
                - La variable con mayor dispersión sigue siendo el **ingreso medio**, "
                  lo que sugiere fuerte heterogeneidad económica entre códigos ZIP.
                - Al cruzar ingreso y % mexicana en la gráfica de burbujas, "
                  se identifican los mercados prioritarios para un Go-to-Market "
                  enfocado en hogares mexicanos.
                """
            )
        else:
            st.info("No hay datos disponibles para generar conclusiones con los filtros actuales.")

# ========= TAB 3: MODELO =========
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@st.cache_data(show_spinner=False)
def run_kmeans_segmentacion(df_input, features):
    """
    Recibe un dataframe con columnas numéricas en escala original,
    entrena K-Means para distintos k, selecciona el mejor según Silhouette
    y devuelve:
      - df_con_clusters: df_input + columna 'cluster'
      - centroids_df: centroides en escala ORIGINAL
      - sil_scores: dict {k: score}
      - best_k, best_score
    """
    # Copia para no tocar el original
    df_model = df_input.copy()

    # 1) Estandarizar variables numéricas
    X = df_model[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = X_scaled.shape[0]

    # Rango de k dinámico y acotado para acelerar
    k_max = min(6, n - 1)     # como máximo 6 clusters y < n
    k_min = 2
    if k_max < k_min + 1:
        # Muy pocos registros para hacer clustering razonable
        return None, None, {}, None, None

    k_values = list(range(k_min, k_max + 1))

    best_k = None
    best_score = -1
    best_labels = None
    sil_scores = {}

    # 2) Probar distintos k y calcular Silhouette
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Si por algún motivo todos caen en un solo cluster, Silhouette no sirve
        if len(set(labels)) == 1:
            score = -1
        else:
            score = silhouette_score(X_scaled, labels)

        sil_scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # 3) Si no se encontró nada razonable
    if best_k is None or best_labels is None:
        return None, None, sil_scores, None, None

    # 4) Agregamos los clusters al df
    df_model["cluster"] = best_labels

    # 5) Centroides en escala ORIGINAL: promedio por cluster
    centroids_df = (
        df_model.groupby("cluster")[features]
        .mean()
        .reset_index()
        .sort_values("cluster")
    )

    return df_model, centroids_df, sil_scores, best_k, best_score


def run_kmeans_manual(df_input, features, k_manual):
    """
    Entrena K-Means con un número específico de clusters (k_manual).
    Devuelve:
      - df_con_clusters: df_input + columna 'cluster'
      - centroids_df: centroides en escala ORIGINAL
    """
    df_model = df_input.copy()
    
    X = df_model[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k_manual, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    df_model["cluster"] = labels
    
    centroids_df = (
        df_model.groupby("cluster")[features]
        .mean()
        .reset_index()
        .sort_values("cluster")
    )
    
    return df_model, centroids_df


with tab_model:
    st.subheader("Modelo K-Means (segmentación de zonas)")

    # ================== PREPARACIÓN DE DATOS ==================
    # Aseguramos que exista el df_merged con ciudad/estado
    if "df_merged" not in globals() or df_merged.empty:
        st.error("No hay datos preparados para el modelo (df_merged está vacío).")
    else:
        # Columnas que usaremos
        features = ["pop_mexicana", "pop_total", "ingreso_medio", "pct_mexicana"]

        # Nos quedamos solo con filas completas en estas columnas
        df_model_base = df_merged.dropna(subset=features).copy()

        # ================== FILTRO EN SIDEBAR ==================
        with st.sidebar:
            st.markdown("### 🎯 Filtros para el modelo")

            # Filtro por estado
            states = (
                ["Todos"]
                + sorted(
                    df_model_base["PHYSICAL_STATE"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )
            state_selected = st.selectbox("Estado (PHYSICAL_STATE):", states, index=0)

            # Filtro por ciudad (dinámico según el estado)
            if state_selected != "Todos":
                df_temp = df_model_base[
                    df_model_base["PHYSICAL_STATE"].astype(str) == str(state_selected)
                ]
            else:
                df_temp = df_model_base
            
            cities = (
                ["Todas"]
                + sorted(
                    df_temp["PHYSICAL_CITY"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )
            city_selected = st.selectbox("Ciudad (PHYSICAL_CITY):", cities, index=0)
            
            # Selector de número de clusters
            st.markdown("#### Número de clusters")
            k_manual = st.slider(
                "Elige k (manual):",
                min_value=2,
                max_value=8,
                value=2,
                help="Selecciona un valor para usar clustering manual"
            )
            use_manual = st.checkbox("✋ Usar valor manual en lugar de automático (Silhouette)", value=False)

        # Filtramos por estado y ciudad (si aplica)
        df_model_filtered = df_model_base.copy()
        
        if state_selected != "Todos":
            df_model_filtered = df_model_filtered[
                df_model_filtered["PHYSICAL_STATE"].astype(str) == str(state_selected)
            ]
        
        if city_selected != "Todas":
            df_model_filtered = df_model_filtered[
                df_model_filtered["PHYSICAL_CITY"].astype(str) == str(city_selected)
            ]

        n_rows = len(df_model_filtered)

        st.markdown("**Variables utilizadas en el modelo**")
        st.write(features)
        
        filter_text = ""
        if state_selected != "Todos":
            filter_text += f" en {state_selected}"
        if city_selected != "Todas":
            filter_text += f", {city_selected}"
        
        st.write(f"Registros usados en el modelo (después de filtros{filter_text}): **{n_rows}**")

        if n_rows < 10:
            st.warning(
                "El estado seleccionado tiene muy pocos ZIP codes para entrenar un modelo de clustering robusto. "
                "Selecciona otro estado o usa 'Todos'."
            )
        else:
            # ================== ENTRENAMIENTO (CACHÉ) ==================
            if use_manual and k_manual is not None:
                with st.spinner(f"Entrenando modelo K-Means con k={k_manual}..."):
                    df_km, centroids_df = run_kmeans_manual(df_model_filtered, features, k_manual)
                best_k = k_manual
                best_score = None
                sil_scores = {}
            else:
                with st.spinner("Entrenando modelo K-Means y calculando Silhouette..."):
                    df_km, centroids_df, sil_scores, best_k, best_score = run_kmeans_segmentacion(
                        df_model_filtered, features
                    )

            st.session_state["df_km"] = df_km

            if df_km is None or best_k is None:
                st.error("No fue posible encontrar una configuración de clusters válida.")
            else:
                # ================== RESULTADOS ==================

                # Tabla de Silhouette por k (solo si se usó automático)
                if sil_scores:
                    st.markdown("### Selección automática del número de clusters (k - Silhouette Score)")
                    sil_df = (
                        pd.DataFrame(
                            [{"k": k, "silhouette": score} for k, score in sil_scores.items()]
                        )
                        .sort_values("k")
                        .reset_index(drop=True)
                    )
                    st.dataframe(sil_df, use_container_width=True)

                    st.success(
                        f"El número óptimo de clusters según el coeficiente de Silhouette es **k = {best_k}** "
                        f"(score = {best_score:.3f})."
                    )
                else:
                    st.info(f"✅ Modelo entrenado con **k = {best_k}** clusters (selección manual).")

                # Centroides
                st.markdown("### Centroides de los clusters (en escala original)")
                st.dataframe(centroids_df, use_container_width=True)

                # Distribución de ZIP codes por cluster
                st.markdown("### Distribución de ZIP codes por cluster")
                cluster_counts = df_km["cluster"].value_counts().sort_index()
                col_bar, _ = st.columns([2, 1])
                with col_bar:
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    cluster_counts.plot(kind="bar", ax=ax, color="steelblue")
                    ax.set_xlabel("Cluster")
                    ax.set_ylabel("Cantidad de ZIP codes")
                    ax.set_title("Distribución por cluster")
                    plt.tight_layout()
                    st.pyplot(fig)

                # Ejemplo de registros con su cluster
                st.markdown("### Ejemplo de registros con cluster asignado")
                cols_show = ["zip_code", "PHYSICAL_CITY", "PHYSICAL_STATE"] + features + [
                    "cluster"
                ]
                st.dataframe(df_km[cols_show].head(30), use_container_width=True)

                # ================== CONCLUSIONES AUTOMÁTICAS ==================
                st.markdown("### 🧠 Conclusiones automáticas del modelo")

                if state_selected == "Todos" and city_selected == "Todas":
                    location_text = "todo Estados Unidos"
                elif state_selected == "Todos":
                    location_text = f"la ciudad de **{city_selected}**"
                elif city_selected == "Todas":
                    location_text = f"el estado de **{state_selected}**"
                else:
                    location_text = f"**{city_selected}**, {state_selected}"

                st.markdown(
                    f"""
- Se identificaron **{best_k} clusters** de ZIP codes para {location_text}.
- El dataset utilizado para este modelo contiene **{n_rows} ZIP codes** después de aplicar los filtros.
- Los centroides muestran cómo varían la **población total**, la **población mexicana** y el **ingreso medio**
  entre los distintos segmentos.
- Los clusters con ingresos medios más altos son candidatos naturales para lanzar **SKUs premium**;
  los de ingreso bajo, para **líneas de entrada** o formatos más pequeños.
- Combinando esta segmentación con el análisis de mercado (ventas por ZIP / retailer) podrás decidir
  en qué zonas enfocar los lanzamientos de nuevos productos.
"""
                )

# ========= TAB 4: MAPA GEOESPACIAL =========

with tab_geo:
    st.markdown("## 🗺️ Mapa Geoespacial")
    st.write(
        "Visualizamos los ZIP codes segmentados sobre un mapa tipo Uber, "
        "usando PyDeck. Cada punto representa un ZIP code; el color indica el cluster."
    )

    # Ruta del GeoParquet que descargaste
    GEO_PARQUET_PATH = "georef-united-states-of-america-zc-point.parquet"  # ajusta el nombre si es distinto

    @st.cache_data
    def load_geo_points():
        from shapely.wkb import loads as wkb_loads
        
        gdf = pd.read_parquet(GEO_PARQUET_PATH)

        if "zip_code" not in gdf.columns:
            st.error(
                "El archivo GeoParquet no contiene la columna 'zip_code'. "
                f"Columnas disponibles: {list(gdf.columns)}"
            )
            return None

        gdf["zip_code"] = gdf["zip_code"].astype(str).str.zfill(5)

        if "geo_point_2d" in gdf.columns:
            coord_col = "geo_point_2d"
        elif "Geo Point" in gdf.columns:
            coord_col = "Geo Point"
        else:
            st.error(
                "No se encontró la columna de coordenadas ('geo_point_2d' o 'Geo Point') "
                f"en el GeoParquet. Columnas disponibles: {list(gdf.columns)}"
            )
            return None

        lats = []
        lons = []
        for geom_data in gdf[coord_col]:
            try:
                point = wkb_loads(geom_data)
                lats.append(point.y)
                lons.append(point.x)
            except Exception as e:
                logging.warning(f"Failed to parse geometry: {e}")
                lats.append(None)
                lons.append(None)

        gdf["lat"] = lats
        gdf["lon"] = lons
        gdf = gdf.dropna(subset=["lat", "lon"])

        return gdf[["zip_code", "lat", "lon"]]

    geo_df = load_geo_points()
    if geo_df is None:
        st.stop()

    # -------- Dataset del modelo con clusters --------
    if "df_km" in st.session_state and st.session_state["df_km"] is not None:
        df_model = st.session_state["df_km"].copy()
    else:
        try:
            df_model = df_merged.copy()
        except NameError:
            st.error("No encontré el dataframe 'df_merged' en tu app. Ajusta el nombre aquí para usar el dataset correcto.")
            st.stop()

        st.info("💡 Tip: Ve a la pestaña 'Modelo K-Means' para entrenar un modelo y obtener clusters más precisos.")

    # Aseguramos formato 5 dígitos también aquí
    df_model["zip_code"] = df_model["zip_code"].astype(str).str.zfill(5)

    # Si por alguna razón aún no tienes la columna 'cluster', creamos una de emergencia
    if "cluster" not in df_model.columns:
        st.warning(
            "No encontré la columna 'cluster' en el dataset del modelo. "
            "Voy a crear clusters ficticios basados en cuantiles de ingreso_medio solo para poder mostrar el mapa."
        )
        df_model = df_model.copy()
        df_model["cluster"] = pd.qcut(
            df_model["ingreso_medio"], 5, labels=False, duplicates="drop"
        )

    # -------- Join entre ACS+modelo y GeoParquet por zip_code --------
    before_geo_join = len(df_model)
    df_map = pd.merge(df_model, geo_df, on="zip_code", how="inner")
    after_geo_join = len(df_map)
    lost_in_geo_join = before_geo_join - after_geo_join
    
    if lost_in_geo_join > 0:
        logging.warning(f"Merge con GeoParquet: {lost_in_geo_join} registros no encontraron coordenadas geográficas")

    if df_map.empty:
        st.error(
            "Después del join entre ACS y el GeoParquet no quedaron registros. "
            "Revisa que los ZIP codes estén en formato de 5 dígitos en ambas fuentes."
        )
        st.stop()

    # Centro del mapa
    midpoint = (df_map["lat"].mean(), df_map["lon"].mean())

    # Selector de visualización
    viz_mode = st.radio(
        "📊 Tipo de visualización:",
        ["📍 Polígonos por cluster", "🔥 Mapa de calor (hexágonos)"],
        horizontal=True
    )

    # Capa de hexágonos (Mapa de calor)
    hexagon_layer = pdk.Layer(
        "HexagonLayer",
        data=df_map,
        get_position="[lon, lat]",
        radius=8000,
        elevation_scale=50,
        elevation_range=[0, 3000],
        pickable=True,
        extruded=True,
        colorRange=[
            [0, 0, 255],       # Azul = Baja densidad
            [0, 255, 0],       # Verde
            [255, 255, 0],     # Amarillo
            [255, 127, 0],     # Naranja
            [255, 0, 0],       # Rojo = Alta densidad
        ],
    )

    # Crear polígonos Voronoi coloreados por cluster
    @st.cache_data
    def create_voronoi_polygons(df_data):
        """Crea polígonos Voronoi para cada ZIP code"""
        coords = df_data[["lon", "lat"]].values
        
        if len(coords) < 4:
            return None
        
        try:
            vor = Voronoi(coords)
            
            # Paleta de colores para clusters (8 colores distintos)
            cluster_colors = {
                0: [255, 0, 0, 150],        # Rojo
                1: [0, 0, 255, 150],        # Azul
                2: [0, 255, 0, 150],        # Verde
                3: [255, 255, 0, 150],      # Amarillo
                4: [255, 0, 255, 150],      # Magenta
                5: [0, 255, 255, 150],      # Cian
                6: [255, 165, 0, 150],      # Naranja
                7: [128, 0, 128, 150],      # Púrpura
            }
            
            features = []
            for idx, (zip_code, cluster) in enumerate(zip(df_data["zip_code"].values, df_data["cluster"].values)):
                region = vor.regions[vor.point_region[idx]]
                
                if -1 not in region and len(region) > 0:
                    vertices = vor.vertices[region]
                    if len(vertices) > 2:
                        cluster_id = int(cluster) if cluster is not None else 0
                        color = cluster_colors.get(cluster_id % 8, [128, 128, 128, 150])
                        
                        polygon = {
                            "type": "Feature",
                            "properties": {
                                "zip_code": str(zip_code),
                                "cluster": cluster_id,
                                "color": color
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[v[0], v[1]] for v in vertices] + [[vertices[0][0], vertices[0][1]]]]
                            }
                        }
                        features.append(polygon)
            
            return {
                "type": "FeatureCollection",
                "features": features
            }
        except Exception as e:
            st.warning(f"Error creando Voronoi: {e}")
            return None
    
    voronoi_geojson = create_voronoi_polygons(df_map)
    
    if voronoi_geojson:
        polygon_layer = pdk.Layer(
            "GeoJsonLayer",
            data=voronoi_geojson,
            stroked=True,
            filled=True,
            line_width_min_pixels=1,
            get_fill_color="properties.color",
            get_line_color="[255, 255, 255, 100]",
            pickable=True,
        )
    else:
        st.error("No se pudo crear la visualización de polígonos")
        st.stop()

    # Seleccionar capa según modo
    if viz_mode == "📍 Polígonos por cluster":
        layers = [polygon_layer]
        tooltip = {
            "html": (
                "<b>ZIP:</b> {zip_code}<br/>"
                "<b>Cluster:</b> {cluster}"
            ),
            "style": {"backgroundColor": "black", "color": "white"},
        }
    else:  # Mapa de calor
        layers = [hexagon_layer]
        tooltip = {
            "html": "<b>Hexágono de densidad</b><br/>Datos agregados de la zona",
            "style": {"backgroundColor": "black", "color": "white"},
        }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=3,
            pitch=40,
        ),
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip=tooltip,
    )
    
    st.markdown("""
        <style>
            .deckgl-widget { display: none !important; }
            div[data-testid="stPydeck"] .mapboxgl-ctrl-top-right { display: none !important; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.pydeck_chart(deck, use_container_width=True)

    # -------- Conclusiones rápidas --------
    st.markdown("### 🧠 Conclusiones automáticas del mapa")

    n_clusters = df_map["cluster"].nunique()
    top_zip = (
        df_map.sort_values("ingreso_medio", ascending=False)
        .iloc[0][["zip_code", "ingreso_medio", "pct_mexicana", "cluster"]]
    )

    st.markdown(
        f"""
- Se están visualizando **{len(df_map):,} ZIP codes** después del join con el GeoParquet.
- El modelo está usando **{n_clusters} clusters** para segmentar las zonas.
- El ZIP con mayor ingreso medio en el mapa es **{top_zip['zip_code']}**,  
  con un ingreso aproximado de **${top_zip['ingreso_medio']:,.0f}** y  
  **{top_zip['pct_mexicana']:.1f}%** de población mexicana (cluster **{int(top_zip['cluster'])}**).
        """
    )
    
    # -------- Estadísticas por cluster --------
    st.markdown("### 📊 Análisis por cluster")
    
    cluster_stats = (
        df_map.groupby("cluster").agg({
            "zip_code": "count",
            "ingreso_medio": ["mean", "min", "max"],
            "pct_mexicana": "mean",
            "PHYSICAL_CITY": lambda x: x.mode()[0] if len(x.mode()) > 0 else "N/A"
        }).round(0)
    )
    cluster_stats.columns = ["ZIP codes", "Ingreso medio", "Ingreso mín", "Ingreso máx", "% Mexicana", "Ciudad principal"]
    st.dataframe(cluster_stats, use_container_width=True)
    
    # Selector de cluster para ver detalles
    st.markdown("### 🔍 Detalle de ZIP codes por cluster")
    selected_cluster = st.selectbox(
        "Selecciona un cluster para ver sus ZIP codes:",
        sorted(df_map["cluster"].unique()),
        format_func=lambda x: f"Cluster {int(x)}"
    )
    
    df_cluster = df_map[df_map["cluster"] == selected_cluster][
        ["zip_code", "PHYSICAL_CITY", "PHYSICAL_STATE", "ingreso_medio", "pct_mexicana", "pop_total"]
    ].sort_values("ingreso_medio", ascending=False)
    
    st.write(f"**{len(df_cluster)} ZIP codes** en este cluster")
    st.dataframe(df_cluster, use_container_width=True)
