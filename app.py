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
from shapely.geometry import Polygon, Point
import json

logging.basicConfig(level=logging.INFO)


def _find_col(cols, candidates):
    """Returns the first column name that exists in `cols`."""
    for c in candidates:
        if c in cols:
            return c
    return None

# IMPORTANT: wide layout
st.set_page_config(
    page_title="Geospatial Go-to-Market Model",
    layout="wide"
)

# ===== REQUIRED FILES VALIDATION =====
REQUIRED_FILES = {
    "GEO_PARQUET_PATH": "georef-united-states-of-america-zc-point.parquet",
    "ZIP_DIM_PATH": "ZIP_Locale_Detail(ZIP_DETAIL).csv"
}

for file_key, file_path in REQUIRED_FILES.items():
    if not os.path.exists(file_path):
        st.error(f"❌ Required file not found: {file_path}")
        st.stop()

# ===== CENSUS API CONFIGURATION =====

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
if not CENSUS_API_KEY:
    st.error("❌ CENSUS_API_KEY environment variable not configured. Please configure it in Streamlit Cloud Secrets.")
    st.stop()
CENSUS_YEAR = "2022"
CENSUS_DATASET = "acs/acs5"

CENSUS_VARS_BASE = [
    "NAME",
    "B03001_004E",  # Mexican population
    "B01003_001E",  # Total population
    "B19013_001E"   # Median household income
]

CENSUS_VARS_OPTIONAL = [
    "B16001_002E",  # Spanish spoken at home
    "B15002_001E"   # Educational attainment
]

CENSUS_VARS = CENSUS_VARS_BASE + CENSUS_VARS_OPTIONAL

VARIABLE_LABELS = {
    "pop_mexicana": "Mexican Population",
    "pop_total": "Total Population",
    "ingreso_medio": "Median Household Income",
    "spanish_home": "Spanish Spoken at Home",
    "education": "Educational Attainment"
}

CENSUS_BASE_URL = f"https://api.census.gov/data/{CENSUS_YEAR}/{CENSUS_DATASET}"

# ===== FUNCIÓN PARA OBTENER DATOS DEL CENSUS =====
def fetch_census_data():
    """
    Calls the U.S. Census API (ACS 5-year) and returns a DataFrame.
    Tries with optional variables first, falls back to base variables if needed.
    """
    census_vars_to_use = CENSUS_VARS
    
    params = {
        "get": ",".join(census_vars_to_use),
        "for": "zip code tabulation area:*",
        "key": CENSUS_API_KEY
    }

    try:
        logging.info(f"Fetching Census data with {len(census_vars_to_use)} variables...")
        response = requests.get(CENSUS_BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logging.warning("Timeout with full variable set, falling back to base variables...")
        census_vars_to_use = CENSUS_VARS_BASE
        params["get"] = ",".join(census_vars_to_use)
        try:
            response = requests.get(CENSUS_BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"❌ Census API error (even with base variables): {e}")
            raise
    except requests.exceptions.RequestException as e:
        logging.warning(f"Request error: {e}, trying base variables...")
        census_vars_to_use = CENSUS_VARS_BASE
        params["get"] = ",".join(census_vars_to_use)
        try:
            response = requests.get(CENSUS_BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"❌ Census API error: {e}")
            raise

    try:
        if isinstance(data, dict) and "error" in data:
            st.warning(f"Census API returned error: {data['error']}")
            logging.warning(f"Census API error response: {data}")
            census_vars_to_use = CENSUS_VARS_BASE
            params["get"] = ",".join(census_vars_to_use)
            response = requests.get(CENSUS_BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)
        logging.info(f"Successfully fetched Census data with columns: {list(df.columns)}")
        return df
    except Exception as e:
        st.error(f"Could not parse Census response: {e}")
        raise

# ===== LOAD DATA FUNCTION =====
@st.cache_data(show_spinner=False)
def load_data():
    try:
        with st.spinner("⏳ Loading Census Bureau data (may take 1-2 minutes)..."):
            df = fetch_census_data()
        return df
    except Exception as e:
        st.error(f"Error loading Census data: {e}")
        return pd.DataFrame()

# ===== LOAD GEO POINTS FUNCTION (PARQUET) =====
@st.cache_data
def load_geo_points():
    """
    Loads the Parquet file with ZIP code points (lat/lon)
    Converts WKB geometry to lat/lon coordinates
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

# ===== VORONOI POLYGON GENERATION =====
def generate_voronoi_polygons(df_map):
    """
    Generates Voronoi polygons for ZIP codes and returns as GeoJSON features.
    Only creates polygons within the continental US bounds to avoid edge artifacts.
    """
    if df_map.empty or len(df_map) < 3:
        return []
    
    points = df_map[["lon", "lat"]].values
    
    try:
        vor = Voronoi(points)
    except Exception as e:
        logging.error(f"Voronoi computation failed: {e}")
        return []
    
    us_bounds = {
        "min_lon": -125,
        "max_lon": -66,
        "min_lat": 24,
        "max_lat": 50
    }
    
    features = []
    cluster_colors = {
        0: [255, 0, 0, 180], 1: [0, 0, 255, 180], 2: [0, 255, 0, 180], 3: [255, 255, 0, 180],
        4: [255, 0, 255, 180], 5: [0, 255, 255, 180], 6: [255, 165, 0, 180], 7: [128, 0, 128, 180]
    }
    
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        
        if len(region) < 3 or -1 in region:
            continue
        
        vertices = vor.vertices[region]
        
        clipped_vertices = []
        for v in vertices:
            lon, lat = v
            if (us_bounds["min_lon"] <= lon <= us_bounds["max_lon"] and
                us_bounds["min_lat"] <= lat <= us_bounds["max_lat"]):
                clipped_vertices.append([lon, lat])
        
        if len(clipped_vertices) >= 3:
            try:
                polygon = Polygon(clipped_vertices)
                if polygon.is_valid:
                    cluster = int(df_map.iloc[i].get("cluster", 0))
                    color = cluster_colors.get(cluster, [128, 128, 128, 180])
                    
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [clipped_vertices]
                        },
                        "properties": {
                            "zip_code": str(df_map.iloc[i].get("zip_code", "")),
                            "cluster": str(cluster),
                            "cluster_display": f"Cluster {cluster}",
                            "color": color,
                            "income": float(df_map.iloc[i].get("ingreso_medio", 0))
                        }
                    })
            except Exception as e:
                logging.debug(f"Polygon creation failed for region {i}: {e}")
    
    return features

# ====== FUNCIÓN PARA PREPARAR Y LIMPIAR DATOS ======
@st.cache_data(show_spinner=False)
def prepare_census_data():
    """Loads, cleans, and prepares Census data"""
    df = load_data()   # 1) cargamos solo una vez

    # 2) Renombrar columnas para que sean más legibles
    rename_map = {
        "B03001_004E": "pop_mexicana",
        "B01003_001E": "pop_total",
        "B19013_001E": "ingreso_medio",
        "B16001_002E": "spanish_home",
        "B15002_001E": "education",
        "zip code tabulation area": "zip_code"
    }
    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # 3) Convertir a numérico
    numeric_cols = ["pop_mexicana", "pop_total", "ingreso_medio", "spanish_home", "education"]
    for col in numeric_cols:
        if col in df.columns:
            before_count = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_count = df[col].notna().sum()
            lost_records = before_count - after_count
            if lost_records > 0:
                logging.warning(f"Columna '{col}': {lost_records} registros convertidos a NaN durante conversión a numérico")
        else:
            logging.info(f"Columna opcional '{col}' no disponible en los datos del Census")
            df[col] = np.nan

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
        logging.warning(f"Limpieza outliers Census: {before_clean - after_clean} records removed")

    # Remover registros donde población mexicana > población total (error lógico)
    before_logic = len(df)
    df = df[df["pop_mexicana"] <= df["pop_total"]].copy()
    after_logic = len(df)
    if before_logic > after_logic:
        logging.warning(f"Logical cleanup (Mexican population > total population): {before_logic - after_logic} records removed")

    # 4) Crear porcentaje de población mexicana
    df["pct_mexicana"] = (df["pop_mexicana"] / df["pop_total"]).fillna(0) * 100
    
    return df, before_clean, after_clean

# ====== FUNCIÓN PARA PREPARAR DATOS CON DIMENSIÓN DE ZIPs =====
@st.cache_data(show_spinner=False)
def prepare_merged_data():
    """Loads and prepares merged data with ZIP dimension"""
    with st.spinner("📊 Preparing and cleaning data (this may take 2-3 minutes)..."):
        df, before_clean, after_clean = prepare_census_data()
        
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
            logging.warning(f"Merge with ZIP_DIM: {unmatched} records without city/state information")

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
            logging.warning(f"Outlier cleanup en df_merged: {before_outlier_clean - after_outlier_clean} records removed")
    
    return df_merged, before_outlier_clean, after_outlier_clean

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
    Runs K-Means with different k and selects the best using Silhouette.
    Returns:
      - df_clustered: df_input + column 'cluster'
      - best_k: optimal number of clusters
      - best_score: Silhouette score
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

# ========= PESTAÑAS =========
tab_intro, tab_eda, tab_model, tab_geo, tab_sku = st.tabs(
    ["Introduction", "Descriptive EDA", "K-Means Model", "Geospatial Map", "SKU Strategy"]
)

# ========= TAB 1: INTRODUCCIÓN =========
with tab_intro:
    st.subheader("Welcome 👋")
    st.write(
        """
        This application will be the prototype of a Go-to-Market model
        for the Mexican market in the United States.

        In the other tabs you will see:
        - Descriptive EDA of zones
        - A clustering model (K-Means)
        - A geospatial map with the results
        """
    )
    
    if "df_merged" not in st.session_state:
        df_merged, before_outlier_clean, after_outlier_clean = prepare_merged_data()
        st.session_state["df_merged"] = df_merged
        st.session_state["outlier_clean_stats"] = (before_outlier_clean, after_outlier_clean)
    else:
        df_merged = st.session_state["df_merged"]
        before_outlier_clean, after_outlier_clean = st.session_state.get("outlier_clean_stats", (0, 0))
    
    if after_outlier_clean < before_outlier_clean:
        with st.expander("ℹ️ Datos: Limpieza aplicada"):
            st.info(
                f"""
**Data cleaning applied:**
- ❌ Removed **{before_outlier_clean - after_outlier_clean}** records with outliers
  - Negative values in population or income
  - Incomes menores a $1,000 o mayores a $500,000
  - Population mexicana mayor a población total
  - Census Bureau error codes (-666,666,666)

**Final records:** {after_outlier_clean:,} ZIP codes
                """
            )

# ========= TAB 2: EDA =========
with tab_eda:
    st.markdown("## Descriptive Data Exploration (EDA)")

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

    # Income medio razonable: entre 0 y 250k
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
        st.markdown("### 🧩 Filters")

        # Range of median income according to clean data
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

        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.hist(df_plot["pop_total"].dropna(), bins=30)
            ax.set_title("Population total", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.hist(df_plot["pop_mexicana"].dropna(), bins=30)
            ax.set_title("Population mexicana", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.hist(df_plot["pct_mexicana"].dropna(), bins=30)
            ax.set_title("% Population mexicana", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        # ============ BOXPLOTS ============
        st.markdown("#### Boxplots (detección de outliers)")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.boxplot(df_plot["ingreso_medio"].dropna())
            ax.set_title("Income medio", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.boxplot(df_plot["pop_total"].dropna())
            ax.set_title("Population total", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots(figsize=(4, 2.5))
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

                ax.set_xlabel("Income medio (USD)")
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
            st.info("No data available para generar conclusiones con los filtros actuales.")

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
        # Too few records for reasonable clustering
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
    st.subheader("K-Means Model (Zone Segmentation)")

    # ================== PREPARACIÓN DE DATOS ==================
    # Aseguramos que exista el df_merged con ciudad/estado
    if "df_merged" not in st.session_state or st.session_state["df_merged"].empty:
        st.error("No data prepared para el modelo (df_merged is empty).")
    else:
        df_merged = st.session_state["df_merged"]
        # Available features for clustering - start with base mandatory features
        mandatory_features = ["pop_mexicana", "pop_total", "ingreso_medio", "pct_mexicana"]
        optional_features = ["spanish_home", "education"]
        
        available_features = [f for f in mandatory_features if f in df_merged.columns]
        for f in optional_features:
            if f in df_merged.columns:
                available_features.append(f)
        
        if not available_features:
            st.error("No features available for clustering")
            st.stop()

        # Dropna only on mandatory features to keep more data
        mandatory_for_dropna = [f for f in mandatory_features if f in df_merged.columns]
        df_model_base = df_merged.dropna(subset=mandatory_for_dropna).copy()
        
        if df_model_base.empty:
            st.error("No data available after cleaning")
            st.stop()

        # ================== FILTRO EN SIDEBAR ==================
        with st.sidebar:
            st.markdown("### 🎯 Model filters")

            # Variable selection
            st.markdown("#### Select variables for clustering")
            default_features = ["pop_mexicana", "ingreso_medio", "spanish_home"]
            features_selected = st.multiselect(
                "Choose variables to use in K-Means:",
                options=available_features,
                default=[f for f in default_features if f in available_features],
                format_func=lambda x: VARIABLE_LABELS.get(x, x)
            )
            
            if not features_selected:
                st.warning("⚠️ Please select at least one variable for clustering")
                features = None
            else:
                features = features_selected

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
            state_selected = st.selectbox("State (PHYSICAL_STATE):", states, index=0)

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
            city_selected = st.selectbox("City (PHYSICAL_CITY):", cities, index=0)
            
            # Selector de número de clusters
            st.markdown("#### Number of clusters")
            k_manual = st.slider(
                "Choose k (manual):",
                min_value=2,
                max_value=8,
                value=2,
                help="Selecciona un valor para usar clustering manual"
            )
            use_manual = st.checkbox("✋ Use manual value instead of automatic (Silhouette)", value=False)

        if features is None or len(features) == 0:
            st.error("Please select at least one variable for clustering")
        else:
            # Remove features that are all NaN
            features_with_data = [f for f in features if df_model_base[f].notna().sum() > 0]
            if not features_with_data:
                st.error("Selected features have no data. Please select different features.")
                st.stop()
            
            if len(features_with_data) < len(features):
                st.warning(f"Removed features with no data. Using: {[VARIABLE_LABELS.get(f, f) for f in features_with_data]}")
                features = features_with_data
            
            # Filtramos por estado y ciudad (si aplica)
            df_model_filtered = df_model_base[df_model_base[features].notna().all(axis=1)].copy()
            
            if state_selected != "Todos":
                df_model_filtered = df_model_filtered[
                    df_model_filtered["PHYSICAL_STATE"].astype(str) == str(state_selected)
                ]
            
            if city_selected != "Todas":
                df_model_filtered = df_model_filtered[
                    df_model_filtered["PHYSICAL_CITY"].astype(str) == str(city_selected)
                ]

            n_rows = len(df_model_filtered)

            st.markdown("**Variables used in the model**")
            feature_labels = [VARIABLE_LABELS.get(f, f) for f in features]
            st.write(feature_labels)
            
            filter_text = ""
            if state_selected != "Todos":
                filter_text += f" en {state_selected}"
            if city_selected != "Todas":
                filter_text += f", {city_selected}"
            
            st.write(f"Records used in the model (after filters{filter_text}): **{n_rows}**")

            if n_rows < 10:
                st.warning(
                    "The selected state has too few ZIP codes to train a robust clustering model. "
                    "Select another state or use 'Todos'."
                )
            else:
                # ================== ENTRENAMIENTO (CACHÉ) ==================
                if use_manual and k_manual is not None:
                    with st.spinner(f"Training K-Means model with k={k_manual}..."):
                        df_km, centroids_df = run_kmeans_manual(df_model_filtered, features, k_manual)
                    best_k = k_manual
                    best_score = None
                    sil_scores = {}
                else:
                    with st.spinner("Training K-Means model and calculating Silhouette..."):
                        df_km, centroids_df, sil_scores, best_k, best_score = run_kmeans_segmentacion(
                            df_model_filtered, features
                        )

                st.session_state["df_km"] = df_km

                if df_km is None or best_k is None:
                    st.error("Could not find a valid cluster configuration.")
                else:
                    # ================== RESULTADOS ==================

                    # Tabla de Silhouette por k (solo si se usó automático)
                    if sil_scores:
                        st.markdown("### Automatic selection of number of clusters (k - Silhouette Score)")
                        sil_df = (
                            pd.DataFrame(
                                [{"k": k, "silhouette": score} for k, score in sil_scores.items()]
                            )
                            .sort_values("k")
                            .reset_index(drop=True)
                        )
                        st.dataframe(sil_df, use_container_width=True)

                        st.success(
                            f"The optimal number of clusters according to the Silhouette coefficient is **k = {best_k}** "
                            f"(score = {best_score:.3f})."
                        )
                    else:
                        st.info(f"✅ Model trained with **k = {best_k}** clusters (manual selection).")

                # Centroides
                st.markdown("### Cluster centroids (in original scale)")
                st.dataframe(centroids_df, use_container_width=True)

                # Cluster Summary with Spanish percentage
                st.markdown("### Cluster Summary")
                cluster_summary = []
                for cluster_id in sorted(df_km["cluster"].unique()):
                    cluster_data = df_km[df_km["cluster"] == cluster_id]
                    
                    summary_row = {
                        "cluster": cluster_id,
                        "ZIP codes": len(cluster_data),
                        "Income medio": cluster_data["ingreso_medio"].mean(),
                        "Income min": cluster_data["ingreso_medio"].min(),
                        "Income max": cluster_data["ingreso_medio"].max(),
                        "% Mexican": cluster_data["pct_mexicana"].mean(),
                        "Ciudad principal": cluster_data["PHYSICAL_CITY"].mode()[0] if len(cluster_data["PHYSICAL_CITY"].mode()) > 0 else "N/A"
                    }
                    
                    if "spanish_home" in cluster_data.columns and cluster_data["spanish_home"].notna().sum() > 0:
                        total_households = cluster_data["pop_total"].sum()
                        spanish_households = cluster_data["spanish_home"].sum()
                        pct_spanish = (spanish_households / total_households * 100) if total_households > 0 else 0
                        summary_row["% Spanish Home"] = pct_spanish
                    
                    cluster_summary.append(summary_row)
                
                summary_df = pd.DataFrame(cluster_summary)
                st.dataframe(summary_df, use_container_width=True)

                # Distribution of ZIP codes by cluster
                st.markdown("### Distribution of ZIP codes by cluster")
                cluster_counts = df_km["cluster"].value_counts().sort_index()
                col_bar, _ = st.columns([2, 1])
                with col_bar:
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    cluster_counts.plot(kind="bar", ax=ax, color="steelblue")
                    ax.set_xlabel("Cluster")
                    ax.set_ylabel("Number of ZIP codes")
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
                st.markdown("### 🧠 Automatic model conclusions")

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
- Identified **{best_k} clusters** ZIP codes for {location_text}.
- The dataset used for this model contains **{n_rows} ZIP codes** después de aplicar los filtros.
- The centroids show how vary la **población total**, la **población mexicana** y el **ingreso medio**
  among the different segments.
- Clusters with higher median incomes son candidatos naturales para lanzar **SKUs premium**;
  los de ingreso bajo, para **líneas de entrada** or smaller formats.
- By combining this segmentation with market analysis (ventas por ZIP / retailer) podrás decidir
  en qué zonas enfocar the launches of new products.
"""
                )

# ========= TAB 4: MAPA GEOESPACIAL =========

with tab_geo:
    st.markdown("## 🗺️ Geospatial Map")
    st.write(
        "Visualizing ZIP codes segmented on an Uber-type map, "
        "using PyDeck. Each point represents a ZIP code; the color indicates the cluster."
    )

    # Ruta del GeoParquet que descargaste
    GEO_PARQUET_PATH = "georef-united-states-of-america-zc-point.parquet"  # ajusta el nombre si es distinto

    @st.cache_data
    def load_geo_points_optimized():
        """Loads GeoParquet with optimizations - reads only needed columns"""
        try:
            # Try to read just the columns we need
            gdf = pd.read_parquet(
                GEO_PARQUET_PATH,
                columns=["zip_code", "geo_point_2d"] if "geo_point_2d" in pd.read_parquet(GEO_PARQUET_PATH, columns=[0]).columns else None
            )
        except:
            gdf = pd.read_parquet(GEO_PARQUET_PATH)

        if "zip_code" not in gdf.columns:
            return None

        gdf["zip_code"] = gdf["zip_code"].astype(str).str.zfill(5)

        coord_col = "geo_point_2d" if "geo_point_2d" in gdf.columns else "Geo Point" if "Geo Point" in gdf.columns else None
        
        if coord_col is None:
            return None

        # Faster geometry parsing
        from shapely.wkb import loads as wkb_loads
        gdf[["lat", "lon"]] = gdf[coord_col].apply(
            lambda x: pd.Series([None, None]) if x is None else pd.Series(
                [wkb_loads(x).y, wkb_loads(x).x] if hasattr(x, '__len__') else [None, None]
            )
        )
        
        gdf = gdf.dropna(subset=["lat", "lon"])
        return gdf[["zip_code", "lat", "lon"]]

    with st.spinner("⏳ Loading geographic data..."):
        geo_df = load_geo_points_optimized()
        if geo_df is None:
            st.error("Could not load geographic data from GeoParquet")
            st.stop()

    # -------- Dataset del modelo con clusters --------
    if "df_km" in st.session_state and st.session_state["df_km"] is not None:
        df_model = st.session_state["df_km"].copy()
    elif "df_merged" in st.session_state:
        df_model = st.session_state["df_merged"].copy()
    else:
        st.error("No data prepared. Reload the page.")
        st.stop()

    # Aseguramos formato 5 dígitos también aquí
    df_model["zip_code"] = df_model["zip_code"].astype(str).str.zfill(5)

    # If for some reason you still do not have the column 'cluster', we create an emergency one
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

    # Density control
    st.markdown("#### Map visualization options")
    density_percent = st.slider(
        "Show % of ZIP codes (for performance):",
        min_value=10,
        max_value=100,
        value=75,
        step=10,
        help="Reduce for faster loading on slower connections"
    )
    
    n_sample = max(100, int(len(df_map) * density_percent / 100))
    if len(df_map) > n_sample:
        df_map_viz = df_map.sample(n=n_sample, random_state=42)
        st.info(f"📊 Showing {n_sample} of {len(df_map)} ZIP codes ({density_percent}%)")
    else:
        df_map_viz = df_map.copy()

    try:
        with st.spinner("🔄 Generating Voronoi map (may take 10-30 seconds)..."):
            features = generate_voronoi_polygons(df_map_viz)
        
        if not features:
            st.error("Could not generate Voronoi polygons. Try reducing the density percentage.")
        else:
            geojson_data = {
                "type": "FeatureCollection",
                "features": features
            }
            
            voronoi_layer = pdk.Layer(
                "GeoJsonLayer",
                data=geojson_data,
                stroked=True,
                filled=True,
                line_width_min_pixels=1,
                line_width_max_pixels=2,
                get_fill_color="properties.color",
                get_line_color=[255, 255, 255, 100],
                pickable=True,
            )
            
            deck = pdk.Deck(
                layers=[voronoi_layer],
                initial_view_state=pdk.ViewState(
                    latitude=midpoint[0],
                    longitude=midpoint[1],
                    zoom=3,
                    pitch=0,
                ),
                map_style="mapbox://styles/mapbox/dark-v10",
                tooltip={
                    "html": "{cluster_display}",
                    "style": {"backgroundColor": "black", "color": "white", "font-size": "16px", "padding": "10px", "border-radius": "5px"},
                },
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
    
    except Exception as e:
        st.error(f"Error rendering Voronoi map: {e}")
        logging.error(f"Voronoi map error: {e}")

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
    cluster_stats.columns = ["ZIP codes", "Income medio", "Income mín", "Income máx", "% Mexican", "Ciudad principal"]
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

# ========= TAB 5: ESTRATEGIA SKU =========

with tab_sku:
    st.markdown("## 🛍️ SKU Strategy Integrated with K-Means")
    st.write("SKU Tiers analysis crossed with K-Means Clusters for optimized launch strategy.")
    
    if "df_km" in st.session_state and st.session_state["df_km"] is not None:
        df_sku_base = st.session_state["df_km"].copy()
    elif "df_merged" in st.session_state:
        df_sku_base = st.session_state["df_merged"].copy()
    else:
        df_sku_base = prepare_merged_data()[0]
    
    if df_sku_base.empty:
        st.error("No data available")
        st.stop()
    
    # Clasificación SKU (vectorizada)
    df_sku_base["sku_tier"] = "Value"
    df_sku_base.loc[(df_sku_base["ingreso_medio"] >= 40000) & (df_sku_base["ingreso_medio"] < 60000), "sku_tier"] = "Mid-Value"
    df_sku_base.loc[(df_sku_base["ingreso_medio"] >= 60000) & (df_sku_base["ingreso_medio"] < 80000) & (df_sku_base["pct_mexicana"] >= 10), "sku_tier"] = "Mid-Market"
    df_sku_base.loc[(df_sku_base["ingreso_medio"] >= 80000) & (df_sku_base["pct_mexicana"] >= 15), "sku_tier"] = "Premium"
    
    # Si hay columna cluster de K-Means, usarla; si no, crear ficticios
    if "cluster" not in df_sku_base.columns or df_sku_base["cluster"].isna().all():
        st.info("ℹ️ Run el modelo K-Means primero to see Cluster × SKU analysis. Showing analysis by SKU Tier only.")
        has_clusters = False
    else:
        has_clusters = True
        df_sku_base["cluster"] = df_sku_base["cluster"].fillna(-1).astype(int)
    
    # ============ 0. CATÁLOGO DE SKUs POR TIER ============
    st.markdown("---")
    st.markdown("### 📦 0. SKU Catalog by Segment")
    
    sku_catalog = {
        "Premium": [
            {"name": "Organic Premium Hass Avocado", "size": "3 pack", "price": "$9.99", "format": "Premium box", "features": "USDA Organic, Pesticide-free, Direct import"},
            {"name": "Artisanal Oaxaca Cheese", "size": "16oz", "price": "$11.99", "format": "Premium bag", "features": "Fresh cheese, Traditional craftsmanship, Fair trade"},
            {"name": "Organic Poblano Peppers", "size": "2 lb", "price": "$12.99", "format": "Premium bag", "features": "Fresh harvest, Chemical-free, Small producers"}
        ],
        "Mid-Market": [
            {"name": "100% Nixtamalized Corn Tortillas", "size": "24 pack", "price": "$6.99", "format": "Family pack", "features": "Traditional recipe, Fresh, Preservative-free"},
            {"name": "Premium Black Beans", "size": "16oz", "price": "$7.49", "format": "Family bag", "features": "Ready-cooked, Consistent quality, Delicious"},
            {"name": "Homemade Red Salsa", "size": "15oz", "price": "$5.99", "format": "4-pack", "features": "Homemade recipe, Fresh tomatoes, Popular"}
        ],
        "Mid-Value": [
            {"name": "Long Grain White Rice", "size": "2 lb", "price": "$4.49", "format": "Bulk bag", "features": "Quality rice, Large volume, Affordable"},
            {"name": "Assorted Dried Peppers", "size": "8oz", "price": "$3.99", "format": "Large bag", "features": "Guajillo/ancho mix, Maximum value, Culinary use"},
            {"name": "Homemade Chicken Broth", "size": "14oz", "price": "$3.49", "format": "4-pack bundle", "features": "Concentrated broth, Daily use, Competitive price"}
        ],
        "Value": [
            {"name": "Store Brand Refried Beans", "size": "16oz", "price": "$2.99", "format": "Bulk 10-pack", "features": "Ready to eat, Best market price, Large volume"},
            {"name": "White Flour Tortillas", "size": "30 pack", "price": "$2.49", "format": "Family multipack", "features": "Large tortillas, Budget-friendly, Large size"},
            {"name": "Budget Green Salsa", "size": "16oz", "price": "$1.99", "format": "12x Multipack", "features": "Basic salsa, Minimum price, Maximum quantity"}
        ]
    }
    
    col_prem, col_mid, col_val, col_econ = st.columns(4)
    
    with col_prem:
        st.subheader("🟡 Premium")
        for sku in sku_catalog["Premium"]:
            st.write(f"**{sku['name']}**")
            st.caption(f"{sku['size']} | {sku['price']} | {sku['format']}\n{sku['features']}")
            st.divider()
    
    with col_mid:
        st.subheader("🟢 Mid-Market")
        for sku in sku_catalog["Mid-Market"]:
            st.write(f"**{sku['name']}**")
            st.caption(f"{sku['size']} | {sku['price']} | {sku['format']}\n{sku['features']}")
            st.divider()
    
    with col_val:
        st.subheader("🔵 Mid-Value")
        for sku in sku_catalog["Mid-Value"]:
            st.write(f"**{sku['name']}**")
            st.caption(f"{sku['size']} | {sku['price']} | {sku['format']}\n{sku['features']}")
            st.divider()
    
    with col_econ:
        st.subheader("🟣 Value")
        for sku in sku_catalog["Value"]:
            st.write(f"**{sku['name']}**")
            st.caption(f"{sku['size']} | {sku['price']} | {sku['format']}\n{sku['features']}")
            st.divider()
    
    # ============ 1. HEATMAP CLUSTER × SKU ============
    if has_clusters:
        st.markdown("---")
        st.markdown("### 🔥 1. Cluster × SKU Tier Matrix (Heatmap)")
        
        matriz_cluster_sku = pd.crosstab(
            df_sku_base["cluster"],
            df_sku_base["sku_tier"],
            margins=False
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(matriz_cluster_sku.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(matriz_cluster_sku.columns)))
        ax.set_yticks(range(len(matriz_cluster_sku.index)))
        ax.set_xticklabels(matriz_cluster_sku.columns, rotation=45)
        ax.set_yticklabels([f"Cluster {int(i)}" for i in matriz_cluster_sku.index])
        ax.set_ylabel("Cluster K-Means")
        ax.set_xlabel("SKU Tier")
        ax.set_title("Distribution of ZIP codes: Cluster × SKU Tier")
        
        # Añadir valores en celdas
        for i in range(len(matriz_cluster_sku.index)):
            for j in range(len(matriz_cluster_sku.columns)):
                val = matriz_cluster_sku.values[i, j]
                text = ax.text(j, i, f"{int(val)}", ha="center", va="center", color="black", fontweight="bold")
        
        plt.colorbar(im, ax=ax, label="# ZIP codes")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(matriz_cluster_sku, use_container_width=True)
    
    # ============ 2. RECOMENDACIONES POR CLUSTER ============
    if has_clusters:
        st.markdown("---")
        st.markdown("### 🎯 2. SKU Recommendation by K-Means Cluster")
        
        cluster_analysis_data = []
        
        for cluster_id in sorted(df_sku_base["cluster"].unique()):
            if cluster_id == -1:
                continue
            
            cluster_data = df_sku_base[df_sku_base["cluster"] == cluster_id]
            
            # SKU dominante en este cluster
            sku_dist = cluster_data["sku_tier"].value_counts()
            dominant_sku = sku_dist.index[0]
            dominant_pct = (sku_dist.iloc[0] / len(cluster_data) * 100)
            
            avg_income = cluster_data["ingreso_medio"].mean()
            avg_pct_mex = cluster_data["pct_mexicana"].mean()
            pop_mex = cluster_data["pop_mexicana"].sum()
            n_zips = len(cluster_data)
            
            cluster_analysis_data.append({
                "Cluster": cluster_id,
                "SKU Primary": dominant_sku,
                "% SKU": f"{dominant_pct:.0f}%",
                "Income Prom": f"${avg_income:,.0f}",
                "% Mexican": f"{avg_pct_mex:.1f}%",
                "Mexican Pop": f"{pop_mex:,.0f}",
                "# ZIPs": n_zips
            })
            
            top_cities_cluster = cluster_data.groupby("PHYSICAL_CITY")["pop_mexicana"].sum().nlargest(3)
            
            with st.expander(f"**Cluster {int(cluster_id)}** - PRIMARY: {dominant_sku} ({dominant_pct:.0f}%)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Income Prom", f"${avg_income:,.0f}")
                    st.metric("% Mexican", f"{avg_pct_mex:.1f}%")
                with col2:
                    st.metric("Mexican Pop", f"{pop_mex:,.0f}")
                    st.metric("# ZIP codes", n_zips)
                with col3:
                    st.metric("SKU Primary", dominant_sku)
                    st.metric("Dominance", f"{dominant_pct:.0f}%")
                
                st.markdown("**📍 Top 3 Cities:**")
                for city, pop in top_cities_cluster.items():
                    st.markdown(f"  - {city}: {pop:,.0f} mexicanos")
                
                st.markdown(f"\n**🎯 SKUs Recomendados para Cluster {int(cluster_id)}:**")
                tier_skus = sku_catalog.get(dominant_sku, [])
                for idx, sku in enumerate(tier_skus[:2], 1):
                    st.markdown(f"""
**{idx}. {sku['name']}**
- Tamaño: {sku['size']} | Precio: {sku['price']}
- Formato: {sku['format']}
- Características: {sku['features']}
                    """)
                
                st.markdown(f"**📊 Composición SKU en Cluster:**")
                sku_mix = cluster_data["sku_tier"].value_counts()
                for sku_tier, count in sku_mix.items():
                    pct = count/len(cluster_data)*100
                    st.markdown(f"  - **{sku_tier}**: {count} ZIPs ({pct:.0f}%)")
        
        st.markdown("---")
        st.markdown("#### Summary by Cluster")
        st.dataframe(pd.DataFrame(cluster_analysis_data), use_container_width=True)
    
    # ============ 3. TABLA COMPARATIVA: CLUSTER vs SKU PERFORMANCE ============
    st.markdown("---")
    st.markdown("### 📊 3. Comparative Table: Cluster × SKU Performance")
    
    if has_clusters:
        performance_data = []
        
        for cluster_id in sorted(df_sku_base[df_sku_base["cluster"] != -1]["cluster"].unique()):
            cluster_data = df_sku_base[df_sku_base["cluster"] == cluster_id]
            
            sku_dist = cluster_data["sku_tier"].value_counts()
            dominant_sku = sku_dist.index[0]
            
            avg_income = cluster_data["ingreso_medio"].mean()
            avg_pct_mex = cluster_data["pct_mexicana"].mean()
            pop_mex = cluster_data["pop_mexicana"].sum()
            
            # Penetración estimada basada en tier
            pen_map = {"Premium": 10, "Mid-Market": 15, "Mid-Value": 18, "Value": 20}
            penetration = pen_map.get(dominant_sku, 15)
            
            # ROI rough estimate
            roi_map = {"Premium": 45, "Mid-Market": 40, "Mid-Value": 50, "Value": 60}
            roi = roi_map.get(dominant_sku, 40)
            
            # Risk level
            risk_map = {"Premium": "Low", "Mid-Market": "Low", "Mid-Value": "Medium", "Value": "High"}
            risk = risk_map.get(dominant_sku, "Medium")
            
            # Priority (Premium first, then volume)
            priority_map = {"Premium": 1, "Mid-Market": 2, "Mid-Value": 2.5, "Value": 3}
            priority = priority_map.get(dominant_sku, 2)
            
            performance_data.append({
                "Cluster": f"C{int(cluster_id)}",
                "SKU Primary": dominant_sku,
                "# ZIPs": len(cluster_data),
                "Mexican Pop": f"{pop_mex:,.0f}",
                "Income Prom": f"${avg_income:,.0f}",
                "% Mexican": f"{avg_pct_mex:.1f}%",
                "Est. Penetration Y1": f"{penetration}%",
                "Est. ROI": f"{roi}%",
                "Risk": risk,
                "Priority": priority
            })
        
        perf_df = pd.DataFrame(performance_data).sort_values("Priority")
        st.dataframe(perf_df, use_container_width=True)
        
        st.markdown("**Legend:** Priority 1=Immediate Launch | Priority 2=Q2-Q3 | Priority 3=Q3-Q4")
        
        # ============ 4. TABLA DE RECOMENDACIÓN DE SKUs ESPECÍFICOS POR CLUSTER ============
        st.markdown("---")
        st.markdown("### 🛒 4. Specific SKU Recommendation by Cluster")
        
        sku_rec_data = []
        for cluster_id in sorted(df_sku_base[df_sku_base["cluster"] != -1]["cluster"].unique()):
            cluster_data = df_sku_base[df_sku_base["cluster"] == cluster_id]
            sku_dist = cluster_data["sku_tier"].value_counts()
            dominant_sku = sku_dist.index[0]
            
            tier_skus = sku_catalog.get(dominant_sku, [])
            if tier_skus:
                primary_sku = tier_skus[0]["name"]
                secondary_sku = tier_skus[1]["name"] if len(tier_skus) > 1 else tier_skus[0]["name"]
                primary_price = tier_skus[0]["price"]
            else:
                primary_sku = "N/A"
                secondary_sku = "N/A"
                primary_price = "N/A"
            
            sku_rec_data.append({
                "Cluster": f"C{int(cluster_id)}",
                "Tier": dominant_sku,
                "Primary SKU (Launch)": primary_sku,
                "Precio": primary_price,
                "Secondary SKU (Q2)": secondary_sku,
                "Target Market": "Incomes ${:,.0f}+, {} mexicanos".format(
                    int(cluster_data["ingreso_medio"].mean()),
                    "{:.0f}%".format(cluster_data["pct_mexicana"].mean())
                )
            })
        
        sku_rec_df = pd.DataFrame(sku_rec_data)
        st.dataframe(sku_rec_df, use_container_width=True)
        
        st.markdown("""
**Launch Strategy by Cluster:**
- **SKU Primario**: Lanzar primero en Q1 (Lanzamiento Inmediato)
- **SKU Secundario**: Lanzar en Q2 as line extension
- Adapt formats and sizes according to availability in retailers by ZIP code
        """)
    else:
        st.info("Run K-Means primero para ver tabla comparativa Cluster × SKU")


