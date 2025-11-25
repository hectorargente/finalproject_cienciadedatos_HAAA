# Setup Instructions for Go-to-Market Geoespacial App

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Environment Variables

Before running the app, set your Census API key:

```bash
export CENSUS_API_KEY="your_census_api_key_here"
```

Get a free API key from: https://api.census.gov/data/key_signup.html

## 3. Required Data Files

Ensure these files are in the same directory as `app.py`:

- **georef-united-states-of-america-zc-point.parquet** - ZIP code geographic points (33,121 records)
- **ZIP_Locale_Detail(ZIP_DETAIL).csv** - ZIP code details with city and state information

## 4. Run the App

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`

## 5. Features

- **Introducción**: Overview of the project
- **EDA Descriptiva**: Exploratory data analysis with filters and visualizations
- **Modelo K-Means**: Clustering analysis of ZIP codes by economic characteristics
- **Mapa Geoespacial**: Interactive map showing clusters with Uber-style visualization

## Troubleshooting

### Missing CENSUS_API_KEY
Error: `Missing CENSUS_API_KEY environment variable`
- Solution: Set the environment variable before running the app

### Missing data files
Error: `Archivo requerido no encontrado`
- Solution: Ensure both data files are in the same directory as app.py

### Shapely/PyArrow errors
- Solution: These are installed via requirements.txt. If still having issues, run:
  ```bash
  pip install --upgrade shapely pyarrow
  ```

### WKB parsing errors
- The geo_point_2d column contains WKB (Well-Known Binary) geometry that requires shapely
- This is handled automatically in load_geo_points()
