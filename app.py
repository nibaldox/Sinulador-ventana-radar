"""
Simulador de Áreas de Monitoreo de Radares Geotécnicos
=======================================================

Modos de operación:
  1. Simulación Directa  – Topografía + Polígono objetivo  → Posiciones candidatas del radar
  2. Simulación Inversa  – Topografía + Posición del radar → Área de visibilidad (viewshed)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import json
import io
import time

import rasterio
from shapely.geometry import shape, Polygon, Point

from terrain_analysis import (
    load_dem_from_file,
    resample_dem,
    dem_to_pixel,
    pixel_to_dem,
    compute_viewshed,
    find_radar_candidates,
    candidate_pixels_to_points,
    viewshed_to_polygon,
)
from visualization import (
    dem_surface_fig,
    dem_map_fig,
    viewshed_overlay_fig,
    viewshed_3d_fig,
    candidates_overlay_fig,
)
from sample_data import generate_sample_files, generate_synthetic_mine_dem, dem_to_geotiff_bytes

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Simulador Radar Geotécnico",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.8; font-size: 0.95rem; }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f3460;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>📡 Simulador de Áreas de Monitoreo – Radar Geotécnico</h1>
    <p>Análisis de línea de visión (LOS) y cuencas visuales sobre topografía de mina</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/radar.png", width=64)
    st.title("Configuración")

    mode = st.radio(
        "Modo de simulación",
        ["Simulación Directa", "Simulación Inversa"],
        help=(
            "**Directa**: dado el área a monitorear, busca dónde colocar el radar.\n\n"
            "**Inversa**: dado el punto del radar, calcula qué área es visible."
        ),
    )

    st.divider()
    st.subheader("Parámetros del radar")
    radar_height = st.slider(
        "Altura de la antena sobre el suelo (m)", 1.0, 15.0, 3.0, 0.5
    )
    target_height = st.slider(
        "Altura mínima del punto monitoreado (m)", 0.0, 5.0, 0.5, 0.5
    )
    max_range = st.slider(
        "Alcance máximo del radar (m)", 500, 10000, 3000, 250
    )

    st.divider()
    st.subheader("Rendimiento")
    resolution_factor = st.select_slider(
        "Factor de resolución del DEM",
        options=[0.25, 0.5, 1.0],
        value=0.5,
        help="Valores menores = cálculo más rápido, menor precisión.",
    )
    if mode == "Simulación Directa":
        coverage_threshold = st.slider(
            "Cobertura mínima requerida (%)", 50, 100, 80, 5
        ) / 100.0
        n_samples = st.slider("Puntos de muestreo en polígono objetivo", 10, 80, 30, 5)

    st.divider()
    use_sample_data = st.checkbox("Usar datos de ejemplo integrados", value=True)

# ---------------------------------------------------------------------------
# Data loading helpers (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_dem(file_bytes: bytes, factor: float):
    """Load and optionally resample a DEM from raw bytes."""
    buf = io.BytesIO(file_bytes)

    class FakeName:
        def read(self):
            return file_bytes

    with rasterio.open(io.BytesIO(file_bytes)) as src:
        dem = src.read(1).astype(np.float64)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs

    if factor != 1.0:
        from scipy.ndimage import zoom
        original_shape = dem.shape
        dem = zoom(dem, factor, order=1)
        # Scale the transform
        from rasterio.transform import from_bounds
        cols_orig = original_shape[1]
        rows_orig = original_shape[0]
        x_min = transform.c
        y_max = transform.f
        x_max = x_min + transform.a * cols_orig
        y_min = y_max + transform.e * rows_orig
        transform = from_bounds(x_min, y_min, x_max, y_max, dem.shape[1], dem.shape[0])

    return dem, transform, crs


@st.cache_data(show_spinner=False)
def _sample_dem_bytes():
    dem, transform, crs = generate_synthetic_mine_dem(width=200, height=200)
    return dem_to_geotiff_bytes(dem, transform, crs)


# ---------------------------------------------------------------------------
# Data loading UI
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown('<p class="section-title">1. Topografía (DEM)</p>', unsafe_allow_html=True)

    if use_sample_data:
        st.info("Usando DEM sintético de mina de ejemplo (200 × 200 px, 2 km × 2 km).")
        dem_bytes = _sample_dem_bytes()
    else:
        uploaded_dem = st.file_uploader(
            "Subir DEM (GeoTIFF)",
            type=["tif", "tiff"],
            help="Archivo GeoTIFF con el Modelo Digital de Elevación de la mina.",
        )
        if uploaded_dem:
            dem_bytes = uploaded_dem.read()
        else:
            dem_bytes = None

    if dem_bytes:
        dem, transform, crs = _load_dem(dem_bytes, resolution_factor)
        st.success(
            f"DEM cargado: {dem.shape[1]} × {dem.shape[0]} px  |  "
            f"Elev. min: {np.nanmin(dem):.0f} m  |  max: {np.nanmax(dem):.0f} m"
        )
    else:
        st.warning("Por favor sube un archivo GeoTIFF con la topografía.")
        st.stop()

# ---------------------------------------------------------------------------
# MODE: Simulación Inversa
# ---------------------------------------------------------------------------

if mode == "Simulación Inversa":
    with col_left:
        st.markdown('<p class="section-title">2. Ubicación del radar</p>', unsafe_allow_html=True)

        nrows, ncols = dem.shape
        cols_idx = np.arange(ncols)
        rows_idx = np.arange(nrows)
        from rasterio.transform import xy as raster_xy
        xs, _ = raster_xy(transform, np.zeros(ncols, dtype=int), cols_idx)
        _, ys = raster_xy(transform, rows_idx, np.zeros(nrows, dtype=int))
        x_min, x_max = float(min(xs)), float(max(xs))
        y_min, y_max = float(min(ys)), float(max(ys))
        x_mid = float(np.mean(xs))
        y_mid = float(np.mean(ys))

        radar_x = st.number_input("Coordenada Este (X) del radar", value=x_min + (x_max - x_min) * 0.75,
                                  min_value=x_min, max_value=x_max, step=10.0, format="%.1f")
        radar_y = st.number_input("Coordenada Norte (Y) del radar", value=y_min + (y_max - y_min) * 0.75,
                                  min_value=y_min, max_value=y_max, step=10.0, format="%.1f")

        run_inverse = st.button("Calcular área de visibilidad", type="primary", use_container_width=True)

    if run_inverse:
        r_radar, c_radar = dem_to_pixel(transform, radar_x, radar_y)
        r_radar = int(np.clip(r_radar, 0, dem.shape[0] - 1))
        c_radar = int(np.clip(c_radar, 0, dem.shape[1] - 1))

        with st.spinner("Calculando cuenca visual (viewshed)… esto puede tomar unos segundos."):
            t0 = time.time()
            viewshed = compute_viewshed(
                dem, transform,
                r_radar, c_radar,
                radar_height=radar_height,
                target_height=target_height,
                max_range_m=max_range,
            )
            elapsed = time.time() - t0

        visible_pct = viewshed.sum() / viewshed.size * 100
        visible_area_km2 = viewshed.sum() * abs(transform.a * transform.e) / 1e6

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Tiempo de cálculo", f"{elapsed:.1f} s")
        m2.metric("Área visible", f"{visible_area_km2:.3f} km²")
        m3.metric("% del DEM visible", f"{visible_pct:.1f} %")

        tab1, tab2, tab3 = st.tabs(["Mapa 2D", "Vista 3D", "Polígono de visibilidad"])

        with tab1:
            fig = viewshed_overlay_fig(dem, viewshed, transform,
                                       radar_xy=(radar_x, radar_y),
                                       title="Área monitoreable desde el radar (verde)")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            z_radar = float(dem[r_radar, c_radar])
            fig3d = viewshed_3d_fig(dem, viewshed, transform,
                                    radar_xy=(radar_x, radar_y),
                                    radar_elev=z_radar,
                                    title="Visibilidad 3D (verde = visible)")
            st.plotly_chart(fig3d, use_container_width=True)

        with tab3:
            with st.spinner("Generando polígono de visibilidad…"):
                dx = abs(transform.a)
                vis_poly = viewshed_to_polygon(viewshed, transform, simplify_tolerance=dx * 2)

            if vis_poly:
                geojson_vis = {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": vis_poly.__geo_interface__,
                        "properties": {
                            "area_km2": float(f"{visible_area_km2:.4f}"),
                            "visible_pct": float(f"{visible_pct:.2f}"),
                        },
                    }],
                }
                st.json(geojson_vis, expanded=False)
                st.download_button(
                    "Descargar área visible (GeoJSON)",
                    data=json.dumps(geojson_vis, indent=2),
                    file_name="area_visible_radar.geojson",
                    mime="application/json",
                )
            else:
                st.warning("No se pudo generar polígono de visibilidad.")

    else:
        with col_right:
            st.plotly_chart(dem_map_fig(dem, transform), use_container_width=True)

# ---------------------------------------------------------------------------
# MODE: Simulación Directa
# ---------------------------------------------------------------------------

else:  # Simulación Directa
    with col_left:
        st.markdown('<p class="section-title">2. Polígono del área a monitorear</p>', unsafe_allow_html=True)

        if use_sample_data:
            from sample_data import generate_target_polygon_geojson
            geojson_str = generate_target_polygon_geojson()
            st.info("Usando polígono de ejemplo (área dentro del pit de la mina).")
            polygon_geojson = json.loads(geojson_str)
        else:
            uploaded_poly = st.file_uploader(
                "Subir polígono objetivo (GeoJSON)",
                type=["geojson", "json"],
                help="Archivo GeoJSON con el polígono del área que desea monitorear.",
            )
            # Allow manual polygon input
            st.markdown("**O ingrese GeoJSON manualmente:**")
            manual_geojson = st.text_area(
                "GeoJSON del polígono",
                height=150,
                placeholder='{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[x1,y1],[x2,y2],...,[x1,y1]]]},"properties":{}}]}',
            )

            if uploaded_poly:
                polygon_geojson = json.loads(uploaded_poly.read())
            elif manual_geojson.strip():
                try:
                    polygon_geojson = json.loads(manual_geojson)
                except json.JSONDecodeError:
                    st.error("GeoJSON inválido. Verifique el formato.")
                    st.stop()
            else:
                polygon_geojson = None

        # Parse polygon
        target_polygon = None
        if polygon_geojson:
            try:
                if polygon_geojson.get("type") == "FeatureCollection":
                    geom = polygon_geojson["features"][0]["geometry"]
                elif polygon_geojson.get("type") == "Feature":
                    geom = polygon_geojson["geometry"]
                else:
                    geom = polygon_geojson
                target_polygon = shape(geom)
                st.success(f"Polígono cargado  |  Área: {target_polygon.area:.0f} m²")
            except Exception as e:
                st.error(f"Error al procesar polígono: {e}")
                target_polygon = None

        if target_polygon is None and not use_sample_data:
            st.warning("Por favor sube o ingresa el polígono del área a monitorear.")
            with col_right:
                st.plotly_chart(dem_map_fig(dem, transform), use_container_width=True)
            st.stop()

        st.divider()
        run_direct = st.button("Buscar ubicaciones del radar", type="primary", use_container_width=True)

    if run_direct and target_polygon is not None:
        # Compute step based on resolution factor
        step = max(1, int(8 / resolution_factor))

        with st.spinner(
            f"Analizando {dem.shape[0] * dem.shape[1] // (step**2):,} posiciones candidatas…"
        ):
            t0 = time.time()
            coverage_map, candidate_mask = find_radar_candidates(
                dem, transform,
                target_polygon=target_polygon,
                radar_height=radar_height,
                target_height=target_height,
                max_range_m=max_range,
                coverage_threshold=coverage_threshold,
                n_target_samples=n_samples,
                step=step,
            )
            elapsed = time.time() - t0

        n_candidates = int(candidate_mask.sum())
        best_coverage = float(coverage_map.max())
        candidates_gdf = candidate_pixels_to_points(candidate_mask, coverage_map, transform, dem)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tiempo de cálculo", f"{elapsed:.1f} s")
        m2.metric("Posiciones candidatas", f"{n_candidates:,}")
        m3.metric("Mejor cobertura", f"{best_coverage:.0%}")
        if len(candidates_gdf) > 0:
            best = candidates_gdf.iloc[0]
            m4.metric("Elevación mejor posición", f"{best['elevation']:.0f} m")

        tab1, tab2, tab3 = st.tabs(["Mapa de candidatos", "Topografía 3D", "Tabla de resultados"])

        with tab1:
            fig = candidates_overlay_fig(dem, coverage_map, transform,
                                         target_polygon, candidates_gdf)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig3d = dem_surface_fig(dem, transform,
                                    title="Topografía del terreno y polígono objetivo")
            st.plotly_chart(fig3d, use_container_width=True)

        with tab3:
            if len(candidates_gdf) > 0:
                display_df = candidates_gdf[["coverage", "elevation"]].copy()
                display_df.index = range(1, len(display_df) + 1)
                display_df.columns = ["Cobertura", "Elevación (m)"]
                display_df["Cobertura"] = display_df["Cobertura"].map(lambda x: f"{x:.0%}")
                display_df["Este (m)"] = candidates_gdf.geometry.x.values.round(1)
                display_df["Norte (m)"] = candidates_gdf.geometry.y.values.round(1)
                st.dataframe(display_df, use_container_width=True)

                # Download
                geojson_candidates = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": row.geometry.__geo_interface__,
                            "properties": {
                                "rank": i + 1,
                                "coverage_pct": round(row["coverage"] * 100, 1),
                                "elevation_m": round(row["elevation"], 1),
                            },
                        }
                        for i, row in candidates_gdf.iterrows()
                    ],
                }
                st.download_button(
                    "Descargar candidatos (GeoJSON)",
                    data=json.dumps(geojson_candidates, indent=2),
                    file_name="ubicaciones_radar_candidatas.geojson",
                    mime="application/json",
                )

                # Show best candidate viewshed automatically
                st.markdown("---")
                st.markdown("#### Visibilidad desde la mejor posición candidata")
                best = candidates_gdf.iloc[0]
                r_best, c_best = int(best["row"]), int(best["col"])
                with st.spinner("Calculando viewshed desde la mejor posición…"):
                    vs = compute_viewshed(
                        dem, transform, r_best, c_best,
                        radar_height=radar_height,
                        target_height=target_height,
                        max_range_m=max_range,
                    )
                fig_vs = viewshed_overlay_fig(
                    dem, vs, transform,
                    radar_xy=(best.geometry.x, best.geometry.y),
                    target_polygon=target_polygon,
                    title="Área visible desde la mejor posición del radar",
                )
                st.plotly_chart(fig_vs, use_container_width=True)
            else:
                st.warning(
                    "No se encontraron posiciones que cumplan el umbral de cobertura. "
                    "Intente reducir el umbral o aumentar el alcance del radar."
                )

    else:
        with col_right:
            fig = dem_map_fig(dem, transform)
            if target_polygon is not None:
                from visualization import _add_polygon_trace
                _add_polygon_trace(fig, target_polygon, "Área objetivo", "red")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <small style='color: gray;'>
    Simulador de Áreas de Monitoreo – Radar Geotécnico &nbsp;|&nbsp;
    Análisis LOS / Viewshed sobre DEM de mina &nbsp;|&nbsp;
    <em>Para uso en evaluación de posicionamiento de radares IBIS, GroundProbe, etc.</em>
    </small>
    """,
    unsafe_allow_html=True,
)
