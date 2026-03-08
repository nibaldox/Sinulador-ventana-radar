"""
Visualization helpers for the radar monitoring simulator.
Produces Plotly figures for DEM, viewshed, and candidate radar locations.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely.geometry import Polygon, Point
from rasterio.transform import xy as raster_xy
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# DEM surface plot
# ---------------------------------------------------------------------------

def dem_surface_fig(
    dem: np.ndarray,
    transform,
    title: str = "Topografía del terreno",
    colorscale: str = "earth",
    z_exaggeration: float = 2.0,
) -> go.Figure:
    """3-D surface plot of a DEM."""
    nrows, ncols = dem.shape

    # Build coordinate arrays
    cols_idx = np.arange(ncols)
    rows_idx = np.arange(nrows)
    xs, _ = raster_xy(transform, np.zeros(ncols, dtype=int), cols_idx)
    _, ys = raster_xy(transform, rows_idx, np.zeros(nrows, dtype=int))

    X, Y = np.meshgrid(xs, ys)
    Z = dem * z_exaggeration

    fig = go.Figure(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=dem,
            colorscale=colorscale,
            colorbar=dict(title="Elevación (m)"),
            hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Elev: %{customdata:.1f} m<extra></extra>",
            customdata=dem,
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Este (m)",
            yaxis_title="Norte (m)",
            zaxis_title=f"Elevación × {z_exaggeration}",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=520,
    )
    return fig


# ---------------------------------------------------------------------------
# 2-D map plot (plan view)
# ---------------------------------------------------------------------------

def dem_map_fig(
    dem: np.ndarray,
    transform,
    title: str = "Vista en planta (topografía)",
) -> go.Figure:
    """2-D plan view of the DEM as a heatmap."""
    nrows, ncols = dem.shape
    cols_idx = np.arange(ncols)
    rows_idx = np.arange(nrows)
    xs, _ = raster_xy(transform, np.zeros(ncols, dtype=int), cols_idx)
    _, ys = raster_xy(transform, rows_idx, np.zeros(nrows, dtype=int))

    # Flip vertically because raster row 0 = top
    fig = go.Figure(
        go.Heatmap(
            z=np.flipud(dem),
            x=xs,
            y=np.flip(ys),
            colorscale="earth",
            colorbar=dict(title="Elev (m)"),
            hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Elev: %{z:.1f} m<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        yaxis_scaleanchor="x",
        margin=dict(l=0, r=0, b=40, t=40),
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# Viewshed overlay
# ---------------------------------------------------------------------------

def viewshed_overlay_fig(
    dem: np.ndarray,
    viewshed: np.ndarray,
    transform,
    radar_xy: Tuple[float, float],
    target_polygon: Optional[Polygon] = None,
    title: str = "Área de monitoreo visible desde el radar",
) -> go.Figure:
    """
    2-D plan view with DEM base + viewshed overlay + radar marker.
    """
    nrows, ncols = dem.shape
    cols_idx = np.arange(ncols)
    rows_idx = np.arange(nrows)
    xs, _ = raster_xy(transform, np.zeros(ncols, dtype=int), cols_idx)
    _, ys = raster_xy(transform, rows_idx, np.zeros(nrows, dtype=int))

    # Masked viewshed (NaN where not visible)
    vis_masked = np.where(viewshed == 1, 1.0, np.nan)

    fig = go.Figure()

    # DEM base
    fig.add_trace(go.Heatmap(
        z=np.flipud(dem),
        x=xs,
        y=np.flip(ys),
        colorscale="gray",
        showscale=False,
        opacity=0.7,
        name="DEM",
        hovertemplate="Elev: %{z:.1f} m<extra>Terreno</extra>",
    ))

    # Viewshed overlay
    fig.add_trace(go.Heatmap(
        z=np.flipud(vis_masked),
        x=xs,
        y=np.flip(ys),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,200,100,0.45)"]],
        showscale=False,
        name="Área visible",
        hoverinfo="skip",
    ))

    # Target polygon
    if target_polygon is not None:
        _add_polygon_trace(fig, target_polygon, name="Área objetivo", color="red")

    # Radar location
    fig.add_trace(go.Scatter(
        x=[radar_xy[0]],
        y=[radar_xy[1]],
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=16, color="yellow",
                    line=dict(color="black", width=2)),
        text=["RADAR"],
        textposition="top center",
        name="Radar",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        yaxis_scaleanchor="x",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=40, t=40),
        height=520,
    )
    return fig


# ---------------------------------------------------------------------------
# Candidate radar locations overlay
# ---------------------------------------------------------------------------

def candidates_overlay_fig(
    dem: np.ndarray,
    coverage_map: np.ndarray,
    transform,
    target_polygon: Polygon,
    candidates_gdf: gpd.GeoDataFrame,
    title: str = "Posiciones candidatas para el radar",
) -> go.Figure:
    """
    2-D plan view with DEM, coverage heatmap, target polygon, and
    candidate radar positions colour-coded by coverage score.
    """
    nrows, ncols = dem.shape
    cols_idx = np.arange(ncols)
    rows_idx = np.arange(nrows)
    xs, _ = raster_xy(transform, np.zeros(ncols, dtype=int), cols_idx)
    _, ys = raster_xy(transform, rows_idx, np.zeros(nrows, dtype=int))

    # Coverage map (resample to full grid – map at step resolution)
    cov_masked = np.where(coverage_map > 0, coverage_map, np.nan)

    fig = go.Figure()

    # DEM base
    fig.add_trace(go.Heatmap(
        z=np.flipud(dem),
        x=xs,
        y=np.flip(ys),
        colorscale="gray",
        showscale=False,
        opacity=0.65,
        name="DEM",
        hoverinfo="skip",
    ))

    # Coverage map overlay
    fig.add_trace(go.Heatmap(
        z=np.flipud(cov_masked),
        x=xs,
        y=np.flip(ys),
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        opacity=0.55,
        colorbar=dict(title="Cobertura"),
        name="Cobertura",
        hovertemplate="Cobertura: %{z:.0%}<extra></extra>",
    ))

    # Target polygon
    _add_polygon_trace(fig, target_polygon, name="Área objetivo", color="blue")

    # Candidate points
    if len(candidates_gdf) > 0:
        fig.add_trace(go.Scatter(
            x=candidates_gdf.geometry.x,
            y=candidates_gdf.geometry.y,
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=10,
                color=candidates_gdf["coverage"],
                colorscale="Viridis",
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(title="Cob. radar", x=1.12),
                line=dict(color="black", width=1),
            ),
            text=[
                f"Elev: {row['elevation']:.1f} m<br>Cob: {row['coverage']:.0%}"
                for _, row in candidates_gdf.iterrows()
            ],
            hovertemplate="%{text}<extra>Candidato</extra>",
            name="Candidatos radar",
        ))

        # Highlight best candidate
        best = candidates_gdf.iloc[0]
        fig.add_trace(go.Scatter(
            x=[best.geometry.x],
            y=[best.geometry.y],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="gold",
                        line=dict(color="black", width=2)),
            text=["MEJOR"],
            textposition="top center",
            name="Mejor posición",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Este (m)",
        yaxis_title="Norte (m)",
        yaxis_scaleanchor="x",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=40, t=40),
        height=540,
    )
    return fig


# ---------------------------------------------------------------------------
# 3-D viewshed surface
# ---------------------------------------------------------------------------

def viewshed_3d_fig(
    dem: np.ndarray,
    viewshed: np.ndarray,
    transform,
    radar_xy: Tuple[float, float],
    radar_elev: float,
    z_exaggeration: float = 2.0,
    title: str = "Visibilidad 3D desde el radar",
) -> go.Figure:
    """3-D surface coloured by visibility."""
    nrows, ncols = dem.shape
    cols_idx = np.arange(ncols)
    rows_idx = np.arange(nrows)
    xs, _ = raster_xy(transform, np.zeros(ncols, dtype=int), cols_idx)
    _, ys = raster_xy(transform, rows_idx, np.zeros(nrows, dtype=int))
    X, Y = np.meshgrid(xs, ys)
    Z = dem * z_exaggeration

    fig = go.Figure(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=viewshed.astype(float),
            colorscale=[[0, "#555"], [0.5, "#888"], [1, "#00e676"]],
            cmin=0, cmax=1,
            showscale=False,
            opacity=0.9,
            hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Elev: %{customdata:.1f} m<extra></extra>",
            customdata=dem,
        )
    )

    # Radar marker (cone)
    fig.add_trace(go.Scatter3d(
        x=[radar_xy[0]],
        y=[radar_xy[1]],
        z=[radar_elev * z_exaggeration + 30],
        mode="markers+text",
        marker=dict(symbol="diamond", size=8, color="yellow"),
        text=["RADAR"],
        textposition="top center",
        name="Radar",
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Este (m)",
            yaxis_title="Norte (m)",
            zaxis_title=f"Elev × {z_exaggeration}",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=520,
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_polygon_trace(fig: go.Figure, polygon: Polygon, name: str, color: str):
    """Add a Shapely polygon outline to a Plotly figure."""
    if polygon.geom_type == "Polygon":
        polys = [polygon]
    else:
        polys = list(polygon.geoms)

    for poly in polys:
        x, y = poly.exterior.xy
        fig.add_trace(go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            line=dict(color=color, width=3),
            fill="toself",
            fillcolor=color.replace(")", ", 0.12)").replace("rgb", "rgba") if color.startswith("rgb") else color,
            opacity=0.25,
            name=name,
        ))
