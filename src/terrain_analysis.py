"""
Terrain Analysis Engine for Geotechnical Radar Monitoring Simulator.

Handles:
- DEM (Digital Elevation Model) loading and processing
- Line-of-Sight (LOS) analysis between points
- Viewshed computation from a radar location
- Inverse viewshed: candidate radar locations that cover a target polygon
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import reproject, Resampling
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import unary_union
import geopandas as gpd
from scipy.ndimage import zoom
from typing import Tuple, List, Optional
import io


# ---------------------------------------------------------------------------
# DEM helpers
# ---------------------------------------------------------------------------

def load_dem_from_file(uploaded_file) -> Tuple[np.ndarray, object, object]:
    """
    Load a DEM from an uploaded file object (GeoTIFF or similar raster).
    Returns (elevation_array, transform, crs).
    """
    data = uploaded_file.read()
    with rasterio.open(io.BytesIO(data)) as src:
        dem = src.read(1).astype(np.float64)
        # Replace nodata with NaN
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs
    return dem, transform, crs


def resample_dem(dem: np.ndarray, factor: float) -> np.ndarray:
    """Resample DEM by a scale factor (factor < 1 = downsample)."""
    return zoom(dem, factor, order=1)


def dem_to_pixel(transform, lon: float, lat: float) -> Tuple[int, int]:
    """Convert geographic coordinates to pixel (row, col)."""
    row, col = rowcol(transform, lon, lat)
    return int(row), int(col)


def pixel_to_dem(transform, row: int, col: int) -> Tuple[float, float]:
    """Convert pixel (row, col) to geographic coordinates (x, y)."""
    x, y = xy(transform, row, col)
    return float(x), float(y)


def dem_pixel_size(transform) -> Tuple[float, float]:
    """Return pixel size in map units (dx, dy)."""
    dx = abs(transform.a)
    dy = abs(transform.e)
    return dx, dy


# ---------------------------------------------------------------------------
# Line-of-sight (LOS)
# ---------------------------------------------------------------------------

def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """Return pixel indices along the line from (r0,c0) to (r1,c1) using Bresenham."""
    pixels = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        pixels.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return pixels


def has_line_of_sight(
    dem: np.ndarray,
    transform,
    r_radar: int, c_radar: int, z_radar: float,
    r_target: int, c_target: int, z_target: float,
    radar_height: float = 3.0,
    target_height: float = 0.5,
) -> bool:
    """
    Check if there is line-of-sight between radar pixel and target pixel.

    radar_height: height of radar antenna above ground (m)
    target_height: height of the monitored point above ground (m)
    """
    dx, dy = dem_pixel_size(transform)
    pixel_dist = np.sqrt(dx**2 + dy**2)

    elev_radar = z_radar + radar_height
    elev_target = z_target + target_height

    pixels = bresenham_line(r_radar, c_radar, r_target, c_target)
    if len(pixels) < 2:
        return True

    nrows, ncols = dem.shape
    for i, (r, c) in enumerate(pixels[1:-1], start=1):
        if r < 0 or r >= nrows or c < 0 or c >= ncols:
            return False
        frac = i / (len(pixels) - 1)
        expected_elev = elev_radar + frac * (elev_target - elev_radar)
        terrain_elev = dem[r, c]
        if np.isnan(terrain_elev):
            return False
        if terrain_elev > expected_elev:
            return False
    return True


# ---------------------------------------------------------------------------
# Viewshed from a radar location
# ---------------------------------------------------------------------------

def compute_viewshed(
    dem: np.ndarray,
    transform,
    radar_row: int, radar_col: int,
    radar_height: float = 3.0,
    target_height: float = 0.5,
    max_range_m: float = 5000.0,
) -> np.ndarray:
    """
    Compute a binary viewshed array (1=visible, 0=not visible) from a radar position.

    Uses a radial sweep with elevation-angle tracking for efficiency.
    """
    nrows, ncols = dem.shape
    dx, dy = dem_pixel_size(transform)

    visible = np.zeros((nrows, ncols), dtype=np.uint8)
    z_radar = dem[radar_row, radar_col]
    if np.isnan(z_radar):
        return visible

    elev_radar = z_radar + radar_height
    max_range_pixels = int(max_range_m / min(dx, dy))

    # Cast rays at multiple azimuths
    num_azimuths = max(360 * 4, int(2 * np.pi * max_range_pixels))
    azimuths = np.linspace(0, 2 * np.pi, num_azimuths, endpoint=False)

    for az in azimuths:
        max_elev_angle = -np.inf
        sin_az = np.sin(az)
        cos_az = np.cos(az)

        for dist_px in range(1, max_range_pixels + 1):
            r = int(round(radar_row + dist_px * cos_az))
            c = int(round(radar_col + dist_px * sin_az))
            if r < 0 or r >= nrows or c < 0 or c >= ncols:
                break

            dist_m = dist_px * np.sqrt((dx * sin_az)**2 + (dy * cos_az)**2)
            terrain = dem[r, c]
            if np.isnan(terrain):
                break

            elev_angle = (terrain + target_height - elev_radar) / dist_m if dist_m > 0 else 0

            if elev_angle >= max_elev_angle:
                visible[r, c] = 1
                max_elev_angle = elev_angle

    visible[radar_row, radar_col] = 1
    return visible


# ---------------------------------------------------------------------------
# Inverse viewshed: find radar candidates that see a target polygon
# ---------------------------------------------------------------------------

def polygon_sample_pixels(
    polygon: Polygon,
    dem: np.ndarray,
    transform,
    n_samples: int = 30,
) -> List[Tuple[int, int]]:
    """
    Sample pixels inside a Shapely polygon (in map coordinates).
    Returns list of (row, col) tuples that are inside the polygon and have valid elevation.
    """
    minx, miny, maxx, maxy = polygon.bounds
    nrows, ncols = dem.shape

    samples = []
    # Grid sampling inside bounding box
    xs = np.linspace(minx, maxx, int(np.sqrt(n_samples)) + 2)
    ys = np.linspace(miny, maxy, int(np.sqrt(n_samples)) + 2)

    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if polygon.contains(pt):
                r, c = dem_to_pixel(transform, x, y)
                if 0 <= r < nrows and 0 <= c < ncols and not np.isnan(dem[r, c]):
                    samples.append((r, c))

    return samples if samples else []


def find_radar_candidates(
    dem: np.ndarray,
    transform,
    target_polygon: Polygon,
    candidate_mask: Optional[np.ndarray] = None,
    radar_height: float = 3.0,
    target_height: float = 0.5,
    max_range_m: float = 5000.0,
    coverage_threshold: float = 0.8,
    n_target_samples: int = 30,
    step: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find all grid positions that have LOS to at least `coverage_threshold` fraction
    of sample points in the target polygon.

    Returns:
        coverage_map : float array (nrows, ncols) - fraction of target visible
        candidate_mask_out : bool array (nrows, ncols) - positions above threshold
    """
    nrows, ncols = dem.shape
    dx, dy = dem_pixel_size(transform)
    max_range_px = max_range_m / min(dx, dy)

    target_pixels = polygon_sample_pixels(target_polygon, dem, transform, n_target_samples)
    if not target_pixels:
        return np.zeros((nrows, ncols)), np.zeros((nrows, ncols), dtype=bool)

    n_targets = len(target_pixels)
    coverage_map = np.zeros((nrows, ncols), dtype=np.float32)

    for r_radar in range(0, nrows, step):
        for c_radar in range(0, ncols, step):
            if candidate_mask is not None and not candidate_mask[r_radar, c_radar]:
                continue
            z_radar = dem[r_radar, c_radar]
            if np.isnan(z_radar):
                continue

            visible_count = 0
            for (r_t, c_t) in target_pixels:
                dist_px = np.sqrt((r_t - r_radar)**2 + (c_t - c_radar)**2)
                if dist_px > max_range_px:
                    continue
                z_t = dem[r_t, c_t]
                if np.isnan(z_t):
                    continue
                if has_line_of_sight(
                    dem, transform,
                    r_radar, c_radar, z_radar,
                    r_t, c_t, z_t,
                    radar_height, target_height
                ):
                    visible_count += 1

            coverage_map[r_radar, c_radar] = visible_count / n_targets

    candidate_mask_out = coverage_map >= coverage_threshold
    return coverage_map, candidate_mask_out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def viewshed_to_polygon(
    visible: np.ndarray,
    transform,
    simplify_tolerance: float = 10.0,
) -> Optional[Polygon]:
    """Convert a binary viewshed raster to a simplified Shapely polygon."""
    from rasterio.features import shapes
    import json

    # shapes() requires uint8
    mask = visible.astype(np.uint8)
    geoms = []
    for geom, val in shapes(mask, transform=transform):
        if val == 1:
            geoms.append(Polygon(geom["coordinates"][0]))

    if not geoms:
        return None

    merged = unary_union(geoms)
    if hasattr(merged, "simplify"):
        merged = merged.simplify(simplify_tolerance)
    return merged


def candidate_pixels_to_points(
    candidate_mask: np.ndarray,
    coverage_map: np.ndarray,
    transform,
    dem: np.ndarray,
    max_candidates: int = 50,
) -> gpd.GeoDataFrame:
    """
    Convert candidate radar pixel positions to a GeoDataFrame with metadata.
    Returns top candidates sorted by coverage score.
    """
    rows, cols = np.where(candidate_mask)
    if len(rows) == 0:
        return gpd.GeoDataFrame(columns=["geometry", "coverage", "elevation", "row", "col"])

    records = []
    for r, c in zip(rows, cols):
        x, y = pixel_to_dem(transform, r, c)
        elev = dem[r, c]
        cov = coverage_map[r, c]
        records.append({
            "geometry": Point(x, y),
            "coverage": float(cov),
            "elevation": float(elev),
            "row": int(r),
            "col": int(c),
        })

    gdf = gpd.GeoDataFrame(records)
    gdf = gdf.sort_values("coverage", ascending=False).head(max_candidates)
    gdf = gdf.reset_index(drop=True)
    return gdf
