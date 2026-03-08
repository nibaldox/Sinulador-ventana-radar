"""
Sample data generator for testing the radar monitoring simulator.

Creates synthetic mine topography (DEM) and example polygons as GeoTIFF / GeoJSON files.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import json
import io
from typing import Tuple


def generate_synthetic_mine_dem(
    width: int = 200,
    height: int = 200,
    pit_center: Tuple[float, float] = (0.5, 0.5),
    pit_depth: float = 300.0,
    base_elevation: float = 1500.0,
    noise_level: float = 10.0,
    seed: int = 42,
) -> Tuple[np.ndarray, object, object]:
    """
    Generate a synthetic open-pit mine DEM.

    The terrain consists of:
    - A central pit (depression)
    - Surrounding benches / ramps
    - Waste dump mounds on the periphery
    - Random noise

    Returns (dem_array, rasterio_transform, CRS).
    """
    rng = np.random.default_rng(seed)

    # Coordinate grids (0..1)
    xs = np.linspace(0, 1, width)
    ys = np.linspace(0, 1, height)
    X, Y = np.meshgrid(xs, ys)

    cx, cy = pit_center
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # ---- Pit profile (paraboloid inside, flat outside) ----
    pit_radius = 0.35
    dem = np.where(
        dist < pit_radius,
        base_elevation - pit_depth * (1 - (dist / pit_radius)**2),
        base_elevation,
    )

    # ---- Benches (terraced steps in the pit wall) ----
    bench_height = 15.0
    bench_width = 0.03
    for i in range(1, 8):
        r = pit_radius - i * bench_width * 1.2
        if r <= 0:
            break
        mask = (dist > r) & (dist < r + bench_width)
        dem[mask] += bench_height * (1 - i / 8)

    # ---- Waste dump on the NE corner ----
    dump_cx, dump_cy = 0.82, 0.82
    dump_dist = np.sqrt((X - dump_cx)**2 + (Y - dump_cy)**2)
    dump_height = 80.0
    dump_radius = 0.15
    dump = np.where(
        dump_dist < dump_radius,
        dump_height * (1 - (dump_dist / dump_radius)**2),
        0.0,
    )
    dem += dump

    # ---- Haul road (cut through the east wall) ----
    road_mask = (X > 0.65) & (X < 0.75) & (Y > 0.3) & (Y < 0.7)
    dem[road_mask] = base_elevation - pit_depth * 0.1

    # ---- Random terrain noise ----
    dem += rng.normal(0, noise_level, dem.shape)

    # Smooth slightly
    from scipy.ndimage import gaussian_filter
    dem = gaussian_filter(dem, sigma=1.5)

    # ---- Raster metadata ----
    # Place the DEM at a fake UTM-like coordinate system (meters)
    # Extent: 2 km x 2 km centred at (500000, 7000000)
    x_min, x_max = 499000.0, 501000.0
    y_min, y_max = 6999000.0, 7001000.0

    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    crs = CRS.from_epsg(32719)  # WGS84 / UTM zone 19S (typical for Andes mines)

    return dem, transform, crs


def dem_to_geotiff_bytes(
    dem: np.ndarray,
    transform,
    crs,
) -> bytes:
    """Serialize a DEM to GeoTIFF bytes (in-memory)."""
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=dem.shape[0],
        width=dem.shape[1],
        count=1,
        dtype=dem.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(dem.astype(np.float32), 1)
    buf.seek(0)
    return buf.read()


def generate_target_polygon_geojson(
    center_x: float = 499700.0,
    center_y: float = 7000300.0,
    radius: float = 150.0,
    n_vertices: int = 8,
) -> str:
    """
    Generate a simple convex polygon GeoJSON representing the area to monitor.
    Coordinates are in the same projected CRS as the synthetic DEM.
    """
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    # Add some irregularity
    rng = np.random.default_rng(7)
    radii = radius + rng.uniform(-20, 20, n_vertices)

    coords = [
        [center_x + r * np.cos(a), center_y + r * np.sin(a)]
        for r, a in zip(radii, angles)
    ]
    coords.append(coords[0])  # close ring

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
                "properties": {"name": "Target monitoring area"},
            }
        ],
    }
    return json.dumps(geojson, indent=2)


def generate_sample_files():
    """Return (geotiff_bytes, geojson_str) for the built-in sample dataset."""
    dem, transform, crs = generate_synthetic_mine_dem()
    tiff_bytes = dem_to_geotiff_bytes(dem, transform, crs)
    geojson_str = generate_target_polygon_geojson()
    return tiff_bytes, geojson_str
