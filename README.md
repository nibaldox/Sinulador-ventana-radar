# Simulador de Áreas de Monitoreo – Radar Geotécnico 📡

Aplicación web para análisis de línea de visión (LOS) y cuencas visuales sobre topografía de mina, orientada al posicionamiento de radares geotécnicos (IBIS, GroundProbe, Reutech, etc.).

## Características

### Modo 1: Simulación Directa
- Sube la topografía de la mina (GeoTIFF/DEM)
- Sube o dibuja el polígono del área que deseas monitorear
- El sistema evalúa todas las posiciones del terreno y devuelve:
  - Mapa de cobertura (% del área objetivo visible)
  - Lista de posiciones candidatas ordenadas por cobertura
  - Viewshed automático desde la mejor posición

### Modo 2: Simulación Inversa
- Sube la topografía de la mina
- Ingresa las coordenadas del radar
- El sistema calcula:
  - El área total visible (cuenca visual / viewshed)
  - Vista 2D y 3D de la visibilidad
  - Polígono descargable del área visible

## Instalación

```bash
git clone <repo>
cd Sinulador-ventana-radar
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

## Formatos de entrada soportados

| Dato | Formato | Notas |
|------|---------|-------|
| Topografía (DEM) | GeoTIFF (`.tif`) | Proyectado en metros (UTM recomendado) |
| Área objetivo | GeoJSON (`.geojson`) | Polígono en el mismo sistema de coordenadas |
| Posición radar | Coordenadas X/Y | Ingreso manual en la interfaz |

## Parámetros configurables

| Parámetro | Descripción | Rango típico |
|-----------|-------------|--------------|
| Altura de antena | Altura del radar sobre el suelo | 1–10 m |
| Altura objetivo | Altura mínima del punto monitoreado | 0–5 m |
| Alcance máximo | Rango máximo del radar | 500–10.000 m |
| Factor de resolución | Submuestreo del DEM para velocidad | 0.25–1.0 |
| Cobertura mínima | % del área objetivo que debe ser visible | 50–100% |

## Estructura del proyecto

```
Sinulador-ventana-radar/
├── app.py                  # Aplicación Streamlit principal
├── requirements.txt        # Dependencias Python
├── src/
│   ├── terrain_analysis.py # Motor de análisis LOS y viewshed
│   ├── visualization.py    # Gráficos Plotly
│   └── sample_data.py      # Generador de datos sintéticos de ejemplo
└── data/
    └── samples/
        ├── mina_ejemplo.tif            # DEM sintético de mina open-pit
        └── poligono_objetivo.geojson   # Polígono de ejemplo
```

## Algoritmos implementados

- **LOS (Line of Sight)**: Trazado de rayos Bresenham con verificación de elevación a lo largo de cada rayo
- **Viewshed radial**: Barrido en múltiples azimuts con seguimiento del ángulo de elevación máximo
- **Viewshed inverso**: Evaluación de cobertura para cada posición candidata del terreno contra puntos muestreados en el polígono objetivo
