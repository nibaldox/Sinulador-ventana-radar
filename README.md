# Simulador de Radar Geotécnico (GPR)

Simulador interactivo de Ground Penetrating Radar (GPR) para aplicaciones geotécnicas. Permite visualizar y explorar la propagación de ondas electromagnéticas en el subsuelo.

## Características

- **A-Scan**: Visualización de una traza de tiempo individual con reflexiones de capas y anomalías.
- **B-Scan (Radargrama)**: Vista panorámica horizontal que muestra el perfil completo del subsuelo.
- **Modelo del Subsuelo**: Representación visual de las capas geológicas y objetos enterrados.
- **Espectro de Frecuencia**: Análisis espectral de la señal emitida.
- **Anomalías interactivas**: Agrega tuberías, cavidades, rocas y cables al modelo.
- **Controles en tiempo real**: Ajusta frecuencia, ventana de tiempo, ganancia y paleta de colores.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

## Parámetros configurables

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| Frecuencia central | 25–500 MHz | Frecuencia dominante del pulso Ricker |
| Ventana de tiempo | 50–500 ns | Duración del registro de tiempo |
| Ganancia | 0.1–5× | Amplificación de la señal visualizada |
| Capas (εr) | 1–40 | Permitividad relativa de cada capa |
| Trazas B-scan | 20–200 | Resolución horizontal del radargrama |

## Tipos de anomalías

- **Tubería**: Reflexión negativa (objeto metálico o plástico vacío)
- **Cavidad**: Reflexión positiva fuerte (aire atrapado)
- **Roca**: Reflexión moderada positiva
- **Cable**: Reflexión negativa débil

## Estructura del proyecto

```
Sinulador-ventana-radar/
├── main.py              # Punto de entrada
├── requirements.txt     # Dependencias
├── README.md
└── src/
    ├── radar_signal.py  # Motor de simulación GPR
    └── gui.py           # Interfaz gráfica (Tkinter + Matplotlib)
```

## Física del modelo

El simulador usa el **pulso Ricker** (sombrero mexicano) como señal emitida. Los coeficientes de reflexión se calculan a partir de la impedancia característica de cada capa:

```
R = (Z₂ - Z₁) / (Z₂ + Z₁)
Z = √(μ₀ / εᵣ)
```

La velocidad de propagación en cada capa es:

```
v = c / √εᵣ   (m/ns)
```

donde `c ≈ 0.3 m/ns` es la velocidad de la luz.
