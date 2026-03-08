"""
Simulador de Radar Geotécnico (GPR) — Interfaz Streamlit
=========================================================
Ejecutar con:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from radar_signal import SimuladorRadar, CapaSubsuelo

# ------------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Simulador GPR Geotécnico",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    h1, h2, h3, h4 { color: #a78bfa; }
    .block-container { padding-top: 1.5rem; }
    .stButton>button {
        background: #4f46e5; color: white; border: none;
        border-radius: 6px; font-weight: bold; width: 100%;
    }
    .stButton>button:hover { background: #6366f1; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Estado de sesión
# ------------------------------------------------------------------

def _init_state():
    if "anomalias" not in st.session_state:
        st.session_state.anomalias = []


_init_state()

# ------------------------------------------------------------------
# Sidebar — controles
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📡 Simulador GPR")
    st.markdown("---")

    st.markdown("### ⚙️ Parámetros del Radar")
    frecuencia = st.slider("Frecuencia central (MHz)", 25, 500, 100, step=5)
    ventana    = st.slider("Ventana de tiempo (ns)",  50, 500, 200, step=10)
    ganancia   = st.slider("Ganancia", 0.1, 5.0, 1.0, step=0.1)

    st.markdown("---")
    st.markdown("### 🪨 Capas del Subsuelo (εr)")

    eps_suelo  = st.slider("Suelo seco",      1.0, 20.0, 4.0,  step=0.5)
    eps_arcilla= st.slider("Arcilla húmeda",  1.0, 40.0, 10.0, step=0.5)
    eps_roca_f = st.slider("Roca fracturada", 1.0, 20.0, 6.0,  step=0.5)
    eps_roca_s = st.slider("Roca sólida",     1.0, 20.0, 8.0,  step=0.5)

    st.markdown("---")
    st.markdown("### ⚠️ Anomalías")

    tipo_anom = st.selectbox("Tipo", ["tuberia", "cavidad", "roca", "cable"])
    col_x, col_p = st.columns(2)
    with col_x:
        anom_x    = st.number_input("X (m)",    0.0, 30.0, 5.0, step=0.5)
    with col_p:
        anom_prof = st.number_input("Prof (m)", 0.1, 8.0,  2.0, step=0.1)

    col_add, col_clr = st.columns(2)
    with col_add:
        if st.button("＋ Agregar"):
            amps = {"tuberia": -0.8, "cavidad": 0.9, "roca": 0.5, "cable": -0.6}
            st.session_state.anomalias.append({
                "x": anom_x, "profundidad": anom_prof,
                "tipo": tipo_anom, "amplitud": amps[tipo_anom]
            })
    with col_clr:
        if st.button("✕ Limpiar"):
            st.session_state.anomalias = []

    if st.session_state.anomalias:
        for i, a in enumerate(st.session_state.anomalias):
            st.caption(f"  {i+1}. {a['tipo']} — x={a['x']}m, z={a['profundidad']}m")
    else:
        st.caption("Sin anomalías")

    st.markdown("---")
    st.markdown("### 📊 B-Scan")
    n_trazas   = st.slider("Número de trazas", 20, 200, 80, step=10)
    dist_total = st.slider("Distancia total (m)", 2.0, 30.0, 10.0, step=1.0)
    paleta     = st.selectbox("Paleta de color",
                               ["RdBu", "seismic_r", "Greys", "Viridis",
                                "Plasma", "Picnic"])

# ------------------------------------------------------------------
# Construir simulador con los parámetros actuales
# ------------------------------------------------------------------

sim = SimuladorRadar()
sim.frecuencia_central_mhz = frecuencia
sim.ventana_tiempo_ns       = ventana

sim.capas[1].permitividad = eps_suelo
sim.capas[2].permitividad = eps_arcilla
sim.capas[3].permitividad = eps_roca_f
sim.capas[4].permitividad = eps_roca_s

sim.anomalias = list(st.session_state.anomalias)

# Datos computados
t_ns   = sim.tiempos * 1e9
ascan  = sim.generar_ascan(dist_total / 2) * ganancia
bscan  = sim.generar_bscan(0, dist_total, n_trazas) * ganancia
xs_bs  = np.linspace(0, dist_total, n_trazas)

# ------------------------------------------------------------------
# Helpers de color
# ------------------------------------------------------------------

AZUL   = "#7c3aed"
ROJO   = "#ef4444"
AMBER  = "#f59e0b"
VERDE  = "#10b981"
FONDO  = "#12122a"
TEXTO  = "#e2e8f0"
GRILLA = "rgba(124,58,237,0.15)"


def _estilo_ax(fig, row, col):
    """Aplica tema oscuro a un subplot."""
    fig.update_xaxes(
        gridcolor=GRILLA, zerolinecolor=GRILLA,
        tickfont=dict(color=TEXTO, size=10),
        title_font=dict(color=TEXTO),
        row=row, col=col
    )
    fig.update_yaxes(
        gridcolor=GRILLA, zerolinecolor=GRILLA,
        tickfont=dict(color=TEXTO, size=10),
        title_font=dict(color=TEXTO),
        row=row, col=col
    )


# ------------------------------------------------------------------
# Título
# ------------------------------------------------------------------

st.markdown("# 📡 Simulador de Radar Geotécnico (GPR)")
st.markdown(
    f"**Frecuencia:** {frecuencia} MHz &nbsp;|&nbsp; "
    f"**Ventana:** {ventana} ns &nbsp;|&nbsp; "
    f"**Ganancia:** {ganancia}× &nbsp;|&nbsp; "
    f"**Anomalías:** {len(sim.anomalias)}"
)
st.markdown("---")

# ------------------------------------------------------------------
# Fila 1: A-Scan | B-Scan
# ------------------------------------------------------------------

col1, col2 = st.columns(2)

# ── A-Scan ──────────────────────────────────────────────────────────
with col1:
    st.markdown("#### A-Scan (traza central)")

    fig_a = go.Figure()

    # Relleno positivo
    fig_a.add_trace(go.Scatter(
        x=np.where(ascan > 0, ascan, 0), y=t_ns,
        fill="tozerox", fillcolor="rgba(124,58,237,0.25)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Relleno negativo
    fig_a.add_trace(go.Scatter(
        x=np.where(ascan < 0, ascan, 0), y=t_ns,
        fill="tozerox", fillcolor="rgba(239,68,68,0.25)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Línea principal
    fig_a.add_trace(go.Scatter(
        x=ascan, y=t_ns,
        mode="lines",
        line=dict(color=AZUL, width=1.5),
        name="A-Scan",
        hovertemplate="Amp: %{x:.3f}<br>t: %{y:.1f} ns"
    ))

    # Marcas de interfaces de capas
    for capa in sim.capas[1:]:
        t_ref = sim.profundidad_a_tiempo(capa.profundidad)
        if t_ref < ventana:
            fig_a.add_hline(
                y=t_ref,
                line=dict(color=capa.color, dash="dash", width=0.8),
                annotation_text=capa.nombre[:12],
                annotation_font=dict(color=TEXTO, size=9),
                annotation_position="right"
            )

    fig_a.update_layout(
        plot_bgcolor=FONDO, paper_bgcolor=FONDO,
        yaxis=dict(autorange="reversed", title="Tiempo (ns)",
                   gridcolor=GRILLA, tickfont=dict(color=TEXTO)),
        xaxis=dict(title="Amplitud", gridcolor=GRILLA,
                   tickfont=dict(color=TEXTO)),
        legend=dict(font=dict(color=TEXTO)),
        margin=dict(l=50, r=20, t=20, b=40),
        height=380,
    )
    st.plotly_chart(fig_a, use_container_width=True)

# ── B-Scan ──────────────────────────────────────────────────────────
with col2:
    st.markdown("#### B-Scan (Radargrama)")

    vmax = float(np.max(np.abs(bscan))) or 1.0

    fig_b = go.Figure(go.Heatmap(
        z=bscan,
        x=xs_bs,
        y=t_ns,
        zmin=-vmax, zmax=vmax,
        colorscale=paleta,
        colorbar=dict(
            title="Amp",
            tickfont=dict(color=TEXTO),
            titlefont=dict(color=TEXTO)
        ),
        hovertemplate="x: %{x:.2f} m<br>t: %{y:.1f} ns<br>Amp: %{z:.3f}"
    ))

    # Marcas de anomalías
    for a in sim.anomalias:
        t_ref = sim.profundidad_a_tiempo(a["profundidad"])
        fig_b.add_hline(y=t_ref,
                         line=dict(color=AMBER, dash="dot", width=1))
        fig_b.add_vline(x=a["x"],
                         line=dict(color=AMBER, dash="dot", width=1))
        fig_b.add_annotation(
            x=a["x"], y=t_ref,
            text=a["tipo"], font=dict(color=AMBER, size=10),
            showarrow=True, arrowcolor=AMBER, arrowsize=0.6
        )

    fig_b.update_layout(
        plot_bgcolor=FONDO, paper_bgcolor=FONDO,
        yaxis=dict(autorange="reversed", title="Tiempo (ns)",
                   gridcolor=GRILLA, tickfont=dict(color=TEXTO)),
        xaxis=dict(title="Posición (m)", gridcolor=GRILLA,
                   tickfont=dict(color=TEXTO)),
        margin=dict(l=50, r=20, t=20, b=40),
        height=380,
    )
    st.plotly_chart(fig_b, use_container_width=True)

# ------------------------------------------------------------------
# Fila 2: Modelo del subsuelo | Espectro
# ------------------------------------------------------------------

col3, col4 = st.columns(2)

# ── Modelo del subsuelo ─────────────────────────────────────────────
with col3:
    st.markdown("#### Modelo del Subsuelo")

    fig_m = go.Figure()
    prof_max = sim.capas[-1].profundidad + 2

    # Capas
    for i in range(1, len(sim.capas)):
        capa = sim.capas[i]
        y_sup = sim.capas[i - 1].profundidad
        y_inf = capa.profundidad if i < len(sim.capas) - 1 else prof_max

        fig_m.add_shape(type="rect",
            x0=0, x1=dist_total, y0=y_sup, y1=y_inf,
            fillcolor=capa.color, opacity=0.65,
            line=dict(width=0)
        )
        fig_m.add_annotation(
            x=0.3, y=(y_sup + y_inf) / 2,
            text=f"{capa.nombre} (εr={capa.permitividad:.1f})",
            showarrow=False,
            font=dict(color="white", size=10, family="monospace"),
            xanchor="left"
        )

    # Anomalías
    colores_an = {"tuberia": "#3b82f6", "cavidad": "#1e293b",
                  "roca": "#9ca3af", "cable": "#fbbf24"}
    for a in sim.anomalias:
        color_an = colores_an.get(a["tipo"], "#ef4444")
        fig_m.add_shape(type="circle",
            x0=a["x"] - 0.2, x1=a["x"] + 0.2,
            y0=a["profundidad"] - 0.2, y1=a["profundidad"] + 0.2,
            fillcolor=color_an, line=dict(color=AMBER, width=1.5)
        )
        fig_m.add_annotation(
            x=a["x"] + 0.3, y=a["profundidad"],
            text=a["tipo"], font=dict(color=AMBER, size=9),
            showarrow=False
        )

    # Línea de antena
    fig_m.add_hline(y=0, line=dict(color=VERDE, width=2.5),
                    annotation_text="Antena GPR",
                    annotation_font=dict(color=VERDE, size=10))

    fig_m.update_layout(
        plot_bgcolor=FONDO, paper_bgcolor=FONDO,
        xaxis=dict(title="Posición (m)", range=[0, dist_total],
                   gridcolor=GRILLA, tickfont=dict(color=TEXTO)),
        yaxis=dict(title="Profundidad (m)", autorange="reversed",
                   range=[0, prof_max],
                   gridcolor=GRILLA, tickfont=dict(color=TEXTO)),
        margin=dict(l=50, r=20, t=20, b=40),
        height=380,
    )
    st.plotly_chart(fig_m, use_container_width=True)

# ── Espectro de frecuencia ─────────────────────────────────────────
with col4:
    st.markdown("#### Espectro de Frecuencia")

    ascan_c = sim.generar_ascan()
    dt      = sim.tiempo_muestreo
    n       = len(ascan_c)
    freqs   = np.fft.rfftfreq(n, d=dt) / 1e6
    espectro = np.abs(np.fft.rfft(ascan_c))

    fig_e = go.Figure()

    fig_e.add_trace(go.Scatter(
        x=freqs, y=espectro,
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.25)",
        line=dict(color=AZUL, width=1.5),
        name="Espectro",
        hovertemplate="f: %{x:.1f} MHz<br>Amp: %{y:.4f}"
    ))

    fig_e.add_vline(
        x=frecuencia,
        line=dict(color=AMBER, dash="dash", width=1.5),
        annotation_text=f"fc = {frecuencia} MHz",
        annotation_font=dict(color=AMBER, size=10),
        annotation_position="top right"
    )

    fig_e.update_layout(
        plot_bgcolor=FONDO, paper_bgcolor=FONDO,
        xaxis=dict(
            title="Frecuencia (MHz)",
            range=[0, min(freqs[-1], frecuencia * 3)],
            gridcolor=GRILLA, tickfont=dict(color=TEXTO)
        ),
        yaxis=dict(title="Amplitud", gridcolor=GRILLA,
                   tickfont=dict(color=TEXTO)),
        legend=dict(font=dict(color=TEXTO)),
        margin=dict(l=50, r=20, t=20, b=40),
        height=380,
    )
    st.plotly_chart(fig_e, use_container_width=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------

st.markdown("---")
st.caption(
    "📡 Simulador de Radar Geotécnico (GPR) · "
    "Pulso Ricker · Modelo de capas con permitividad dieléctrica · "
    "Reflexiones por impedancia de interfaz"
)
