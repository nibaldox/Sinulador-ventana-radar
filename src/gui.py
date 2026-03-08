"""
Interfaz gráfica del simulador de radar geotécnico (GPR).
Ventana principal con visualización de A-scan, B-scan y modelo del subsuelo.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

from radar_signal import SimuladorRadar, CapaSubsuelo


class VentanaRadar(tk.Tk):
    """Ventana principal del simulador de radar geotécnico."""

    TITULO = "Simulador de Radar Geotécnico (GPR)"
    COLOR_FONDO = "#1e1e2e"
    COLOR_PANEL = "#2a2a3e"
    COLOR_ACENTO = "#7c3aed"
    COLOR_TEXTO = "#e2e8f0"
    COLOR_BOTON = "#4f46e5"
    COLOR_BOTON_HOVER = "#6366f1"
    COLOR_EXITO = "#10b981"
    COLOR_ADVERTENCIA = "#f59e0b"

    def __init__(self):
        super().__init__()
        self.simulador = SimuladorRadar()
        self._configurar_ventana()
        self._crear_widgets()
        self._actualizar_simulacion()

    # ------------------------------------------------------------------
    # Configuración inicial
    # ------------------------------------------------------------------

    def _configurar_ventana(self):
        self.title(self.TITULO)
        self.geometry("1400x850")
        self.minsize(1000, 650)
        self.configure(bg=self.COLOR_FONDO)
        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)

        # Estilos ttk
        estilo = ttk.Style(self)
        estilo.theme_use("clam")
        estilo.configure("TFrame", background=self.COLOR_FONDO)
        estilo.configure("Panel.TFrame", background=self.COLOR_PANEL)
        estilo.configure("TLabel", background=self.COLOR_PANEL,
                          foreground=self.COLOR_TEXTO, font=("Segoe UI", 10))
        estilo.configure("Titulo.TLabel", background=self.COLOR_PANEL,
                          foreground=self.COLOR_TEXTO, font=("Segoe UI", 11, "bold"))
        estilo.configure("TScale", background=self.COLOR_PANEL,
                          troughcolor="#3a3a5e")
        estilo.configure("TCombobox", fieldbackground=self.COLOR_PANEL,
                          background=self.COLOR_PANEL, foreground=self.COLOR_TEXTO)
        estilo.configure("TCheckbutton", background=self.COLOR_PANEL,
                          foreground=self.COLOR_TEXTO)

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------

    def _crear_widgets(self):
        # Barra de título
        self._crear_barra_titulo()

        # Contenedor principal
        contenedor = ttk.Frame(self, style="TFrame")
        contenedor.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Panel izquierdo (controles)
        self.panel_ctrl = ttk.Frame(contenedor, style="Panel.TFrame", width=280)
        self.panel_ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        self.panel_ctrl.pack_propagate(False)
        self._crear_panel_controles(self.panel_ctrl)

        # Panel derecho (gráficas)
        self.panel_grafs = ttk.Frame(contenedor, style="TFrame")
        self.panel_grafs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._crear_panel_graficas(self.panel_grafs)

    def _crear_barra_titulo(self):
        barra = tk.Frame(self, bg=self.COLOR_ACENTO, height=40)
        barra.pack(fill=tk.X)
        barra.pack_propagate(False)

        tk.Label(barra, text="  ◈  Simulador de Radar Geotécnico (GPR)",
                 bg=self.COLOR_ACENTO, fg="white",
                 font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=10)

        tk.Label(barra, text="v1.0",
                 bg=self.COLOR_ACENTO, fg="#c4b5fd",
                 font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=10)

    def _crear_panel_controles(self, padre):
        canvas_ctrl = tk.Canvas(padre, bg=self.COLOR_PANEL,
                                highlightthickness=0)
        scrollbar = ttk.Scrollbar(padre, orient="vertical",
                                   command=canvas_ctrl.yview)
        frame_scroll = ttk.Frame(canvas_ctrl, style="Panel.TFrame")

        frame_scroll.bind("<Configure>",
            lambda e: canvas_ctrl.configure(
                scrollregion=canvas_ctrl.bbox("all")))

        canvas_ctrl.create_window((0, 0), window=frame_scroll, anchor="nw")
        canvas_ctrl.configure(yscrollcommand=scrollbar.set)

        canvas_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._crear_seccion_radar(frame_scroll)
        self._crear_seccion_capas(frame_scroll)
        self._crear_seccion_anomalias(frame_scroll)
        self._crear_seccion_bscan(frame_scroll)
        self._crear_botones_accion(frame_scroll)

    def _separador(self, padre):
        tk.Frame(padre, bg="#444466", height=1).pack(
            fill=tk.X, padx=8, pady=6)

    def _titulo_seccion(self, padre, texto):
        tk.Label(padre, text=texto, bg=self.COLOR_PANEL,
                 fg=self.COLOR_ACENTO,
                 font=("Segoe UI", 10, "bold")).pack(
                     anchor=tk.W, padx=10, pady=(8, 2))

    def _crear_seccion_radar(self, padre):
        self._titulo_seccion(padre, "⚙  Parámetros del Radar")

        # Frecuencia
        ttk.Label(padre, text="Frecuencia central (MHz):").pack(
            anchor=tk.W, padx=12)
        self.var_frecuencia = tk.DoubleVar(value=100.0)
        marco_frec = ttk.Frame(padre, style="Panel.TFrame")
        marco_frec.pack(fill=tk.X, padx=12, pady=2)
        ttk.Scale(marco_frec, from_=25, to=500,
                  variable=self.var_frecuencia,
                  command=self._al_cambiar_param).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_frec = ttk.Label(marco_frec, text="100 MHz", width=10)
        self.lbl_frec.pack(side=tk.RIGHT)

        # Ventana de tiempo
        ttk.Label(padre, text="Ventana de tiempo (ns):").pack(
            anchor=tk.W, padx=12, pady=(6, 0))
        self.var_ventana = tk.DoubleVar(value=200.0)
        marco_vent = ttk.Frame(padre, style="Panel.TFrame")
        marco_vent.pack(fill=tk.X, padx=12, pady=2)
        ttk.Scale(marco_vent, from_=50, to=500,
                  variable=self.var_ventana,
                  command=self._al_cambiar_param).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_ventana = ttk.Label(marco_vent, text="200 ns", width=10)
        self.lbl_ventana.pack(side=tk.RIGHT)

        # Ganancia
        ttk.Label(padre, text="Ganancia:").pack(
            anchor=tk.W, padx=12, pady=(6, 0))
        self.var_ganancia = tk.DoubleVar(value=1.0)
        marco_gan = ttk.Frame(padre, style="Panel.TFrame")
        marco_gan.pack(fill=tk.X, padx=12, pady=2)
        ttk.Scale(marco_gan, from_=0.1, to=5.0,
                  variable=self.var_ganancia,
                  command=self._al_cambiar_param).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_ganancia = ttk.Label(marco_gan, text="1.0x", width=10)
        self.lbl_ganancia.pack(side=tk.RIGHT)

        self._separador(padre)

    def _crear_seccion_capas(self, padre):
        self._titulo_seccion(padre, "🪨  Capas del Subsuelo")

        self.vars_capas = []
        for i, capa in enumerate(self.simulador.capas[1:], 1):
            marco = ttk.Frame(padre, style="Panel.TFrame")
            marco.pack(fill=tk.X, padx=10, pady=2)

            # Indicador de color
            tk.Frame(marco, bg=capa.color, width=12, height=12).pack(
                side=tk.LEFT, padx=(0, 4))

            ttk.Label(marco, text=capa.nombre[:18],
                      width=18).pack(side=tk.LEFT)

            # Permitividad
            var_eps = tk.DoubleVar(value=capa.permitividad)
            ttk.Label(marco, text="εr:", width=3).pack(side=tk.LEFT)
            spin = ttk.Spinbox(marco, from_=1, to=40, increment=0.5,
                               textvariable=var_eps, width=5,
                               command=self._actualizar_simulacion)
            spin.pack(side=tk.LEFT)
            self.vars_capas.append((i, var_eps))

        self._separador(padre)

    def _crear_seccion_anomalias(self, padre):
        self._titulo_seccion(padre, "⚠  Anomalías / Objetos Enterrados")

        # Tipo de anomalía
        ttk.Label(padre, text="Tipo:").pack(anchor=tk.W, padx=12)
        self.var_tipo_anomalia = tk.StringVar(value="tuberia")
        combo_tipo = ttk.Combobox(padre, textvariable=self.var_tipo_anomalia,
                                   values=["tuberia", "cavidad", "roca", "cable"],
                                   state="readonly", width=20)
        combo_tipo.pack(anchor=tk.W, padx=12, pady=2)

        # Posición X
        ttk.Label(padre, text="Posición X (m):").pack(
            anchor=tk.W, padx=12, pady=(4, 0))
        self.var_anomalia_x = tk.DoubleVar(value=5.0)
        ttk.Scale(padre, from_=0, to=10, variable=self.var_anomalia_x
                  ).pack(fill=tk.X, padx=12, pady=2)

        # Profundidad
        ttk.Label(padre, text="Profundidad (m):").pack(
            anchor=tk.W, padx=12, pady=(4, 0))
        self.var_anomalia_prof = tk.DoubleVar(value=2.0)
        ttk.Scale(padre, from_=0.1, to=6, variable=self.var_anomalia_prof
                  ).pack(fill=tk.X, padx=12, pady=2)

        # Botones anomalías
        marco_btn = ttk.Frame(padre, style="Panel.TFrame")
        marco_btn.pack(fill=tk.X, padx=10, pady=4)
        self._boton(marco_btn, "+ Agregar", self._agregar_anomalia,
                    self.COLOR_EXITO).pack(side=tk.LEFT, padx=2)
        self._boton(marco_btn, "✕ Limpiar", self._limpiar_anomalias,
                    "#ef4444").pack(side=tk.LEFT, padx=2)

        self.lbl_anomalias = ttk.Label(padre, text="Sin anomalías")
        self.lbl_anomalias.pack(anchor=tk.W, padx=12, pady=2)

        self._separador(padre)

    def _crear_seccion_bscan(self, padre):
        self._titulo_seccion(padre, "📡  B-Scan (Radargrama)")

        ttk.Label(padre, text="Número de trazas:").pack(
            anchor=tk.W, padx=12)
        self.var_n_trazas = tk.IntVar(value=80)
        ttk.Scale(padre, from_=20, to=200, variable=self.var_n_trazas
                  ).pack(fill=tk.X, padx=12, pady=2)

        ttk.Label(padre, text="Distancia total (m):").pack(
            anchor=tk.W, padx=12, pady=(6, 0))
        self.var_dist_total = tk.DoubleVar(value=10.0)
        ttk.Scale(padre, from_=2, to=30, variable=self.var_dist_total
                  ).pack(fill=tk.X, padx=12, pady=2)

        # Mapa de colores
        ttk.Label(padre, text="Paleta de colores:").pack(
            anchor=tk.W, padx=12, pady=(6, 0))
        self.var_cmap = tk.StringVar(value="seismic")
        combo_cmap = ttk.Combobox(padre, textvariable=self.var_cmap,
                                   values=["seismic", "RdBu", "bwr",
                                           "gray", "viridis", "plasma"],
                                   state="readonly", width=20)
        combo_cmap.pack(anchor=tk.W, padx=12, pady=2)
        combo_cmap.bind("<<ComboboxSelected>>",
                         lambda e: self._actualizar_simulacion())

        self._separador(padre)

    def _crear_botones_accion(self, padre):
        self._titulo_seccion(padre, "▶  Acciones")

        self._boton(padre, "▶  Simular", self._actualizar_simulacion,
                    self.COLOR_BOTON).pack(fill=tk.X, padx=12, pady=3)

        self._boton(padre, "↺  Restablecer", self._restablecer,
                    "#6b7280").pack(fill=tk.X, padx=12, pady=3)

        self._boton(padre, "💾  Guardar imagen", self._guardar_imagen,
                    self.COLOR_ADVERTENCIA).pack(fill=tk.X, padx=12, pady=3)

    def _boton(self, padre, texto, comando, color):
        btn = tk.Button(padre, text=texto, command=comando,
                        bg=color, fg="white",
                        font=("Segoe UI", 9, "bold"),
                        relief=tk.FLAT, cursor="hand2",
                        activebackground=self.COLOR_BOTON_HOVER,
                        activeforeground="white",
                        padx=6, pady=4)
        return btn

    # ------------------------------------------------------------------
    # Panel de gráficas
    # ------------------------------------------------------------------

    def _crear_panel_graficas(self, padre):
        self.fig = Figure(figsize=(10, 7), facecolor=self.COLOR_FONDO)
        gs = GridSpec(2, 2, figure=self.fig, hspace=0.4, wspace=0.35)

        self.ax_ascan = self.fig.add_subplot(gs[0, 0])
        self.ax_bscan = self.fig.add_subplot(gs[0, 1])
        self.ax_modelo = self.fig.add_subplot(gs[1, 0])
        self.ax_espectro = self.fig.add_subplot(gs[1, 1])

        for ax in [self.ax_ascan, self.ax_bscan,
                   self.ax_modelo, self.ax_espectro]:
            ax.set_facecolor("#12122a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
            ax.tick_params(colors=self.COLOR_TEXTO, labelsize=8)
            ax.xaxis.label.set_color(self.COLOR_TEXTO)
            ax.yaxis.label.set_color(self.COLOR_TEXTO)
            ax.title.set_color(self.COLOR_TEXTO)

        self.canvas = FigureCanvasTkAgg(self.fig, master=padre)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(padre, bg=self.COLOR_PANEL)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(bg=self.COLOR_PANEL)
        toolbar.update()

    # ------------------------------------------------------------------
    # Lógica de simulación y actualización
    # ------------------------------------------------------------------

    def _al_cambiar_param(self, _=None):
        self.lbl_frec.config(text=f"{self.var_frecuencia.get():.0f} MHz")
        self.lbl_ventana.config(text=f"{self.var_ventana.get():.0f} ns")
        self.lbl_ganancia.config(text=f"{self.var_ganancia.get():.1f}x")
        self._actualizar_simulacion()

    def _actualizar_simulacion(self, _=None):
        # Actualizar parámetros del simulador
        self.simulador.frecuencia_central_mhz = self.var_frecuencia.get()
        self.simulador.ventana_tiempo_ns = self.var_ventana.get()

        # Actualizar permitividades de capas
        for idx, var_eps in self.vars_capas:
            self.simulador.capas[idx].permitividad = var_eps.get()

        self._dibujar_ascan()
        self._dibujar_bscan()
        self._dibujar_modelo()
        self._dibujar_espectro()
        self.canvas.draw_idle()

    def _dibujar_ascan(self):
        ax = self.ax_ascan
        ax.clear()
        ax.set_facecolor("#12122a")

        ganancia = self.var_ganancia.get()
        ascan = self.simulador.generar_ascan(posicion=self.var_dist_total.get() / 2)
        t_ns = self.simulador.tiempos * 1e9

        ax.plot(ascan * ganancia, t_ns, color="#7c3aed", linewidth=1.2)
        ax.fill_betweenx(t_ns, ascan * ganancia, 0,
                         where=(ascan * ganancia > 0),
                         color="#7c3aed", alpha=0.25)
        ax.fill_betweenx(t_ns, ascan * ganancia, 0,
                         where=(ascan * ganancia < 0),
                         color="#ef4444", alpha=0.25)

        ax.set_xlabel("Amplitud", color=self.COLOR_TEXTO, fontsize=8)
        ax.set_ylabel("Tiempo (ns)", color=self.COLOR_TEXTO, fontsize=8)
        ax.set_title("A-Scan (traza central)", color=self.COLOR_TEXTO, fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.15, color="#7c3aed")
        ax.tick_params(colors=self.COLOR_TEXTO, labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor("#444466")

        # Marcas de interfaces
        for capa in self.simulador.capas[1:]:
            t_ref = self.simulador.profundidad_a_tiempo(capa.profundidad)
            if t_ref < self.var_ventana.get():
                ax.axhline(t_ref, color=capa.color, alpha=0.5,
                           linestyle="--", linewidth=0.8)

    def _dibujar_bscan(self):
        ax = self.ax_bscan
        ax.clear()
        ax.set_facecolor("#12122a")

        n_trazas = max(20, int(self.var_n_trazas.get()))
        dist = self.var_dist_total.get()
        bscan = self.simulador.generar_bscan(0, dist, n_trazas)

        ganancia = self.var_ganancia.get()
        vmax = np.max(np.abs(bscan)) * ganancia
        if vmax == 0:
            vmax = 1

        t_ns = self.simulador.tiempos * 1e9
        im = ax.imshow(bscan * ganancia,
                        aspect="auto",
                        cmap=self.var_cmap.get(),
                        vmin=-vmax, vmax=vmax,
                        extent=[0, dist, t_ns[-1], t_ns[0]],
                        interpolation="bilinear")

        self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

        ax.set_xlabel("Posición (m)", color=self.COLOR_TEXTO, fontsize=8)
        ax.set_ylabel("Tiempo (ns)", color=self.COLOR_TEXTO, fontsize=8)
        ax.set_title("B-Scan (Radargrama)", color=self.COLOR_TEXTO, fontsize=9)
        ax.tick_params(colors=self.COLOR_TEXTO, labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor("#444466")

        # Marcar anomalías en el B-scan
        for anomalia in self.simulador.anomalias:
            t_ref = self.simulador.profundidad_a_tiempo(anomalia['profundidad'])
            ax.axhline(t_ref, color="#f59e0b", alpha=0.6,
                       linestyle=":", linewidth=1)
            ax.axvline(anomalia['x'], color="#f59e0b", alpha=0.6,
                       linestyle=":", linewidth=1)
            ax.annotate(anomalia['tipo'],
                        xy=(anomalia['x'], t_ref),
                        color="#fbbf24", fontsize=7,
                        xytext=(3, -10), textcoords="offset points")

    def _dibujar_modelo(self):
        ax = self.ax_modelo
        ax.clear()
        ax.set_facecolor("#12122a")

        dist = self.var_dist_total.get()
        capas = self.simulador.capas

        for i in range(len(capas) - 1):
            y_sup = capas[i].profundidad
            y_inf = capas[i + 1].profundidad
            rect = plt.Rectangle((0, y_sup), dist, y_inf - y_sup,
                                   color=capas[i + 1].color, alpha=0.7)
            ax.add_patch(rect)
            ax.text(0.1, (y_sup + y_inf) / 2,
                    f"{capas[i + 1].nombre} (εr={capas[i + 1].permitividad:.1f})",
                    va="center", color="white", fontsize=7,
                    fontweight="bold")

        # Última capa hasta el fondo
        prof_max = capas[-1].profundidad + 2
        rect = plt.Rectangle((0, capas[-1].profundidad), dist,
                               prof_max - capas[-1].profundidad,
                               color=capas[-1].color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(0.1, capas[-1].profundidad + 1,
                f"{capas[-1].nombre} (εr={capas[-1].permitividad:.1f})",
                va="center", color="white", fontsize=7, fontweight="bold")

        # Dibujar anomalías
        for anomalia in self.simulador.anomalias:
            colores = {"tuberia": "#3b82f6", "cavidad": "#1e1e2e",
                       "roca": "#9ca3af", "cable": "#fbbf24"}
            color_an = colores.get(anomalia['tipo'], "#ef4444")
            circulo = plt.Circle((anomalia['x'], anomalia['profundidad']),
                                  0.15, color=color_an, zorder=5)
            ax.add_patch(circulo)
            ax.text(anomalia['x'] + 0.2, anomalia['profundidad'],
                    anomalia['tipo'], color="#fbbf24", fontsize=7)

        ax.set_xlim(0, dist)
        ax.set_ylim(0, prof_max)
        ax.invert_yaxis()
        ax.set_xlabel("Posición (m)", color=self.COLOR_TEXTO, fontsize=8)
        ax.set_ylabel("Profundidad (m)", color=self.COLOR_TEXTO, fontsize=8)
        ax.set_title("Modelo del Subsuelo", color=self.COLOR_TEXTO, fontsize=9)
        ax.grid(True, alpha=0.15, color="#7c3aed")
        ax.tick_params(colors=self.COLOR_TEXTO, labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor("#444466")

        # Línea de antena
        ax.axhline(0, color="#10b981", linewidth=2, label="Antena GPR")
        ax.legend(fontsize=7, loc="lower right",
                  labelcolor=self.COLOR_TEXTO,
                  facecolor=self.COLOR_PANEL, edgecolor="#444466")

    def _dibujar_espectro(self):
        ax = self.ax_espectro
        ax.clear()
        ax.set_facecolor("#12122a")

        ascan = self.simulador.generar_ascan()
        dt = self.simulador.tiempo_muestreo
        n = len(ascan)
        freq = np.fft.rfftfreq(n, d=dt) / 1e6  # MHz
        espectro = np.abs(np.fft.rfft(ascan))

        ax.fill_between(freq, espectro, color="#7c3aed", alpha=0.4)
        ax.plot(freq, espectro, color="#a78bfa", linewidth=1.2)

        # Marcar frecuencia central
        fc = self.simulador.frecuencia_central_mhz
        ax.axvline(fc, color="#f59e0b", linestyle="--",
                   linewidth=1, label=f"fc={fc:.0f} MHz")

        ax.set_xlabel("Frecuencia (MHz)", color=self.COLOR_TEXTO, fontsize=8)
        ax.set_ylabel("Amplitud", color=self.COLOR_TEXTO, fontsize=8)
        ax.set_title("Espectro de Frecuencia", color=self.COLOR_TEXTO, fontsize=9)
        ax.set_xlim(0, min(freq[-1], fc * 3))
        ax.legend(fontsize=7, labelcolor=self.COLOR_TEXTO,
                  facecolor=self.COLOR_PANEL, edgecolor="#444466")
        ax.grid(True, alpha=0.15, color="#7c3aed")
        ax.tick_params(colors=self.COLOR_TEXTO, labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor("#444466")

    # ------------------------------------------------------------------
    # Eventos de botones
    # ------------------------------------------------------------------

    def _agregar_anomalia(self):
        x = self.var_anomalia_x.get()
        prof = self.var_anomalia_prof.get()
        tipo = self.var_tipo_anomalia.get()

        amplitudes = {"tuberia": -0.8, "cavidad": 0.9,
                      "roca": 0.5, "cable": -0.6}
        amp = amplitudes.get(tipo, -0.7)

        self.simulador.agregar_anomalia(x, prof, tipo, amp)
        n = len(self.simulador.anomalias)
        self.lbl_anomalias.config(text=f"{n} anomalía(s) activa(s)")
        self._actualizar_simulacion()

    def _limpiar_anomalias(self):
        self.simulador.limpiar_anomalias()
        self.lbl_anomalias.config(text="Sin anomalías")
        self._actualizar_simulacion()

    def _restablecer(self):
        self.var_frecuencia.set(100.0)
        self.var_ventana.set(200.0)
        self.var_ganancia.set(1.0)
        self.var_n_trazas.set(80)
        self.var_dist_total.set(10.0)
        self.var_cmap.set("seismic")
        self.simulador.limpiar_anomalias()
        self.lbl_anomalias.config(text="Sin anomalías")

        self.simulador = SimuladorRadar()
        for idx, var_eps in self.vars_capas:
            var_eps.set(self.simulador.capas[idx].permitividad)

        self._actualizar_simulacion()

    def _guardar_imagen(self):
        from tkinter import filedialog
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Guardar imagen del radargrama"
        )
        if ruta:
            self.fig.savefig(ruta, dpi=150, bbox_inches="tight",
                              facecolor=self.COLOR_FONDO)
            messagebox.showinfo("Guardado", f"Imagen guardada en:\n{ruta}")

    def _al_cerrar(self):
        plt.close("all")
        self.destroy()
