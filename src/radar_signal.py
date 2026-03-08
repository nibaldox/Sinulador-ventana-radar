"""
Módulo de simulación de señal de radar geotécnico (GPR).
Simula la propagación de ondas electromagnéticas en el subsuelo.
"""

import numpy as np


class CapaSubsuelo:
    """Representa una capa del subsuelo con propiedades electromagnéticas."""

    def __init__(self, nombre: str, profundidad: float, permitividad: float,
                 conductividad: float, color: str = "#8B4513"):
        self.nombre = nombre
        self.profundidad = profundidad        # metros desde superficie
        self.permitividad = permitividad      # permitividad relativa (εr)
        self.conductividad = conductividad    # S/m
        self.color = color

    @property
    def velocidad_onda(self) -> float:
        """Velocidad de propagación de la onda en la capa (m/ns)."""
        c = 0.3  # velocidad de la luz en m/ns
        return c / np.sqrt(self.permitividad)

    @property
    def impedancia(self) -> float:
        """Impedancia característica de la capa."""
        mu0 = 1.0
        return np.sqrt(mu0 / self.permitividad)


class SimuladorRadar:
    """
    Simulador de Ground Penetrating Radar (GPR) geotécnico.
    Genera A-scans y B-scans sintéticos basados en modelos del subsuelo.
    """

    def __init__(self):
        # Parámetros del radar
        self.frecuencia_central = 100e6     # Hz (100 MHz)
        self.tiempo_muestreo = 0.1e-9       # segundos (0.1 ns)
        self.ventana_tiempo = 200e-9        # segundos (200 ns)
        self.posicion_antena = 0.0          # metros
        self.paso_antena = 0.05             # metros entre mediciones

        # Capas del subsuelo por defecto
        self.capas: list[CapaSubsuelo] = [
            CapaSubsuelo("Aire/Superficie", 0.0, 1.0, 0.0, "#87CEEB"),
            CapaSubsuelo("Suelo seco", 0.5, 4.0, 0.001, "#D2B48C"),
            CapaSubsuelo("Arcilla húmeda", 1.5, 10.0, 0.01, "#8B6914"),
            CapaSubsuelo("Roca fracturada", 3.0, 6.0, 0.005, "#696969"),
            CapaSubsuelo("Roca sólida", 5.0, 8.0, 0.002, "#4A4A4A"),
        ]

        # Objetos enterrados (anomalías)
        self.anomalias: list[dict] = []

        # Datos generados
        self.tiempos = np.arange(0, self.ventana_tiempo, self.tiempo_muestreo)
        self.n_muestras = len(self.tiempos)

    @property
    def frecuencia_central_mhz(self) -> float:
        return self.frecuencia_central / 1e6

    @frecuencia_central_mhz.setter
    def frecuencia_central_mhz(self, valor: float):
        self.frecuencia_central = valor * 1e6

    @property
    def ventana_tiempo_ns(self) -> float:
        return self.ventana_tiempo * 1e9

    @ventana_tiempo_ns.setter
    def ventana_tiempo_ns(self, valor: float):
        self.ventana_tiempo = valor * 1e-9
        self.tiempos = np.arange(0, self.ventana_tiempo, self.tiempo_muestreo)
        self.n_muestras = len(self.tiempos)

    def pulso_ricker(self, t_centro: float) -> np.ndarray:
        """Genera un pulso Ricker (sombrero mexicano) centrado en t_centro."""
        f = self.frecuencia_central
        t = self.tiempos - t_centro
        factor = (np.pi * f * t) ** 2
        pulso = (1 - 2 * factor) * np.exp(-factor)
        return pulso

    def coeficiente_reflexion(self, capa1: CapaSubsuelo,
                               capa2: CapaSubsuelo) -> float:
        """Calcula el coeficiente de reflexión en la interfaz entre dos capas."""
        z1 = capa1.impedancia
        z2 = capa2.impedancia
        return (z2 - z1) / (z2 + z1)

    def calcular_tiempo_llegada(self, profundidad: float,
                                 posicion: float = 0.0) -> float:
        """
        Calcula el tiempo de llegada del eco desde una profundidad dada.
        Considera la velocidad en cada capa.
        """
        tiempo_total = 0.0
        prof_acumulada = 0.0

        for i, capa in enumerate(self.capas[:-1]):
            prof_siguiente = self.capas[i + 1].profundidad
            if prof_acumulada >= profundidad:
                break

            prof_en_capa = min(profundidad - prof_acumulada,
                               prof_siguiente - capa.profundidad)
            tiempo_total += prof_en_capa / capa.velocidad_onda
            prof_acumulada = prof_siguiente

        if prof_acumulada < profundidad:
            capa_final = self.capas[-1]
            tiempo_total += (profundidad - prof_acumulada) / capa_final.velocidad_onda

        return 2 * tiempo_total * 1e-9  # ida y vuelta, convertir ns a segundos

    def generar_ascan(self, posicion: float = 0.0) -> np.ndarray:
        """
        Genera un A-scan (traza de tiempo) para una posición dada.
        """
        ascan = np.zeros(self.n_muestras)

        # Reflexiones en interfaces entre capas
        for i in range(len(self.capas) - 1):
            capa_sup = self.capas[i]
            capa_inf = self.capas[i + 1]

            coef = self.coeficiente_reflexion(capa_sup, capa_inf)
            if abs(coef) < 0.001:
                continue

            prof = capa_inf.profundidad
            t_llegada = self.calcular_tiempo_llegada(prof, posicion)

            # Atenuación con la profundidad
            atenuacion = np.exp(-0.3 * prof)

            # Agregar pulso al A-scan
            pulso = self.pulso_ricker(t_llegada) * coef * atenuacion
            ascan += pulso

        # Reflexiones de anomalías
        for anomalia in self.anomalias:
            dx = posicion - anomalia['x']
            dy = anomalia['profundidad']
            dist = np.sqrt(dx**2 + dy**2)

            # Obtener velocidad promedio hasta la anomalía
            v_media = self._velocidad_media(anomalia['profundidad'])
            t_llegada = 2 * dist / v_media * 1e-9

            if t_llegada < self.ventana_tiempo:
                amplitud = anomalia['amplitud'] * np.exp(-0.4 * dist)
                pulso = self.pulso_ricker(t_llegada) * amplitud
                ascan += pulso

        # Agregar ruido gaussiano
        ruido = np.random.normal(0, 0.02, self.n_muestras)
        ascan += ruido

        return ascan

    def _velocidad_media(self, profundidad: float) -> float:
        """Calcula la velocidad media de propagación hasta una profundidad."""
        tiempo = 0.0
        prof_acum = 0.0

        for i, capa in enumerate(self.capas[:-1]):
            prof_sig = self.capas[i + 1].profundidad
            if prof_acum >= profundidad:
                break
            en_capa = min(profundidad - prof_acum, prof_sig - capa.profundidad)
            tiempo += en_capa / capa.velocidad_onda
            prof_acum = prof_sig

        if prof_acum < profundidad:
            tiempo += (profundidad - prof_acum) / self.capas[-1].velocidad_onda

        if tiempo > 0:
            return profundidad / tiempo
        return 0.3 / np.sqrt(self.capas[1].permitividad)

    def generar_bscan(self, x_inicio: float = 0.0, x_fin: float = 10.0,
                       n_trazas: int = 100) -> np.ndarray:
        """
        Genera un B-scan (radargrama) barriendo posiciones horizontales.
        Retorna matriz (n_muestras x n_trazas).
        """
        posiciones = np.linspace(x_inicio, x_fin, n_trazas)
        bscan = np.zeros((self.n_muestras, n_trazas))

        for j, pos in enumerate(posiciones):
            bscan[:, j] = self.generar_ascan(pos)

        return bscan

    def agregar_anomalia(self, x: float, profundidad: float,
                          tipo: str = "tuberia", amplitud: float = -0.8):
        """Agrega un objeto enterrado (anomalía) al modelo."""
        self.anomalias.append({
            'x': x,
            'profundidad': profundidad,
            'tipo': tipo,
            'amplitud': amplitud
        })

    def limpiar_anomalias(self):
        """Elimina todas las anomalías del modelo."""
        self.anomalias.clear()

    def profundidad_a_tiempo(self, profundidad: float) -> float:
        """Convierte profundidad (m) a tiempo de llegada (ns)."""
        return self.calcular_tiempo_llegada(profundidad) * 1e9

    def tiempo_a_profundidad(self, tiempo_ns: float) -> float:
        """Convierte tiempo (ns) a profundidad aproximada (m)."""
        # Velocidad promedio en el subsuelo
        v_media = self._velocidad_media(5.0)
        return (tiempo_ns * 1e-9 * v_media) / 2
