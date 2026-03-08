"""
Simulador de Radar Geotécnico (GPR)
====================================
Punto de entrada principal de la aplicación.

Uso:
    python main.py

Requisitos:
    pip install numpy matplotlib

Descripción:
    Simula el comportamiento de un radar de penetración de suelo (GPR)
    para aplicaciones geotécnicas. Permite visualizar A-scans, B-scans
    (radargramas), el modelo del subsuelo y el espectro de frecuencia
    de la señal emitida.
"""

import sys
import os

# Agregar el directorio src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gui import VentanaRadar


def main():
    app = VentanaRadar()
    app.mainloop()


if __name__ == "__main__":
    main()
