"""Experimentos: efecto de tau y de la semilla sobre tau-EO (mochila 0/1).

Uso: python experimentos.py
Genera resultados/experimentos.csv
"""
import time

import numpy as np

from eo import extremal_optimization
from operadores import guardar_resultados, leer_instancia

INSTANCIAS = ["datos/knapPI_1_50_1000.csv", "datos/knapPI_1_500_1000.csv", "datos/knapPI_1_5000_1000.csv"]
TAUS = [1.0, 1.4, 2.0, 3.0]
SEMILLAS = [1, 2, 3, 4, 5]
ITERACIONES = 1000
SALIDA = "resultados/experimentos.csv"


def main():
    filas = []
    for ruta in INSTANCIAS:
        capacidad, pesos, valores, z_optimo = leer_instancia(ruta, 1)
        for tau in TAUS:
            for semilla in SEMILLAS:
                inicio = time.perf_counter()
                mejor, historial = extremal_optimization(
                    pesos, valores, capacidad, ITERACIONES, tau, np.random.default_rng(semilla)
                )
                tiempo = time.perf_counter() - inicio
                brecha = 100 * (z_optimo - mejor["beneficio"]) / z_optimo
                print(f"{ruta} tau={tau} semilla={semilla} -> mejor={mejor['beneficio']} brecha={brecha:.2f}% t={tiempo:.2f}s")
                filas += [{
                    "instancia": ruta.split("/")[-1],
                    "n": len(pesos),
                    "semilla": semilla,
                    "tau": tau,
                    "z_optimo": z_optimo,
                    "tiempo_total_s": round(tiempo, 4),
                    **fila,
                } for fila in historial]
    guardar_resultados(filas, SALIDA)
    print(f"Guardado: {SALIDA} ({len(filas)} filas)")


if __name__ == "__main__":
    main()
