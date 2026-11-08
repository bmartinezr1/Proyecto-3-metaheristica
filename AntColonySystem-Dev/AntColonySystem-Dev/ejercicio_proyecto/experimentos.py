"""Experimentos de sensibilidad de ACS: se varía un parámetro a la vez."""
import argparse
import csv
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from acs import ParametrosACS, ejecutar_acs
from operadores import calcular_rpd, formatear_secuencia, leer_instancia

CARPETA = Path(__file__).parent
INSTANCIAS = ['tai20_5.txt', 'tai50_10.txt', 'tai100_10.txt']
SEMILLAS = [1, 2, 3, 4, 5]
BASE = ParametrosACS(semilla=0, hormigas=20, iteraciones=100, alpha=0.1, beta=2.0, q0=0.9)
FACTORES = {
    'hormigas': [5, 10, 20, 40],
    'iteraciones': [25, 50, 100, 200],
    'alpha': [0.05, 0.1, 0.3, 0.6],
    'beta': [0.0, 1.0, 2.0, 5.0],
    'q0': [0.5, 0.75, 0.9, 0.98],
}
CAMPOS = [
    'instancia', 'factor', 'valor', 'semilla', 'hormigas', 'iteraciones', 'alpha', 'beta',
    'q0', 'makespan', 'mejor_conocido', 'rpd', 'tiempo_segundos', 'mejor_secuencia',
]


def ejecutar_experimentos(salida):
    salida.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    with salida.open('w', newline='', encoding='utf-8') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        escritor.writeheader()
        for instancia in INSTANCIAS:
            tiempos, mejor_conocido = leer_instancia(CARPETA / 'datos' / instancia)
            for factor, valores in FACTORES.items():
                for valor in valores:
                    for semilla in SEMILLAS:
                        parametros = replace(BASE, semilla=semilla, **{factor: valor})
                        secuencia, makespan, historial = ejecutar_acs(tiempos, parametros)
                        fila = {
                            'instancia': instancia,
                            'factor': factor,
                            'valor': valor,
                            **asdict(parametros),
                            'makespan': makespan,
                            'mejor_conocido': mejor_conocido,
                            'rpd': round(calcular_rpd(makespan, mejor_conocido), 4),
                            'tiempo_segundos': round(historial[-1]['tiempo_segundos'], 4),
                            'mejor_secuencia': formatear_secuencia(secuencia),
                        }
                        escritor.writerow(fila)
                        archivo.flush()
                        filas.append(fila)
                    mostrar_resumen(filas[-len(SEMILLAS):])
    return filas


def mostrar_resumen(filas):
    rpd = np.array([fila['rpd'] for fila in filas])
    tiempo = np.mean([fila['tiempo_segundos'] for fila in filas])
    primera = filas[0]
    print(
        f"{primera['instancia']:<14} {primera['factor']:<11} {primera['valor']:>6} | "
        f"RPD media {rpd.mean():6.2f} %  mín {rpd.min():6.2f} %  "
        f"desv {rpd.std():5.2f}  | tiempo {tiempo:6.2f} s"
    )



parser = argparse.ArgumentParser(description='Ejecuta los experimentos de parámetros de ACS.')
parser.add_argument(
    'salida',
    nargs='?',
    default=CARPETA / 'resultados' / 'experimentos.csv',
    type=Path,
    help='Archivo CSV de salida.',
)
argumentos = parser.parse_args()
ejecutar_experimentos(argumentos.salida)
print('Resultados guardados en:', argumentos.salida)
