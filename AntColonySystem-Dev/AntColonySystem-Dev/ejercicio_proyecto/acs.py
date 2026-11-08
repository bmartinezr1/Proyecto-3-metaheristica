import argparse
from dataclasses import dataclass
from pathlib import Path
from time import process_time

import numpy as np

from operadores import (
    actualizar_feromonas,
    actualizar_mejor_solucion,
    calcular_feromona_inicial,
    calcular_heuristica,
    calcular_rpd,
    construir_soluciones,
    evaluar_soluciones,
    formatear_secuencia,
    guardar_resultados,
    inicializar_feromonas,
    leer_instancia,
)


@dataclass
class ParametrosACS:
    semilla: int
    hormigas: int
    iteraciones: int
    alpha: float
    beta: float
    q0: float

    def __post_init__(self):
        if self.hormigas <= 0:
            raise ValueError('La cantidad de hormigas debe ser mayor que cero.')
        if self.iteraciones <= 0:
            raise ValueError('La cantidad de iteraciones debe ser mayor que cero.')
        if not 0 < self.alpha <= 1:
            raise ValueError('La evaporación (alpha) debe estar entre 0 (excluido) y 1.')
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError('El peso heurístico (beta) debe ser finito y no negativo.')
        if not 0 <= self.q0 <= 1:
            raise ValueError('q0 debe estar entre 0 y 1.')


def ejecutar_acs(tiempos, parametros):
    inicio = process_time()
    # Inicializar
    generador = np.random.default_rng(parametros.semilla)
    feromona_inicial = calcular_feromona_inicial(tiempos)
    feromonas = inicializar_feromonas(len(tiempos), feromona_inicial)
    heuristica = calcular_heuristica(tiempos)
    mejor_secuencia, mejor_makespan = None, float('inf')
    historial = []

    for iteracion in range(1, parametros.iteraciones + 1):
        # Construir soluciones (incluye la actualización local de feromonas)
        soluciones = construir_soluciones(
            parametros.hormigas,
            feromonas,
            heuristica,
            feromona_inicial,
            parametros.alpha,
            parametros.beta,
            parametros.q0,
            generador,
        )

        # Evaluar soluciones
        makespans = evaluar_soluciones(soluciones, tiempos)
        mejor_hormiga = int(np.argmin(makespans))
        secuencia_iteracion = soluciones[mejor_hormiga]
        makespan_iteracion = int(makespans[mejor_hormiga])

        # Actualizar feromonas (actualización global con la mejor hormiga de la iteración)
        actualizar_feromonas(feromonas, secuencia_iteracion, makespan_iteracion, parametros.alpha)

        # Actualizar mejor solución
        mejor_secuencia, mejor_makespan = actualizar_mejor_solucion(
            mejor_secuencia, mejor_makespan, secuencia_iteracion, makespan_iteracion
        )

        historial.append({
            'iteracion': iteracion,
            'secuencia_iteracion': formatear_secuencia(secuencia_iteracion),
            'makespan_iteracion': makespan_iteracion,
            'mejor_secuencia': formatear_secuencia(mejor_secuencia),
            'mejor_makespan': mejor_makespan,
            'tiempo_segundos': process_time() - inicio,
        })

    return mejor_secuencia, mejor_makespan, historial


def crear_parser():
    parser = argparse.ArgumentParser(
        description='Resuelve una instancia PFSP de Taillard mediante Ant Colony System.'
    )
    parser.add_argument('instancia', help='Archivo de instancia en formato Taillard.')
    parser.add_argument('semilla', type=int, help='Semilla aleatoria.')
    parser.add_argument('hormigas', type=int, help='Número de hormigas.')
    parser.add_argument('iteraciones', type=int, help='Número de iteraciones.')
    parser.add_argument('alpha', type=float, help='Evaporación de feromona, en (0, 1].')
    parser.add_argument('beta', type=float, help='Peso de la información heurística.')
    parser.add_argument('q0', type=float, help='Probabilidad de explotación, en [0, 1].')
    parser.add_argument('salida', help='Archivo CSV de salida.')
    return parser


def main():
    parser = crear_parser()
    argumentos = parser.parse_args()

    try:
        parametros = ParametrosACS(
            argumentos.semilla,
            argumentos.hormigas,
            argumentos.iteraciones,
            argumentos.alpha,
            argumentos.beta,
            argumentos.q0,
        )
        tiempos, mejor_conocido = leer_instancia(argumentos.instancia)
        secuencia, makespan, historial = ejecutar_acs(tiempos, parametros)
        guardar_resultados(
            argumentos.salida, 'ACS', argumentos.instancia, parametros.semilla,
            {
                'iteraciones': parametros.iteraciones,
                'hormigas': parametros.hormigas,
                'alpha': parametros.alpha,
                'beta': parametros.beta,
                'q0': parametros.q0,
            },
            mejor_conocido, historial,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    n, m = tiempos.shape
    print(f'Instancia: {Path(argumentos.instancia).name} ({n} trabajos x {m} máquinas)')
    print('Mejor secuencia:', formatear_secuencia(secuencia))
    print('Makespan:', makespan)
    print('Mejor conocido:', mejor_conocido)
    print(f'RPD: {calcular_rpd(makespan, mejor_conocido):.2f} %')
    print(f"Tiempo de CPU: {historial[-1]['tiempo_segundos']:.3f} segundos")
    print('Resultados guardados en:', argumentos.salida)


if __name__ == '__main__':
    main()
