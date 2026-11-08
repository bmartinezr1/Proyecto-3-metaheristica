"""Algoritmo genético para PFSP; ejecutar con --help para ver los argumentos."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from operadores import (
    actualizar_mejor_solucion,
    calcular_rpd,
    evaluar_poblacion,
    formatear_secuencia,
    generar_poblacion,
    guardar_resultados,
    leer_instancia,
    siguiente_generacion,
)


def ejecutar_ag(tiempos, semilla, tamano_poblacion, iteraciones, prob_cruce, prob_mutacion):
    """Devuelve la mejor secuencia, su makespan y una fila por generación."""
    if tamano_poblacion < 2:
        raise ValueError('La población debe contener al menos dos individuos.')
    if iteraciones <= 0:
        raise ValueError('Las iteraciones deben ser mayores que cero.')
    if not 0 <= prob_cruce <= 1 or not 0 <= prob_mutacion <= 1:
        raise ValueError('Las probabilidades de cruce y mutación deben estar entre 0 y 1.')
    if tiempos.shape[0] < 2:
        raise ValueError('La instancia debe tener al menos dos trabajos.')

    inicio = process_time()
    generador = np.random.default_rng(semilla)

    # Inicializar y evaluar la población una sola vez antes del ciclo.
    poblacion = generar_poblacion(tamano_poblacion, tiempos.shape[0], generador)
    makespans = evaluar_poblacion(poblacion, tiempos)
    indice_mejor = int(np.argmin(makespans))
    mejor_secuencia = list(poblacion[indice_mejor])
    mejor_makespan = makespans[indice_mejor]
    historial = []

    for iteracion in range(1, iteraciones + 1):
        # Selección por torneo, cruce OX, inserción y elitismo están en operadores.py.
        poblacion = siguiente_generacion(
            poblacion, makespans, prob_cruce, prob_mutacion, generador
        )
        makespans = evaluar_poblacion(poblacion, tiempos)

        indice_mejor = int(np.argmin(makespans))
        secuencia_iteracion = poblacion[indice_mejor]
        makespan_iteracion = makespans[indice_mejor]
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
        description='Resuelve una instancia PFSP de Taillard mediante un algoritmo genético.'
    )
    parser.add_argument('instancia', help='Archivo de instancia en formato Taillard.')
    parser.add_argument('semilla', type=int, help='Semilla aleatoria.')
    parser.add_argument('poblacion', type=int, help='Cantidad de individuos (mínimo 2).')
    parser.add_argument('iteraciones', type=int, help='Cantidad de generaciones (mínimo 1).')
    parser.add_argument('prob_cruce', type=float, help='Probabilidad de cruce OX, entre 0 y 1.')
    parser.add_argument('prob_mutacion', type=float, help='Probabilidad de mutación por inserción, entre 0 y 1.')
    parser.add_argument('salida', help='Archivo CSV de salida.')
    return parser


def main():
    parser = crear_parser()
    argumentos = parser.parse_args()
    try:
        tiempos, mejor_conocido = leer_instancia(argumentos.instancia)
        secuencia, makespan, historial = ejecutar_ag(
            tiempos, argumentos.semilla, argumentos.poblacion, argumentos.iteraciones,
            argumentos.prob_cruce, argumentos.prob_mutacion,
        )
        guardar_resultados(
            argumentos.salida, 'AG', argumentos.instancia, argumentos.semilla,
            {
                'iteraciones': argumentos.iteraciones,
                'poblacion': argumentos.poblacion,
                'prob_cruce': argumentos.prob_cruce,
                'prob_mutacion': argumentos.prob_mutacion,
            },
            mejor_conocido, historial,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    trabajos, maquinas = tiempos.shape
    print(f'Instancia: {Path(argumentos.instancia).name} ({trabajos} trabajos x {maquinas} máquinas)')
    print('Mejor secuencia:', formatear_secuencia(secuencia))
    print('Makespan:', makespan)
    print('Mejor conocido:', mejor_conocido)
    print(f'RPD: {calcular_rpd(makespan, mejor_conocido):.2f} %')
    print(f"Tiempo de CPU: {historial[-1]['tiempo_segundos']:.3f} segundos")
    print('Resultados guardados en:', argumentos.salida)


if __name__ == '__main__':
    main()
