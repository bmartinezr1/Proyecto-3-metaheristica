"""Extremal Optimization (τ-EO) para PFSP; ejecutar con --help para ver los argumentos."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from operadores import (
    actualizar_mejor_solucion,
    calcular_makespan,
    calcular_ocio_por_posicion,
    calcular_rpd,
    formatear_secuencia,
    guardar_resultados,
    leer_instancia,
    mover_trabajo,
    seleccionar_posicion_tau,
)


def ejecutar_eo(tiempos, semilla, iteraciones, tau):
    """Devuelve la mejor secuencia, su makespan y una fila por iteración.

    Cada iteración evalúa un solo makespan (el del nuevo estado), además del
    cálculo del ocio que define el fitness de cada especie.
    """
    if iteraciones <= 0:
        raise ValueError('Las iteraciones deben ser mayores que cero.')
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('tau debe ser finito y mayor que cero.')
    if tiempos.shape[0] < 2:
        raise ValueError('La instancia debe tener al menos dos trabajos.')

    inicio = process_time()
    generador = np.random.default_rng(semilla)

    # Inicializar el ecosistema: una permutación aleatoria de los trabajos.
    secuencia = generador.permutation(tiempos.shape[0]).tolist()
    mejor_secuencia = list(secuencia)
    mejor_makespan = calcular_makespan(secuencia, tiempos)
    historial = []

    for iteracion in range(1, iteraciones + 1):
        # Fitness de cada especie: menos ocio provocado = mejor adaptada.
        fitness = -calcular_ocio_por_posicion(secuencia, tiempos)

        # Elegir la especie con ley de potencia y reemplazarla (se acepta siempre).
        posicion = seleccionar_posicion_tau(fitness, tau, generador)
        secuencia = mover_trabajo(secuencia, posicion, generador)

        # Evaluar el ecosistema y actualizar la mejor solución.
        makespan = calcular_makespan(secuencia, tiempos)
        mejor_secuencia, mejor_makespan = actualizar_mejor_solucion(
            mejor_secuencia, mejor_makespan, secuencia, makespan
        )

        historial.append({
            'iteracion': iteracion,
            'secuencia_iteracion': formatear_secuencia(secuencia),
            'makespan_iteracion': makespan,
            'mejor_secuencia': formatear_secuencia(mejor_secuencia),
            'mejor_makespan': mejor_makespan,
            'tiempo_segundos': process_time() - inicio,
        })

    return mejor_secuencia, mejor_makespan, historial


def crear_parser():
    parser = argparse.ArgumentParser(
        description='Resuelve una instancia PFSP de Taillard mediante Extremal Optimization (τ-EO).'
    )
    parser.add_argument('instancia', help='Archivo de instancia en formato Taillard.')
    parser.add_argument('semilla', type=int, help='Semilla aleatoria.')
    parser.add_argument('iteraciones', type=int, help='Cantidad de iteraciones (mínimo 1).')
    parser.add_argument('tau', type=float, help='Parámetro de la ley de potencia P(k) ∝ k^-tau (mayor que 0).')
    parser.add_argument('salida', help='Archivo CSV de salida.')
    return parser


def main():
    parser = crear_parser()
    argumentos = parser.parse_args()
    try:
        tiempos, mejor_conocido = leer_instancia(argumentos.instancia)
        secuencia, makespan, historial = ejecutar_eo(
            tiempos, argumentos.semilla, argumentos.iteraciones, argumentos.tau,
        )
        guardar_resultados(
            argumentos.salida, 'EO', argumentos.instancia, argumentos.semilla,
            {'iteraciones': argumentos.iteraciones, 'tau': argumentos.tau},
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
