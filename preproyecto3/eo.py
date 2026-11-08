"""Extremal Optimization (tau-EO) para el Problema de la Mochila 0/1.

Uso:
    python eo.py INSTANCIA ITERACIONES TAU SEMILLA SALIDA.csv [--indice N]
"""
import argparse
import os
import sys
import time

import numpy as np

from operadores import (
    actualizar_mejor,
    evaluar_ecosistema,
    evaluar_fitness_especies,
    guardar_resultados,
    inicializar_ecosistema,
    leer_instancia,
    mejor_inicial,
    reemplazar_especie,
    seleccionar_especie_tau,
)


def extremal_optimization(pesos, valores, capacidad, iteraciones, tau, rng):
    """Ejecuta tau-EO y devuelve (mejor_solucion, historial por iteración)."""
    x = inicializar_ecosistema(pesos, capacidad, rng)
    mejor = actualizar_mejor(mejor_inicial(len(pesos)), x, evaluar_ecosistema(x, pesos, valores, capacidad))
    historial = []

    for iteracion in range(1, iteraciones + 1):
        fitness = evaluar_fitness_especies(x, pesos, valores, capacidad)
        especie = seleccionar_especie_tau(fitness, tau, rng)
        x = reemplazar_especie(x, especie)

        actual = evaluar_ecosistema(x, pesos, valores, capacidad)
        mejor = actualizar_mejor(mejor, x, actual)
        historial.append({
            "iteracion": iteracion,
            "beneficio_actual": actual["beneficio"],
            "peso_actual": actual["peso"],
            "valido_actual": actual["valido"],
            "mejor_beneficio": mejor["beneficio"],
            "mejor_peso": mejor["peso"],
            "mejor_valido": mejor["valido"],
        })
    return mejor, historial


def parsear_argumentos():
    p = argparse.ArgumentParser(description="Extremal Optimization (tau-EO) para la mochila 0/1")
    p.add_argument("instancia", help="archivo de instancias tipo knapPI")
    p.add_argument("iteraciones", type=int, help="número de pasos evolutivos")
    p.add_argument("tau", type=float, help="parámetro tau de la ley de potencia (> 0)")
    p.add_argument("semilla", type=int, help="semilla aleatoria")
    p.add_argument("salida", help="archivo CSV de resultados")
    p.add_argument("--indice", type=int, default=1, help="instancia dentro del archivo, desde 1 (por defecto 1)")
    args = p.parse_args()
    if args.iteraciones < 1 or args.tau <= 0:
        p.error("se requiere iteraciones >= 1 y tau > 0")
    return args


def main():
    args = parsear_argumentos()
    try:
        capacidad, pesos, valores, z_optimo = leer_instancia(args.instancia, args.indice)
    except (OSError, ValueError) as e:
        print(f"Error al leer la instancia: {e}", file=sys.stderr)
        sys.exit(1)

    nombre = f"{os.path.basename(args.instancia)}#{args.indice}"
    print(f"Instancia: {nombre} | n={len(pesos)} | C={capacidad} | óptimo={z_optimo}")
    print(f"tau={args.tau} | semilla={args.semilla} | iteraciones={args.iteraciones}")

    inicio = time.perf_counter()
    mejor, historial = extremal_optimization(
        pesos, valores, capacidad, args.iteraciones, args.tau, np.random.default_rng(args.semilla)
    )
    tiempo = time.perf_counter() - inicio

    paso = max(1, args.iteraciones // 10)
    for fila in historial[::paso]:
        print(f"  iter {fila['iteracion']:>6} | mejor={fila['mejor_beneficio']}")

    brecha = 100 * (z_optimo - mejor["beneficio"]) / z_optimo
    print(f"Mejor beneficio: {mejor['beneficio']} | peso: {mejor['peso']}/{capacidad} | brecha: {brecha:.2f}%")
    print(f"Tiempo total: {tiempo:.3f} s")

    filas = [{
        "instancia": nombre,
        "semilla": args.semilla,
        "tau": args.tau,
        "z_optimo": z_optimo,
        "tiempo_total_s": round(tiempo, 4),
        **fila,
    } for fila in historial]
    guardar_resultados(filas, args.salida)
    print(f"Resultados guardados en: {args.salida}")


if __name__ == "__main__":
    main()
