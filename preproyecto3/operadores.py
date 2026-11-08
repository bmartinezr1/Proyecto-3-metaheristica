"""Operadores de Extremal Optimization (tau-EO) para la Mochila 0/1.

Cada elemento de la mochila es una "especie". El estado x es un vector binario
(1 = dentro de la mochila, 0 = fuera).
"""
import os

import numpy as np
import pandas as pd


def leer_instancia(ruta, indice=1):
    """Lee la instancia número `indice` (desde 1) de un archivo tipo knapPI.

    Devuelve (capacidad, pesos, valores, z_optimo).
    """
    with open(ruta) as f:
        lineas = [linea.strip() for linea in f if linea.strip()]

    inicios = [i for i, linea in enumerate(lineas) if linea.startswith("knapPI")]
    if not 1 <= indice <= len(inicios):
        raise ValueError(f"El archivo tiene {len(inicios)} instancias; {indice} está fuera de rango")

    ini = inicios[indice - 1] + 1
    fin = inicios[indice] if indice < len(inicios) else len(lineas)
    cabecera = dict(linea.split() for linea in lineas[ini:ini + 4])  # n, c, z, time
    filas = np.array([linea.split(",") for linea in lineas[ini + 4:fin] if "," in linea], dtype=int)

    if len(filas) != int(cabecera["n"]):
        raise ValueError("El número de elementos no coincide con 'n' de la cabecera")

    capacidad, z_optimo = int(cabecera["c"]), int(cabecera["z"])
    # Columnas: índice, beneficio, peso, solución óptima (0/1)
    return capacidad, filas[:, 2], filas[:, 1], z_optimo


def inicializar_ecosistema(pesos, capacidad, rng):
    """Estado inicial factible: llena la mochila en orden aleatorio (depende de la semilla)."""
    x = np.zeros(len(pesos), dtype=int)
    peso = 0
    for i in rng.permutation(len(pesos)):
        if peso + pesos[i] <= capacidad:
            x[i] = 1
            peso += pesos[i]
    return x


def evaluar_fitness_especies(x, pesos, valores, capacidad):
    """Fitness individual q_i en [0, 1]; valores bajos = especie mal adaptada.

    Un elemento está bien adaptado si está dentro y es eficiente (v/w alto)
    o si está fuera y es ineficiente. Si la mochila excede la capacidad,
    todo elemento dentro queda peor que cualquier elemento fuera.
    """
    # Eficiencia relativa: posición de v/w entre todos los elementos, en [0, 1]
    eficiencia = np.argsort(np.argsort(valores / pesos)) / max(1, len(pesos) - 1)
    q = np.where(x == 1, eficiencia, 1 - eficiencia)

    if int(pesos @ x) > capacidad:
        q = q - x  # con sobrepeso, los elementos dentro quedan peor que cualquiera fuera
    return q


def probabilidades_tau(n, tau):
    """Probabilidades de ranking P(k) proporcional a k^-tau, para k = 1..n."""
    pesos_rango = np.arange(1, n + 1, dtype=float) ** -tau
    return pesos_rango / pesos_rango.sum()


def seleccionar_especie_tau(q, tau, rng):
    """Ordena las especies de peor a mejor y elige una según la ley de potencia."""
    orden = np.argsort(q, kind="stable")  # orden[0] es la peor especie (k = 1)
    k = rng.choice(len(q), p=probabilidades_tau(len(q), tau))
    return orden[k]


def reemplazar_especie(x, i):
    """Operador bit-flip: cambia el estado del elemento i. No modifica x."""
    nuevo = x.copy()
    nuevo[i] = 1 - nuevo[i]
    return nuevo


def evaluar_ecosistema(x, pesos, valores, capacidad):
    """Beneficio, peso y validez del estado, con penalización por exceso de capacidad."""
    beneficio = int(valores @ x)
    peso = int(pesos @ x)
    exceso = max(0, peso - capacidad)
    penalizacion = (valores / pesos).max() * exceso
    return {
        "beneficio": beneficio,
        "peso": peso,
        "valido": peso <= capacidad,
        "objetivo": beneficio - penalizacion,
    }


def mejor_inicial(n):
    """Mejor solución válida de partida: la mochila vacía."""
    return {"x": np.zeros(n, dtype=int), "beneficio": 0, "peso": 0, "valido": True}


def actualizar_mejor(mejor, x, evaluacion):
    """Guarda x como mejor solución solo si es válida y mejora el beneficio."""
    if evaluacion["valido"] and evaluacion["beneficio"] > mejor["beneficio"]:
        return {"x": x.copy(), "beneficio": evaluacion["beneficio"],
                "peso": evaluacion["peso"], "valido": True}
    return mejor


def guardar_resultados(filas, ruta):
    """Escribe la lista de diccionarios como CSV con pandas."""
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)
