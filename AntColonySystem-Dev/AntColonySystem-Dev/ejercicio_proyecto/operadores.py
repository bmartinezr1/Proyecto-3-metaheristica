"""Datos y operadores de AG, AM y ACS para el problema PFSP.

Convenciones:
- Los trabajos se numeran internamente de 0 a n-1 (en pantalla y CSV de 1 a n).
- `tiempos[trabajo, maquina]` es el tiempo de procesamiento p_{i,j}.
- Una solución es una permutación de trabajos (no regresa al trabajo inicial).
- Las feromonas y la heurística tienen forma (n + 1, n): la fila `anterior`
  indica el trabajo programado justo antes y la columna el siguiente. La fila n
  es un nodo de inicio ficticio, para que las hormigas también aprendan qué
  trabajo va primero. A diferencia del TSP, las matrices no son simétricas.
"""
import csv
from pathlib import Path

import numpy as np


def leer_instancia(entrada):
    """Lee una instancia en formato Taillard: n, m, semilla, cota superior y tiempos."""
    lineas = Path(entrada).read_text(encoding='utf-8').splitlines()
    filas = [linea.split() for linea in lineas if linea.split() and linea.split()[0].isdigit()]
    if len(filas) < 2 or len(filas[0]) < 4:
        raise ValueError('La instancia debe seguir el formato de Taillard.')

    n, m, _, cota_superior = (int(valor) for valor in filas[0][:4])
    if n <= 0 or m <= 0 or cota_superior <= 0:
        raise ValueError('Los tamaños y la cota superior deben ser positivos.')
    tiempos = np.array(filas[1:1 + m], dtype=int).T
    if tiempos.shape != (n, m):
        raise ValueError(f'Se esperaban {m} máquinas con {n} trabajos cada una.')
    if np.any(tiempos <= 0):
        raise ValueError('Los tiempos de procesamiento de Taillard deben ser positivos.')
    return tiempos, cota_superior


def secuencia_valida(secuencia, n):
    return len(secuencia) == n and set(secuencia) == set(range(n))


def calcular_finalizacion(secuencia, tiempos):
    """Matriz C(i, j) = max(C(i-1, j), C(i, j-1)) + p[π_i, j] de tamaño (n + 1, m + 1)."""
    n, m = tiempos.shape
    if not secuencia_valida(secuencia, n):
        raise ValueError('La secuencia debe contener cada trabajo exactamente una vez.')

    # La fila 0 y la columna 0 son las condiciones de borde C(0, j) = C(i, 0) = 0.
    finalizacion = np.zeros((n + 1, m + 1), dtype=int)
    for i, trabajo in enumerate(secuencia, start=1):
        for j in range(1, m + 1):
            finalizacion[i, j] = (
                max(finalizacion[i - 1, j], finalizacion[i, j - 1])
                + tiempos[trabajo, j - 1]
            )
    return finalizacion


def calcular_makespan(secuencia, tiempos):
    """Calcula C_max = C(n, m)."""
    return int(calcular_finalizacion(secuencia, tiempos)[-1, -1])


def calcular_ocio_por_posicion(secuencia, tiempos):
    """Tiempo que las máquinas esperan a cada trabajo (según su posición en la secuencia).

    La máquina j queda ociosa antes de procesar el trabajo i durante
    max(C(i-1, j), C(i, j-1)) - C(i-1, j). Se suma sobre las m máquinas.
    """
    finalizacion = calcular_finalizacion(secuencia, tiempos)
    inicio = np.maximum(finalizacion[:-1, 1:], finalizacion[1:, :-1])
    return (inicio - finalizacion[:-1, 1:]).sum(axis=1)


def calcular_rpd(makespan, mejor_conocido):
    """Desviación porcentual relativa respecto del mejor valor conocido."""
    return 100.0 * (makespan - mejor_conocido) / mejor_conocido


def calcular_heuristica(tiempos):
    """Visibilidad η[i, j] = 1 / (1 + ocio estimado al programar j después de i).

    Si j sigue a i, la máquina k+1 queda ociosa cuando j tarda más en la máquina k
    que i en la máquina k+1: ocio[i, j] = Σ_k max(0, p[j, k] - p[i, k+1]).
    Desde el nodo de inicio, el ocio es lo que esperan las máquinas 2..m: Σ_{k<m} p[j, k].
    """
    ocio = np.maximum(0, tiempos[np.newaxis, :, :-1] - tiempos[:, np.newaxis, 1:]).sum(axis=2)
    ocio_inicio = tiempos[:, :-1].sum(axis=1)
    return 1.0 / (1.0 + np.vstack([ocio, ocio_inicio]))


def secuencia_spt(tiempos):
    """Ordena los trabajos de menor a mayor tiempo total (regla SPT)."""
    return [int(trabajo) for trabajo in np.argsort(tiempos.sum(axis=1), kind='stable')]


def calcular_feromona_inicial(tiempos):
    """τ0 = 1 / (n · C_max de la secuencia SPT)."""
    makespan = calcular_makespan(secuencia_spt(tiempos), tiempos)
    return 1.0 / (len(tiempos) * makespan)


def inicializar_feromonas(n, feromona_inicial):
    """Matriz (n + 1, n); la última fila corresponde al nodo de inicio."""
    return np.full((n + 1, n), feromona_inicial, dtype=float)


def calcular_probabilidades(feromonas, heuristica, anterior, libres, beta):
    pesos = feromonas[anterior, libres] * heuristica[anterior, libres] ** beta
    total = pesos.sum()
    if total <= 0:
        return np.full(len(libres), 1.0 / len(libres))
    return pesos / total


def elegir_siguiente(libres, probabilidades, q0, generador):
    """Regla pseudoaleatoria proporcional: explota con q0, explora con 1 - q0."""
    if generador.random() < q0:
        return int(libres[np.argmax(probabilidades)])
    return int(generador.choice(libres, p=probabilidades))


def actualizar_feromona_local(feromonas, anterior, siguiente, alpha, feromona_inicial):
    feromonas[anterior, siguiente] = (
        (1 - alpha) * feromonas[anterior, siguiente] + alpha * feromona_inicial
    )


def construir_secuencia(feromonas, heuristica, feromona_inicial, alpha, beta, q0, generador):
    """Una hormiga agrega trabajos pendientes hasta completar la permutación."""
    n = feromonas.shape[1]
    anterior = n  # nodo de inicio
    pendientes = np.ones(n, dtype=bool)
    secuencia = []

    while len(secuencia) < n:
        libres = np.flatnonzero(pendientes)
        probabilidades = calcular_probabilidades(feromonas, heuristica, anterior, libres, beta)
        siguiente = elegir_siguiente(libres, probabilidades, q0, generador)
        actualizar_feromona_local(feromonas, anterior, siguiente, alpha, feromona_inicial)
        secuencia.append(siguiente)
        pendientes[siguiente] = False
        anterior = siguiente
    return secuencia


def construir_soluciones(hormigas, feromonas, heuristica, feromona_inicial, alpha, beta, q0, generador):
    return [
        construir_secuencia(feromonas, heuristica, feromona_inicial, alpha, beta, q0, generador)
        for _ in range(hormigas)
    ]


def evaluar_soluciones(soluciones, tiempos):
    return np.array([calcular_makespan(secuencia, tiempos) for secuencia in soluciones])


def actualizar_feromonas(feromonas, secuencia, makespan, alpha):
    """Actualización global: solo la mejor hormiga de la iteración deposita 1 / C_max."""
    anteriores = [feromonas.shape[1]] + secuencia[:-1]
    feromonas[anteriores, secuencia] = (
        (1 - alpha) * feromonas[anteriores, secuencia] + alpha / makespan
    )


def actualizar_mejor_solucion(mejor_secuencia, mejor_makespan, secuencia, makespan):
    """Conserva la mejor solución encontrada hasta el momento."""
    if makespan < mejor_makespan:
        return list(secuencia), int(makespan)
    return mejor_secuencia, mejor_makespan


def formatear_secuencia(secuencia):
    return ' '.join(str(trabajo + 1) for trabajo in secuencia)


# Operadores evolutivos compartidos por AG y AM.
def generar_poblacion(tamano, cantidad_trabajos, generador):
    """Crea permutaciones de trabajos independientes."""
    return [generador.permutation(cantidad_trabajos).tolist() for _ in range(tamano)]


def evaluar_poblacion(poblacion, tiempos):
    return [calcular_makespan(individuo, tiempos) for individuo in poblacion]


def seleccionar_padre(poblacion, makespans, generador):
    """Torneo de dos individuos: gana el de menor makespan."""
    primero, segundo = generador.choice(len(poblacion), size=2, replace=False)
    ganador = primero if makespans[primero] <= makespans[segundo] else segundo
    return poblacion[ganador]


def cruce_ox(padre1, padre2, generador):
    """Conserva un segmento y completa cada hijo en el orden circular del otro padre."""
    n = len(padre1)
    inicio, fin = sorted(generador.choice(n, size=2, replace=False))

    def crear_hijo(base, otro):
        hijo = [None] * n
        hijo[inicio:fin + 1] = base[inicio:fin + 1]
        usados = set(hijo[inicio:fin + 1])
        posiciones = list(range(fin + 1, n)) + list(range(fin + 1))
        restantes = [otro[posicion] for posicion in posiciones if otro[posicion] not in usados]
        huecos = [posicion for posicion in posiciones if hijo[posicion] is None]
        for posicion, trabajo in zip(huecos, restantes):
            hijo[posicion] = trabajo
        return hijo

    return crear_hijo(padre1, padre2), crear_hijo(padre2, padre1)


def mutacion_insercion(individuo, generador):
    """Mueve un trabajo a otra posición sin alterar el individuo recibido."""
    mutado = list(individuo)
    if len(mutado) > 1:
        origen, destino = generador.choice(len(mutado), size=2, replace=False)
        mutado.insert(destino, mutado.pop(origen))
    return mutado


def siguiente_generacion(poblacion, makespans, prob_cruce, prob_mutacion, generador):
    """Aplica elitismo de uno, torneo, OX y mutación por inserción."""
    tamano = len(poblacion)
    elite = list(poblacion[int(np.argmin(makespans))])
    nueva = [elite]

    while len(nueva) < tamano:
        padre1 = seleccionar_padre(poblacion, makespans, generador)
        padre2 = seleccionar_padre(poblacion, makespans, generador)
        if generador.random() < prob_cruce:
            hijo1, hijo2 = cruce_ox(padre1, padre2, generador)
        else:
            hijo1, hijo2 = list(padre1), list(padre2)

        for hijo in (hijo1, hijo2):
            if len(nueva) < tamano:
                if generador.random() < prob_mutacion:
                    hijo = mutacion_insercion(hijo, generador)
                nueva.append(hijo)

    return nueva


def busqueda_local_insercion(secuencia, tiempos):
    """Acepta la primera mejora, con un máximo de 10*n vecinos evaluados."""
    actual = list(secuencia)
    costo_actual = calcular_makespan(actual, tiempos)
    n = len(actual)
    evaluaciones = 0
    limite = 10 * n

    while evaluaciones < limite:
        hubo_mejora = False
        for origen in range(n):
            for destino in range(n):
                if origen == destino:
                    continue
                vecino = list(actual)
                vecino.insert(destino, vecino.pop(origen))
                costo_vecino = calcular_makespan(vecino, tiempos)
                evaluaciones += 1
                if costo_vecino < costo_actual:
                    actual, costo_actual = vecino, costo_vecino
                    hubo_mejora = True
                    break
                if evaluaciones >= limite:
                    break
            if hubo_mejora:
                break
            if evaluaciones >= limite:
                break
        if not hubo_mejora:
            break
    return actual, costo_actual


# Operadores de Extremal Optimization (EO): el ecosistema es la permutación y
# cada posición de la secuencia es una especie.
def seleccionar_posicion_tau(fitness, tau, generador):
    """Ordena las especies de peor (k = 1) a mejor (k = n) y elige k con P(k) ∝ k^-tau."""
    n = len(fitness)
    orden = np.argsort(fitness, kind='stable')
    pesos = np.arange(1, n + 1, dtype=float) ** -tau
    k = generador.choice(n, p=pesos / pesos.sum())
    return int(orden[k])


def mover_trabajo(secuencia, posicion, generador):
    """Reemplazo de la especie elegida: inserta ese trabajo en otra posición al azar."""
    nueva = list(secuencia)
    destino = int(generador.integers(len(nueva) - 1))
    if destino >= posicion:
        destino += 1  # nunca elige la posición de origen
    nueva.insert(destino, nueva.pop(posicion))
    return nueva


# Salida común de AG, AM, ACS y EO.
CAMPOS_RESULTADO = [
    'algoritmo', 'instancia', 'semilla', 'iteraciones',
    'poblacion', 'prob_cruce', 'prob_mutacion', 'frecuencia_bl',
    'hormigas', 'alpha', 'beta', 'q0', 'tau',
    'iteracion', 'secuencia_iteracion', 'makespan_iteracion',
    'mejor_secuencia', 'mejor_makespan', 'mejor_conocido', 'rpd', 'tiempo_segundos',
]


def guardar_resultados(salida, algoritmo, instancia, semilla, parametros, mejor_conocido, historial):
    """Guarda una fila por iteración y deja vacíos los parámetros ajenos al método."""
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open('w', newline='', encoding='utf-8') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS_RESULTADO)
        escritor.writeheader()
        for registro in historial:
            escritor.writerow({
                'algoritmo': algoritmo,
                'instancia': Path(instancia).name,
                'semilla': semilla,
                **parametros,
                **registro,
                'mejor_conocido': mejor_conocido,
                'rpd': calcular_rpd(registro['mejor_makespan'], mejor_conocido),
            })
