import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from acs import ParametrosACS, ejecutar_acs
from ag import ejecutar_ag
from am import ejecutar_am
from eo import ejecutar_eo
from graficar import graficar_comparacion
from main import ALGORITMOS, PARAMETROS, ejecutar_algoritmo, resumir
from operadores import (
    actualizar_feromona_local,
    actualizar_feromonas,
    actualizar_mejor_solucion,
    busqueda_local_insercion,
    calcular_feromona_inicial,
    calcular_heuristica,
    calcular_makespan,
    calcular_ocio_por_posicion,
    calcular_rpd,
    cruce_ox,
    construir_secuencia,
    evaluar_poblacion,
    generar_poblacion,
    guardar_resultados,
    inicializar_feromonas,
    leer_instancia,
    mover_trabajo,
    mutacion_insercion,
    secuencia_valida,
    seleccionar_posicion_tau,
    siguiente_generacion,
)

DATOS = Path(__file__).parent / 'datos'
RESULTADOS = Path(__file__).parent / 'resultados' / 'comparacion_algoritmos'

# 3 trabajos x 2 máquinas: tiempos[trabajo, maquina]
EJEMPLO = np.array([
    [3, 2],
    [1, 4],
    [2, 1],
])


def probar_lectura_taillard():
    tiempos, cota = leer_instancia(DATOS / 'tai20_5.txt')
    assert tiempos.shape == (20, 5)
    assert cota == 1278
    # Primera fila de ta001 (máquina 1) y primer valor de la máquina 2.
    assert list(tiempos[:10, 0]) == [54, 83, 15, 71, 77, 36, 53, 38, 27, 87]
    assert tiempos[0, 1] == 79

    for nombre, forma, cota_esperada in [
        ('tai50_10.txt', (50, 10), 2991),
        ('tai100_10.txt', (100, 10), 5770),
    ]:
        tiempos, cota = leer_instancia(DATOS / nombre)
        assert tiempos.shape == forma
        assert cota == cota_esperada
        assert tiempos.min() >= 1 and tiempos.max() <= 99


def probar_makespan_calculado_a_mano():
    # Secuencia 1-2-3:          Secuencia 2-1-3:
    #   M1: 3, 4, 6               M1: 1, 4, 6
    #   M2: 5, 9, 10              M2: 5, 7, 8
    assert calcular_makespan([0, 1, 2], EJEMPLO) == 10
    assert calcular_makespan([1, 0, 2], EJEMPLO) == 8
    # Con una sola máquina, C_max es la suma de los tiempos.
    assert calcular_makespan([2, 0, 1], EJEMPLO[:, :1]) == 6

    try:
        calcular_makespan([0, 0, 2], EJEMPLO)
    except ValueError:
        pass
    else:
        raise AssertionError('Una secuencia con trabajos repetidos debe rechazarse.')


def probar_secuencias_y_rpd():
    assert secuencia_valida([2, 0, 1], 3)
    assert not secuencia_valida([0, 1], 3)
    assert not secuencia_valida([0, 1, 1], 3)
    assert np.isclose(calcular_rpd(1300, 1250), 4.0)
    assert calcular_rpd(1278, 1278) == 0.0


def probar_heuristica():
    heuristica = calcular_heuristica(EJEMPLO)
    assert heuristica.shape == (4, 3)
    # Ocio de programar el trabajo 1 tras el 3: max(0, p[1,M1] - p[3,M2]) = 3 - 1 = 2.
    assert np.isclose(heuristica[2, 0], 1 / 3)
    assert np.isclose(heuristica[0, 1], 1.0)
    # Desde el inicio el ocio es el tiempo en M1: 3, 1 y 2.
    assert np.allclose(heuristica[3], [1 / 4, 1 / 2, 1 / 3])


def probar_feromonas():
    tau0 = calcular_feromona_inicial(EJEMPLO)
    # SPT ordena por tiempo total (5, 5, 3): secuencia 3-1-2 con C_max = 11.
    assert np.isclose(tau0, 1 / (3 * 11))

    feromonas = inicializar_feromonas(3, tau0)
    assert feromonas.shape == (4, 3)
    assert np.all(feromonas == tau0)

    feromonas[0, 1] = 0.2
    actualizar_feromona_local(feromonas, 0, 1, 0.1, tau0)
    assert np.isclose(feromonas[0, 1], 0.9 * 0.2 + 0.1 * tau0)
    assert feromonas[1, 0] == tau0  # matriz dirigida: no se toca la arista inversa

    antes = feromonas.copy()
    actualizar_feromonas(feromonas, [1, 0, 2], 8, 0.1)
    for anterior, siguiente in [(3, 1), (1, 0), (0, 2)]:
        assert np.isclose(feromonas[anterior, siguiente], 0.9 * antes[anterior, siguiente] + 0.1 / 8)
    assert feromonas[0, 1] == antes[0, 1]  # arista fuera de la secuencia


def probar_construccion():
    tiempos, _ = leer_instancia(DATOS / 'tai20_5.txt')
    tau0 = calcular_feromona_inicial(tiempos)
    feromonas = inicializar_feromonas(20, tau0)
    heuristica = calcular_heuristica(tiempos)
    generador = np.random.default_rng(7)
    for _ in range(10):
        secuencia = construir_secuencia(feromonas, heuristica, tau0, 0.1, 2.0, 0.9, generador)
        assert secuencia_valida(secuencia, 20)


def probar_mejor_solucion():
    assert actualizar_mejor_solucion(None, float('inf'), [1, 0], 9) == ([1, 0], 9)
    assert actualizar_mejor_solucion([1, 0], 9, [0, 1], 12) == ([1, 0], 9)
    assert actualizar_mejor_solucion([1, 0], 9, [0, 1], 7) == ([0, 1], 7)


def probar_algoritmo():
    tiempos, _ = leer_instancia(DATOS / 'tai20_5.txt')
    parametros = ParametrosACS(123, 5, 15, 0.1, 2.0, 0.9)
    secuencia_1, makespan_1, historial = ejecutar_acs(tiempos, parametros)
    secuencia_2, makespan_2, _ = ejecutar_acs(tiempos, parametros)

    # Reproducibilidad: misma semilla, misma secuencia y makespan.
    assert secuencia_1 == secuencia_2
    assert makespan_1 == makespan_2
    assert calcular_makespan(secuencia_1, tiempos) == makespan_1

    # La mejor solución mejora o se mantiene en cada iteración.
    mejores = [registro['mejor_makespan'] for registro in historial]
    assert mejores == sorted(mejores, reverse=True)
    assert all(
        registro['mejor_makespan'] <= registro['makespan_iteracion'] for registro in historial
    )


def probar_operadores_ag():
    generador = np.random.default_rng(42)
    padre1 = list(range(7))
    padre2 = list(reversed(padre1))
    for _ in range(30):
        hijo1, hijo2 = cruce_ox(padre1, padre2, generador)
        mutado = mutacion_insercion(hijo1, generador)
        assert secuencia_valida(hijo1, 7)
        assert secuencia_valida(hijo2, 7)
        assert secuencia_valida(mutado, 7)
        assert padre1 == list(range(7)) and padre2 == list(reversed(padre1))
        assert hijo1 != mutado  # se eligieron posiciones distintas

    poblacion = generar_poblacion(5, 3, generador)
    makespans = evaluar_poblacion(poblacion, EJEMPLO)
    antes = [list(individuo) for individuo in poblacion]
    nueva = siguiente_generacion(poblacion, makespans, 1.0, 1.0, generador)
    assert len(nueva) == 5  # también funciona con población impar
    assert all(secuencia_valida(individuo, 3) for individuo in nueva)
    assert nueva[0] == antes[int(np.argmin(makespans))]  # elitismo
    assert poblacion == antes  # los padres no se mutan en sitio


def probar_ag_completo():
    tiempos, mejor_conocido = leer_instancia(DATOS / 'tai20_5.txt')
    secuencia, makespan, historial = ejecutar_ag(tiempos, 123, 7, 6, 0.8, 0.1)
    repetida, costo_repetido, _ = ejecutar_ag(tiempos, 123, 7, 6, 0.8, 0.1)
    assert (secuencia, makespan) == (repetida, costo_repetido)
    assert secuencia_valida(secuencia, 20)
    assert calcular_makespan(secuencia, tiempos) == makespan
    assert len(historial) == 6
    assert [fila['iteracion'] for fila in historial] == list(range(1, 7))
    assert [fila['mejor_makespan'] for fila in historial] == sorted(
        (fila['mejor_makespan'] for fila in historial), reverse=True
    )
    for fila in historial:
        secuencia_fila = [int(valor) - 1 for valor in fila['mejor_secuencia'].split()]
        assert calcular_makespan(secuencia_fila, tiempos) == fila['mejor_makespan']
        assert fila['mejor_makespan'] <= fila['makespan_iteracion']

    with TemporaryDirectory(dir=Path(__file__).parent) as temporal:
        salida = Path(temporal) / 'res_ag.csv'
        guardar_resultados(
            salida, 'AG', DATOS / 'tai20_5.txt', 123,
            {'iteraciones': 6, 'poblacion': 7, 'prob_cruce': 0.8, 'prob_mutacion': 0.1},
            mejor_conocido, historial,
        )
        with salida.open(newline='', encoding='utf-8') as archivo:
            filas = list(csv.DictReader(archivo))
        assert len(filas) == 6
        assert filas[-1]['algoritmo'] == 'AG'
        assert filas[-1]['mejor_secuencia'] == historial[-1]['mejor_secuencia']
        assert int(filas[-1]['mejor_makespan']) == makespan
        assert np.isclose(float(filas[-1]['rpd']), calcular_rpd(makespan, mejor_conocido))

    for parametros in [(1, 6, 0.8, 0.1), (7, 0, 0.8, 0.1), (7, 6, 1.2, 0.1)]:
        try:
            ejecutar_ag(tiempos, 123, *parametros)
        except ValueError:
            pass
        else:
            raise AssertionError('El AG debe rechazar parámetros fuera de rango.')


def probar_busqueda_local():
    original = [0, 1, 2]
    with patch('operadores.calcular_makespan', wraps=calcular_makespan) as evaluar:
        mejorada, costo = busqueda_local_insercion(original, EJEMPLO)
    assert original == [0, 1, 2]
    assert secuencia_valida(mejorada, 3)
    assert costo == calcular_makespan(mejorada, EJEMPLO)
    assert costo < calcular_makespan(original, EJEMPLO)
    assert evaluar.call_count <= 1 + 10 * len(original)


def probar_am_completo():
    with patch('am.busqueda_local_insercion', wraps=busqueda_local_insercion) as buscar:
        secuencia, makespan, historial = ejecutar_am(EJEMPLO, 123, 5, 12, 0.8, 0.1, 5)
    assert buscar.call_count == 2  # generaciones 5 y 10
    repetida, costo_repetido, _ = ejecutar_am(EJEMPLO, 123, 5, 12, 0.8, 0.1, 5)
    assert (secuencia, makespan) == (repetida, costo_repetido)
    assert secuencia_valida(secuencia, 3)
    assert calcular_makespan(secuencia, EJEMPLO) == makespan
    assert len(historial) == 12
    assert [fila['mejor_makespan'] for fila in historial] == sorted(
        (fila['mejor_makespan'] for fila in historial), reverse=True
    )
    for fila in historial:
        secuencia_fila = [int(valor) - 1 for valor in fila['mejor_secuencia'].split()]
        assert calcular_makespan(secuencia_fila, EJEMPLO) == fila['mejor_makespan']
        assert fila['mejor_makespan'] <= fila['makespan_iteracion']

    # Si no toca búsqueda local, AM sigue exactamente el ciclo del AG.
    ag_secuencia, ag_costo, _ = ejecutar_ag(EJEMPLO, 123, 5, 4, 0.8, 0.1)
    am_secuencia, am_costo, _ = ejecutar_am(EJEMPLO, 123, 5, 4, 0.8, 0.1, 5)
    assert (am_secuencia, am_costo) == (ag_secuencia, ag_costo)

    with TemporaryDirectory(dir=Path(__file__).parent) as temporal:
        salida = Path(temporal) / 'res_am.csv'
        guardar_resultados(
            salida, 'AM', 'ejemplo_3x2.txt', 123,
            {'iteraciones': 12, 'poblacion': 5, 'prob_cruce': 0.8,
             'prob_mutacion': 0.1, 'frecuencia_bl': 5},
            10, historial,
        )
        with salida.open(newline='', encoding='utf-8') as archivo:
            filas = list(csv.DictReader(archivo))
        assert len(filas) == 12
        assert filas[-1]['algoritmo'] == 'AM'
        assert filas[-1]['frecuencia_bl'] == '5'
        assert int(filas[-1]['mejor_makespan']) == makespan

    for frecuencia in (0, -1):
        try:
            ejecutar_am(EJEMPLO, 123, 5, 12, 0.8, 0.1, frecuencia)
        except ValueError:
            pass
        else:
            raise AssertionError('AM debe rechazar frecuencias no positivas.')


def probar_operadores_eo():
    # Ejemplo a mano con la secuencia (1, 2, 3): solo el primer trabajo hace esperar
    # a la máquina 2 (3 unidades); los demás llegan cuando la máquina ya está libre.
    assert list(calcular_ocio_por_posicion([0, 1, 2], EJEMPLO)) == [3, 0, 0]

    # Con tau alto casi siempre se elige la posición peor adaptada (menor fitness).
    generador = np.random.default_rng(0)
    fitness = np.array([-5, -1, -9, -2])
    elecciones = [seleccionar_posicion_tau(fitness, 10.0, generador) for _ in range(200)]
    assert elecciones.count(2) >= 190
    assert set(elecciones) <= {0, 1, 2, 3}

    # Con tau pequeño la elección es casi uniforme y las demás posiciones aparecen.
    elecciones = [seleccionar_posicion_tau(fitness, 0.01, generador) for _ in range(400)]
    assert set(elecciones) == {0, 1, 2, 3}

    # El reemplazo mueve solo el trabajo elegido a una posición distinta.
    original = [0, 1, 2, 3, 4, 5]
    for posicion in range(6):
        for _ in range(20):
            nueva = mover_trabajo(original, posicion, generador)
            assert original == [0, 1, 2, 3, 4, 5]
            assert secuencia_valida(nueva, 6) and nueva != original
            assert [t for t in nueva if t != posicion] == [t for t in original if t != posicion]
    assert mover_trabajo([0, 1], 0, generador) == [1, 0]


def probar_eo_completo():
    tiempos, mejor_conocido = leer_instancia(DATOS / 'tai20_5.txt')
    secuencia, makespan, historial = ejecutar_eo(tiempos, 123, 300, 1.4)
    repetida, costo_repetido, _ = ejecutar_eo(tiempos, 123, 300, 1.4)
    assert (secuencia, makespan) == (repetida, costo_repetido)
    assert secuencia_valida(secuencia, 20)
    assert calcular_makespan(secuencia, tiempos) == makespan
    assert [fila['iteracion'] for fila in historial] == list(range(1, 301))
    mejores = [fila['mejor_makespan'] for fila in historial]
    assert mejores == sorted(mejores, reverse=True)
    assert mejores[-1] == makespan
    for fila in historial:
        actual = [int(valor) - 1 for valor in fila['secuencia_iteracion'].split()]
        assert calcular_makespan(actual, tiempos) == fila['makespan_iteracion']
        assert fila['mejor_makespan'] <= fila['makespan_iteracion']

    with TemporaryDirectory(dir=Path(__file__).parent) as temporal:
        salida = Path(temporal) / 'res_eo.csv'
        guardar_resultados(
            salida, 'EO', DATOS / 'tai20_5.txt', 123,
            {'iteraciones': 300, 'tau': 1.4}, mejor_conocido, historial,
        )
        with salida.open(newline='', encoding='utf-8') as archivo:
            filas = list(csv.DictReader(archivo))
        assert len(filas) == 300
        assert filas[-1]['algoritmo'] == 'EO' and filas[-1]['tau'] == '1.4'
        assert filas[-1]['poblacion'] == '' and filas[-1]['hormigas'] == ''
        assert int(filas[-1]['mejor_makespan']) == makespan
        assert np.isclose(float(filas[-1]['rpd']), calcular_rpd(makespan, mejor_conocido))

    for iteraciones, tau in [(0, 1.4), (10, 0), (10, -1.0), (10, float('inf'))]:
        try:
            ejecutar_eo(tiempos, 123, iteraciones, tau)
        except ValueError:
            pass
        else:
            raise AssertionError('EO debe rechazar parámetros fuera de rango.')


def probar_comparacion_main():
    tiempos, mejor_conocido = leer_instancia(DATOS / 'tai20_5.txt')
    assert ALGORITMOS == ('AG', 'AM', 'ACS', 'EO') and 20 in PARAMETROS
    with TemporaryDirectory(dir=Path(__file__).parent) as temporal:
        for algoritmo in ALGORITMOS:
            secuencia, makespan, historial, parametros = ejecutar_algoritmo(
                algoritmo, tiempos, 7, 200, PARAMETROS[20]
            )
            assert secuencia_valida(secuencia, 20)
            assert calcular_makespan(secuencia, tiempos) == makespan
            guardar_resultados(
                Path(temporal) / f'{algoritmo}_tai20_5_s7.csv', algoritmo,
                DATOS / 'tai20_5.txt', 7, parametros, mejor_conocido, historial,
            )
        # Mismo presupuesto: AG, AM y ACS usan 4 iteraciones de 50; EO usa 200 iteraciones
        # y registra una fila cada 50, es decir 4 filas con la última en la iteración 200.
        assert len(historial) == 4 and historial[-1]['iteracion'] == 200
        resumen = resumir(Path(temporal))
        assert sorted(fila['algoritmo'] for fila in resumen) == sorted(ALGORITMOS)
        assert (Path(temporal) / 'resumen.csv').stat().st_size > 0


def probar_grafico_comparativo():
    with TemporaryDirectory(dir=Path(__file__).parent) as temporal:
        salidas = graficar_comparacion(RESULTADOS, Path(temporal))
        assert {salida.name for salida in salidas} == {
            'convergencia_tai20_5.pdf',
            'convergencia_tai50_10.pdf',
            'convergencia_tai100_10.pdf',
        }
        assert all(salida.stat().st_size > 0 for salida in salidas)



probar_lectura_taillard()
probar_makespan_calculado_a_mano()
probar_secuencias_y_rpd()
probar_heuristica()
probar_feromonas()
probar_construccion()
probar_mejor_solucion()
probar_algoritmo()
probar_operadores_ag()
probar_ag_completo()
probar_busqueda_local()
probar_am_completo()
probar_operadores_eo()
probar_eo_completo()
probar_comparacion_main()
probar_grafico_comparativo()
print('Pruebas correctas.')
