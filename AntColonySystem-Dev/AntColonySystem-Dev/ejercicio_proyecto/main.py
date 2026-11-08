"""Compara AG, AM, ACS y EO sobre instancias PFSP con el mismo presupuesto de evaluaciones.

Ejemplo: python main.py --evaluaciones 30000 --semillas 1 2 3 4 5
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from acs import ParametrosACS, ejecutar_acs
from ag import ejecutar_ag
from am import ejecutar_am
from eo import ejecutar_eo
from operadores import guardar_resultados, leer_instancia

ALGORITMOS = ('AG', 'AM', 'ACS', 'EO')
TAMANO = 50  # individuos (AG, AM) u hormigas (ACS); AG, AM y ACS evalúan TAMANO soluciones por iteración
INSTANCIAS = ['datos/tai20_5.txt', 'datos/tai50_10.txt', 'datos/tai100_10.txt']

# Parámetros por cantidad de trabajos. AG, AM y ACS: ajuste del Proyecto 2.
# AG: (prob_cruce, prob_mutacion). AM: (prob_cruce, prob_mutacion, frecuencia_bl).
# ACS: (alpha, beta, q0). EO: tau (barrido corto 1.2-3.0 con semillas 101-103: sin diferencias claras; 1.6 fue el mejor en 2 de 3 instancias).
PARAMETROS = {
    20: {'AG': (0.86, 0.16), 'AM': (0.85, 0.15, 100), 'ACS': (0.1, 2.0, 0.9), 'EO': 1.6},
    50: {'AG': (0.70, 0.20), 'AM': (0.85, 0.15, 50), 'ACS': (0.1, 2.0, 0.9), 'EO': 1.6},
    100: {'AG': (0.70, 0.20), 'AM': (0.85, 0.15, 50), 'ACS': (0.1, 2.0, 0.9), 'EO': 1.6},
}


def ejecutar_algoritmo(algoritmo, tiempos, semilla, evaluaciones, parametros):
    """Ejecuta un algoritmo con el presupuesto dado y devuelve (secuencia, makespan, historial, parametros CSV)."""
    iteraciones = evaluaciones // TAMANO
    if algoritmo == 'AG':
        p_cruce, p_mutacion = parametros['AG']
        resultado = ejecutar_ag(tiempos, semilla, TAMANO, iteraciones, p_cruce, p_mutacion)
        csv_parametros = {'iteraciones': iteraciones, 'poblacion': TAMANO,
                          'prob_cruce': p_cruce, 'prob_mutacion': p_mutacion}
    elif algoritmo == 'AM':
        p_cruce, p_mutacion, frecuencia = parametros['AM']
        resultado = ejecutar_am(tiempos, semilla, TAMANO, iteraciones, p_cruce, p_mutacion, frecuencia)
        csv_parametros = {'iteraciones': iteraciones, 'poblacion': TAMANO, 'prob_cruce': p_cruce,
                          'prob_mutacion': p_mutacion, 'frecuencia_bl': frecuencia}
    elif algoritmo == 'ACS':
        alpha, beta, q0 = parametros['ACS']
        resultado = ejecutar_acs(tiempos, ParametrosACS(semilla, TAMANO, iteraciones, alpha, beta, q0))
        csv_parametros = {'iteraciones': iteraciones, 'hormigas': TAMANO, 'alpha': alpha, 'beta': beta, 'q0': q0}
    else:
        # EO evalúa una solución por iteración. Se registra una fila cada TAMANO
        # iteraciones para que su CSV tenga el mismo largo que los de AG, AM y ACS.
        evaluaciones = iteraciones * TAMANO
        secuencia, makespan, historial = ejecutar_eo(tiempos, semilla, evaluaciones, parametros['EO'])
        resultado = (secuencia, makespan, historial[TAMANO - 1::TAMANO])
        csv_parametros = {'iteraciones': evaluaciones, 'tau': parametros['EO']}
    return (*resultado, csv_parametros)


def resumir(carpeta):
    """Lee la última fila de cada CSV de ejecución y guarda resumen.csv; devuelve sus filas."""
    filas = []
    for archivo in sorted(carpeta.glob('*_s*.csv')):
        with archivo.open(newline='', encoding='utf-8') as entrada:
            ultima = list(csv.DictReader(entrada))[-1]
        filas.append({
            'instancia': ultima['instancia'],
            'algoritmo': ultima['algoritmo'],
            'semilla': ultima['semilla'],
            'makespan': ultima['mejor_makespan'],
            'rpd': ultima['rpd'],
            'tiempo_segundos': ultima['tiempo_segundos'],
        })
    with (carpeta / 'resumen.csv').open('w', newline='', encoding='utf-8') as salida:
        escritor = csv.DictWriter(salida, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    return filas


def imprimir_tabla(filas):
    grupos = defaultdict(list)
    for fila in filas:
        grupos[(fila['instancia'], fila['algoritmo'])].append(fila)
    print(f"\n{'Instancia':<16}{'Alg.':<6}{'RPD medio':>10}{'RPD mejor':>10}{'Desv.':>8}{'CPU medio (s)':>15}{'Corridas':>10}")
    for (instancia, algoritmo), grupo in sorted(
        grupos.items(), key=lambda item: (item[0][0], ALGORITMOS.index(item[0][1]))
    ):
        rpd = np.array([float(fila['rpd']) for fila in grupo])
        cpu = np.mean([float(fila['tiempo_segundos']) for fila in grupo])
        print(f'{instancia:<16}{algoritmo:<6}{rpd.mean():>10.2f}{rpd.min():>10.2f}{rpd.std():>8.2f}{cpu:>15.1f}{len(grupo):>10}')


def crear_parser():
    parser = argparse.ArgumentParser(
        description='Compara AG, AM, ACS y EO en PFSP con el mismo presupuesto de evaluaciones.'
    )
    parser.add_argument('--instancias', nargs='+', default=INSTANCIAS, help='Archivos de instancias de Taillard.')
    parser.add_argument('--semillas', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='Semillas a ejecutar.')
    parser.add_argument('--evaluaciones', type=int, default=30000,
                        help='Presupuesto de evaluaciones de makespan por algoritmo (múltiplo de %d).' % TAMANO)
    parser.add_argument('--algoritmos', nargs='+', choices=ALGORITMOS, default=list(ALGORITMOS),
                        help='Algoritmos a ejecutar.')
    parser.add_argument('--salida', default='resultados/proyecto3', help='Carpeta de salida de los CSV.')
    return parser


def main():
    parser = crear_parser()
    argumentos = parser.parse_args()
    if argumentos.evaluaciones < TAMANO:
        parser.error(f'El presupuesto debe ser al menos {TAMANO} evaluaciones.')

    carpeta = Path(argumentos.salida)
    try:
        instancias = [(Path(ruta), *leer_instancia(ruta)) for ruta in argumentos.instancias]
        for ruta, tiempos, _ in instancias:
            if tiempos.shape[0] not in PARAMETROS:
                raise ValueError(f'No hay parámetros definidos para {tiempos.shape[0]} trabajos ({ruta.name}).')
    except (OSError, ValueError) as error:
        parser.error(str(error))

    total = len(instancias) * len(argumentos.semillas) * len(argumentos.algoritmos)
    hecho = 0
    for ruta, tiempos, mejor_conocido in instancias:
        for semilla in argumentos.semillas:
            for algoritmo in argumentos.algoritmos:
                secuencia, makespan, historial, csv_parametros = ejecutar_algoritmo(
                    algoritmo, tiempos, semilla, argumentos.evaluaciones, PARAMETROS[tiempos.shape[0]]
                )
                guardar_resultados(
                    carpeta / f'{algoritmo}_{ruta.stem}_s{semilla}.csv',
                    algoritmo, ruta, semilla, csv_parametros, mejor_conocido, historial,
                )
                hecho += 1
                rpd = 100 * (makespan - mejor_conocido) / mejor_conocido
                print(f"[{hecho}/{total}] {algoritmo:<3} {ruta.stem} semilla={semilla} "
                      f"makespan={makespan} RPD={rpd:.2f}% CPU={historial[-1]['tiempo_segundos']:.1f}s")

    imprimir_tabla(resumir(carpeta))
    print('\nResultados guardados en:', carpeta)


if __name__ == '__main__':
    main()
