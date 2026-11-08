"""Gráficos a partir de los CSV de AG, AM, ACS y EO."""
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ALGORITMOS = ('AG', 'AM', 'ACS', 'EO')
COLORES = {'AG': '#2563eb', 'AM': '#f59e0b', 'ACS': '#059669', 'EO': '#dc2626'}


def leer_csv(ruta):
    with open(ruta, newline='', encoding='utf-8') as archivo:
        return list(csv.DictReader(archivo))


def graficar_convergencia(filas, salida):
    iteraciones = [int(fila['iteracion']) for fila in filas]
    plt.figure(figsize=(7, 4))
    plt.plot(iteraciones, [int(fila['makespan_iteracion']) for fila in filas],
             alpha=0.4, label='Mejor de la iteración')
    plt.plot(iteraciones, [int(fila['mejor_makespan']) for fila in filas], label='Mejor global')
    plt.axhline(int(filas[0]['mejor_conocido']), color='gray', linestyle='--', label='Mejor conocido')
    plt.xlabel('Iteración')
    plt.ylabel('Makespan')
    algoritmo = filas[0].get('algoritmo') or 'ACS'
    plt.title(f"Convergencia de {algoritmo} - {filas[0]['instancia']} (semilla {filas[0]['semilla']})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida)
    plt.close()


def evaluaciones_por_iteracion(fila):
    """Soluciones evaluadas en cada iteración: población u hormigas; EO evalúa una."""
    return int(fila['poblacion'] or fila['hormigas'] or 1)


def graficar_comparacion(carpeta, salida):
    """Genera una curva promedio por instancia desde una carpeta de ejecuciones.

    Si entre los algoritmos está EO, el eje x son las evaluaciones de makespan,
    porque una iteración de EO no es comparable con una de AG, AM o ACS.
    """
    grupos = {}
    for archivo in sorted(carpeta.glob('*_s*.csv')):
        filas = leer_csv(archivo)
        if not filas or 'iteracion' not in filas[0]:
            raise ValueError(f'CSV de ejecución inválido: {archivo.name}')
        instancia = Path(filas[0]['instancia']).stem
        algoritmo = filas[0]['algoritmo']
        grupos.setdefault((instancia, algoritmo), []).append(filas)

    if not grupos:
        raise ValueError('La carpeta no contiene CSV de ejecuciones.')

    salida.mkdir(parents=True, exist_ok=True)
    algoritmos = [a for a in ALGORITMOS if any(clave[1] == a for clave in grupos)]
    por_evaluaciones = 'EO' in algoritmos
    generados = []
    for instancia in sorted({clave[0] for clave in grupos}):
        faltantes = [algoritmo for algoritmo in algoritmos if (instancia, algoritmo) not in grupos]
        if faltantes:
            raise ValueError(f'Faltan ejecuciones de {", ".join(faltantes)} para {instancia}.')

        cantidades = {len(grupos[(instancia, algoritmo)]) for algoritmo in algoritmos}
        if len(cantidades) != 1:
            raise ValueError(f'Los algoritmos no tienen igual cantidad de semillas para {instancia}.')

        plt.figure(figsize=(7, 4))
        referencia = None
        for algoritmo in algoritmos:
            ejecuciones = grupos[(instancia, algoritmo)]
            iteraciones = [int(fila['iteracion']) for fila in ejecuciones[0]]
            if any([int(fila['iteracion']) for fila in filas] != iteraciones for filas in ejecuciones):
                raise ValueError(f'Las ejecuciones de {algoritmo} no comparten iteraciones en {instancia}.')
            curvas = [[int(fila['mejor_makespan']) for fila in filas] for filas in ejecuciones]
            eje_x = iteraciones
            if por_evaluaciones:
                eje_x = np.array(iteraciones) * evaluaciones_por_iteracion(ejecuciones[0][0])
            plt.plot(eje_x, np.mean(curvas, axis=0), label=algoritmo,
                     color=COLORES[algoritmo])
            referencia = int(ejecuciones[0][0]['mejor_conocido'])

        semillas = cantidades.pop()
        plt.axhline(referencia, color='#4b5563', linestyle='--',
                    label=f'Mejor conocido ({referencia})')
        plt.xlabel('Evaluaciones de makespan' if por_evaluaciones else 'Iteración')
        plt.ylabel('Mejor makespan acumulado')
        plt.title(f'Convergencia comparativa - {instancia}')
        plt.legend(ncol=4, fontsize=8)
        plt.grid(alpha=0.3)
        plt.figtext(0.5, 0.01, f'Curvas promedio de {semillas} semillas',
                    ha='center', fontsize=7, color='#4b5563')
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        destino = salida / f'convergencia_{instancia}.pdf'
        plt.savefig(destino)
        plt.close()
        generados.append(destino)
    return generados


def graficar_cajas(carpeta, salida):
    """Diagrama de cajas del RPD final por algoritmo, un panel por instancia (usa resumen.csv)."""
    filas = leer_csv(carpeta / 'resumen.csv')
    # Ordena las instancias por cantidad de trabajos (el primer número del nombre).
    instancias = sorted({fila['instancia'] for fila in filas}, key=lambda i: int(re.search(r'\d+', i).group()))
    algoritmos = [a for a in ALGORITMOS if any(fila['algoritmo'] == a for fila in filas)]

    figura, ejes = plt.subplots(1, len(instancias), figsize=(4 * len(instancias), 4), squeeze=False)
    for eje, instancia in zip(ejes[0], instancias):
        datos = [[float(f['rpd']) for f in filas if f['instancia'] == instancia and f['algoritmo'] == a]
                 for a in algoritmos]
        cajas = eje.boxplot(datos, patch_artist=True)
        for caja, algoritmo in zip(cajas['boxes'], algoritmos):
            caja.set_facecolor(COLORES[algoritmo])
            caja.set_alpha(0.6)
        eje.set_xticks(range(1, len(algoritmos) + 1), algoritmos)
        eje.set_title(Path(instancia).stem)
        eje.set_ylabel('RPD final (%)')
        eje.grid(alpha=0.3, axis='y')
    figura.tight_layout()
    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / 'cajas_rpd.pdf'
    figura.savefig(destino)
    plt.close(figura)
    return destino


def graficar_experimentos(filas, salida):
    instancias = list(dict.fromkeys(fila['instancia'] for fila in filas))
    factores = list(dict.fromkeys(fila['factor'] for fila in filas))
    figura, ejes = plt.subplots(2, 3, figsize=(13, 7))

    for eje, factor in zip(ejes.flat, factores):
        for instancia in instancias:
            grupo = [f for f in filas if f['factor'] == factor and f['instancia'] == instancia]
            valores = sorted({float(f['valor']) for f in grupo})
            rpd = [[float(f['rpd']) for f in grupo if float(f['valor']) == v] for v in valores]
            eje.errorbar(valores, [np.mean(grupo) for grupo in rpd],
                         yerr=[np.std(grupo) for grupo in rpd],
                         marker='o', capsize=3, label=instancia)
        eje.set_xlabel(factor)
        eje.set_ylabel('RPD medio (%)')
        eje.grid(alpha=0.3)

    eje = ejes.flat[len(factores)]
    for instancia in instancias:
        grupo = [f for f in filas if f['instancia'] == instancia]
        eje.scatter([float(f['tiempo_segundos']) for f in grupo],
                    [float(f['rpd']) for f in grupo], s=10, alpha=0.6, label=instancia)
    eje.set_xlabel('Tiempo (s)')
    eje.set_ylabel('RPD (%)')
    eje.set_title('Calidad vs. tiempo')
    eje.grid(alpha=0.3)
    eje.legend()

    figura.suptitle('Sensibilidad de ACS a sus parámetros (media ± desviación en 5 semillas)')
    figura.tight_layout()
    figura.savefig(salida)
    plt.close(figura)


def main():
    parser = argparse.ArgumentParser(description='Grafica resultados de AG, AM, ACS o EO para PFSP.')
    parser.add_argument(
        'resultados',
        help='CSV de una ejecución o carpeta con los CSV de la comparación.',
    )
    parser.add_argument(
        'salida',
        nargs='?',
        help='PDF de salida, o carpeta de salida si resultados es una carpeta.',
    )
    argumentos = parser.parse_args()

    entrada = Path(argumentos.resultados)
    if entrada.is_dir():
        salida = Path(argumentos.salida or Path(__file__).parent / 'graficos')
        try:
            generados = graficar_comparacion(entrada, salida)
            if (entrada / 'resumen.csv').exists():
                generados.append(graficar_cajas(entrada, salida))
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        for generado in generados:
            print('Gráfico guardado en:', generado)
        return

    try:
        filas = leer_csv(entrada)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if not filas:
        parser.error('El archivo de resultados no contiene datos.')

    salida = Path(argumentos.salida or Path(__file__).parent / 'graficos' / f'{entrada.stem}.pdf')
    salida.parent.mkdir(parents=True, exist_ok=True)
    if 'iteracion' in filas[0]:
        graficar_convergencia(filas, salida)
    elif 'factor' in filas[0]:
        graficar_experimentos(filas, salida)
    else:
        parser.error('El CSV no contiene iteraciones ni experimentos reconocibles.')
    print('Gráfico guardado en:', salida)


if __name__ == '__main__':
    main()
