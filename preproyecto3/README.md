# Pre-Proyecto 3: Optimización Extremal (τ-EO) para la Mochila 0/1

Implementación modular de Extremal Optimization (τ-EO) para el Problema de la Mochila 0/1 (KP), usando Python 3, NumPy, Pandas, `sys` y `time`.

## Estructura

- `eo.py`: algoritmo principal y CLI. Lee el flujo de τ-EO en pocas líneas.
- `operadores.py`: operadores del ecosistema (lectura, fitness, ranking con ley de potencia, bit-flip, evaluación, guardado CSV).
- `tests.py`: pruebas unitarias (`unittest`).
- `experimentos.py`: barrido de τ y semillas que genera `resultados/experimentos.csv`.
- `graficar.py`: gráficos de convergencia a partir de un CSV.
- `datos/`: instancias Pisinger `knapPI` (100 instancias por archivo; se usa la 1 por defecto).
- `resultados/`, `graficos/`: salidas de los experimentos.

## Ejecución

```
python eo.py datos/knapPI_1_50_1000.csv 1000 1.4 12345 resultados/resultados.csv
python eo.py datos/knapPI_1_5000_1000.csv 1000 1.4 12345 resultados/res_5000.csv --indice 2
```

Parámetros, en orden: archivo de instancias, número de iteraciones, τ (> 0), semilla y archivo CSV de salida. La opción `--indice N` elige la instancia dentro del archivo (por defecto 1).

El CSV tiene una fila por iteración con: instancia, semilla, τ, óptimo conocido, tiempo total, beneficio y peso actuales con su validez, y mejor beneficio, peso y validez encontrados.

Para reproducir los experimentos y los gráficos:

```
python tests.py
python experimentos.py
python graficar.py resultados/experimentos.csv
```

Requiere `numpy`, `pandas` y `matplotlib` (solo para `graficar.py`).

## Decisiones de diseño

- **Estado inicial:** se llena la mochila en orden aleatorio según la semilla. Siempre es factible y la semilla sí influye en el punto de partida.
- **Fitness individual:** cada elemento tiene `q_i` en [0, 1], la posición de `v_i/w_i` entre todos los elementos. Dentro de la mochila `q_i` es la eficiencia; fuera, `1 - eficiencia`. Valores bajos indican especies mal adaptadas. Si la mochila excede la capacidad, todo elemento dentro cae por debajo de los de fuera, de modo que el algoritmo los retira.
- **Selección:** las especies se ordenan de peor (k = 1) a mejor (k = N) y se elige la de rango k con probabilidad ∝ k^(−τ).
- **Reemplazo:** bit-flip del elemento elegido, aceptado siempre.
- **Mejor solución:** se guarda solo la mejor solución factible vista hasta el momento.

Esta versión usa normalización por rango en lugar de dividir por el máximo de `v/w`. Con la división por el máximo, las instancias con eficiencias muy dispersas (por ejemplo, `knapPI_1_5000`, con `v/w` hasta 651) colapsaban el estado hacia la mochila vacía.

## Resultados resumidos (semillas 1–5, 1000 iteraciones)

| Instancia | τ | Mejor medio | Óptimo | Brecha media |
|---|---|---|---|---|
| knapPI_1_50 | 1.0 | 8306 | 8373 | 0.8 % |
| knapPI_1_50 | 1.4 / 2.0 / 3.0 | 8373 | 8373 | 0 % |
| knapPI_1_500 | 1.0 | 22904 | 28857 | 20.6 % |
| knapPI_1_500 | 1.4 / 3.0 | 28834 | 28857 | 0.08 % |
| knapPI_1_500 | 2.0 | 28843 | 28857 | 0.05 % |
| knapPI_1_5000 | 1.0 | 229800 | 276457 | 16.9 % |
| knapPI_1_5000 | 1.4 / 2.0 / 3.0 | ≈ 276 456 | 276457 | < 0.01 % |

Con τ = 1.0 la búsqueda es casi aleatoria y avanza lento. Con τ ≥ 1.4 llega al óptimo o muy cerca en 1000 iteraciones, y en instancias grandes alcanza el óptimo hacia las iteraciones 450–600 (ver `graficos/`). El tiempo de ejecución crece aproximadamente de forma lineal con N: alrededor de 0.05 s para N = 500 y 0.5 s para N = 5000.

## Transferencia a PFSP (preguntas de la actividad)

1. **Componentes que se mantienen:** el bucle principal de `eo.py` (evaluar fitness, seleccionar por τ, reemplazar, evaluar y actualizar la mejor solución), el ranking con ley de potencia y el control de semilla y tiempo.
2. **Funciones que cambian:** la representación inicial (una permutación en lugar de un vector binario), el fitness individual, el cálculo de la evaluación (Cmax mediante la recursión C(i, j)) y el operador de reemplazo. Leer la instancia también cambia, porque PFSP usa una matriz de tiempos `p[i][j]`.
3. **Fitness individual en PFSP:** para cada trabajo en su posición, se mide el tiempo ocioso (idle time) o el retraso que produce en las máquinas respecto del Cmax parcial. Un trabajo con mucha espera acumulada queda peor adaptado.
4. **Operador de reemplazo factible:** en lugar de un bit-flip, se usa un movimiento de inserción o intercambio entre posiciones. Como toda permutación es factible, el reemplazo no puede generar soluciones inválidas, y la factibilidad queda garantizada por construcción.
