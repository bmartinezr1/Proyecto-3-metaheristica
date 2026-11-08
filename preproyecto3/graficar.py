"""Gráfico de convergencia de tau-EO a partir del CSV de resultados.

Uso: python graficar.py resultados/experimentos.csv [--salida graficos/convergencia.png]
Dibuja, para cada tau, el mejor beneficio medio (sobre semillas) por iteración.
Una figura por instancia.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Gráfico de convergencia de tau-EO")
    p.add_argument("csv", help="CSV generado por eo.py o experimentos.py")
    p.add_argument("--salida", default="graficos", help="carpeta donde se guardan los gráficos")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except OSError as e:
        print(f"Error al leer {args.csv}: {e}", file=sys.stderr)
        sys.exit(1)

    carpeta = args.salida
    os.makedirs(carpeta, exist_ok=True)

    for instancia, grupo in df.groupby("instancia"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for tau, sub in grupo.groupby("tau"):
            media = sub.groupby("iteracion")["mejor_beneficio"].mean()
            ax.plot(media.index, media.values, label=f"τ = {tau}")
        ax.axhline(grupo["z_optimo"].iloc[0], color="k", linestyle="--", linewidth=1, label="óptimo")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Mejor beneficio (media de semillas)")
        ax.set_title(instancia)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        destino = os.path.join(carpeta, f"convergencia_{os.path.splitext(instancia)[0]}.png")
        fig.savefig(destino, dpi=150)
        plt.close(fig)
        print(f"Gráfico guardado: {destino}")


if __name__ == "__main__":
    main()
