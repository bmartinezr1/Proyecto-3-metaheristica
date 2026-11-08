"""Pruebas de los operadores y de la ejecución de tau-EO."""
import os
import unittest

import numpy as np

from eo import extremal_optimization
from operadores import (
    actualizar_mejor,
    evaluar_ecosistema,
    evaluar_fitness_especies,
    leer_instancia,
    mejor_inicial,
    probabilidades_tau,
    reemplazar_especie,
    seleccionar_especie_tau,
)

DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "knapPI_1_50_1000.csv")


class TestLectura(unittest.TestCase):
    def test_lee_cabecera_y_primer_elemento(self):
        capacidad, pesos, valores, z = leer_instancia(DATOS, 1)
        self.assertEqual((capacidad, z), (995, 8373))
        self.assertEqual(len(pesos), 50)
        self.assertEqual((pesos[0], valores[0]), (485, 94))  # línea "1,94,485,0"

    def test_indice_fuera_de_rango(self):
        with self.assertRaises(ValueError):
            leer_instancia(DATOS, 1000)


class TestFitness(unittest.TestCase):
    def test_ineficiente_dentro_tiene_menor_fitness(self):
        pesos, valores = np.array([10, 10]), np.array([100, 10])
        q = evaluar_fitness_especies(np.array([1, 1]), pesos, valores, capacidad=100)
        self.assertLess(q[1], q[0])

    def test_sobrepeso_baja_el_fitness_de_los_seleccionados(self):
        pesos, valores = np.array([10, 10]), np.array([100, 100])
        x = np.array([1, 1])
        q_ok = evaluar_fitness_especies(x, pesos, valores, capacidad=20)
        q_pesado = evaluar_fitness_especies(x, pesos, valores, capacidad=10)
        self.assertTrue(np.all(q_pesado < q_ok))


class TestSeleccion(unittest.TestCase):
    def test_probabilidades_siguen_ley_de_potencia(self):
        p = probabilidades_tau(10, 2.0)
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertAlmostEqual(p[0] / p[1], 2.0 ** 2)  # P(1)/P(2) = (2/1)^tau

    def test_tau_alto_elige_la_peor_especie(self):
        q = np.array([0.9, 0.1, 0.5])  # la peor es el índice 1
        rng = np.random.default_rng(0)
        elecciones = [seleccionar_especie_tau(q, 10.0, rng) for _ in range(200)]
        self.assertGreater(elecciones.count(1), 190)


class TestReemplazo(unittest.TestCase):
    def test_bit_flip_cambia_solo_el_elemento_indicado(self):
        x = np.array([0, 1, 0])
        self.assertEqual(reemplazar_especie(x, 1).tolist(), [0, 0, 0])
        self.assertEqual(x.tolist(), [0, 1, 0])  # el original no cambia


class TestMejorSolucion(unittest.TestCase):
    def test_no_guarda_soluciones_invalidas(self):
        pesos, valores = np.array([5, 5, 5]), np.array([1, 1, 1])
        x = np.array([1, 1, 1])  # peso 15 > 10
        mejor = actualizar_mejor(mejor_inicial(3), x, evaluar_ecosistema(x, pesos, valores, 10))
        self.assertEqual(mejor["beneficio"], 0)

    def test_mejor_nunca_empeora_y_siempre_es_valida(self):
        capacidad, pesos, valores, _ = leer_instancia(DATOS, 1)
        mejor, historial = extremal_optimization(pesos, valores, capacidad, 300, 1.4, np.random.default_rng(1))
        beneficios = [h["mejor_beneficio"] for h in historial]
        self.assertEqual(beneficios, sorted(beneficios))
        self.assertTrue(all(h["mejor_valido"] for h in historial))
        self.assertLessEqual(mejor["peso"], capacidad)

    def test_misma_semilla_mismo_resultado(self):
        capacidad, pesos, valores, _ = leer_instancia(DATOS, 1)
        _, h1 = extremal_optimization(pesos, valores, capacidad, 200, 1.4, np.random.default_rng(7))
        _, h2 = extremal_optimization(pesos, valores, capacidad, 200, 1.4, np.random.default_rng(7))
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
