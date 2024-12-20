import numpy as np
import itertools
from typing import List
from utils.constants import X, HN
from quantum_simulator.simulator import NQubitSimulator

class BernsteinVaziraniAlgorithm:
    def __init__(self, simulator: NQubitSimulator):
        """Инициализация алгоритма с симулятором."""
        self.simulator = simulator

    def apply_oracle(self, oracle: np.ndarray):
        """Применяет оракула к квантовому симулятору."""
        self.simulator.apply_n_qubit_gate(oracle)

    def run(self, oracle: np.ndarray) -> List[bool]:
        """Запуск алгоритма Бернштейна-Вазирани с данным оракулом."""
        # Шаг 1: Сброс квантового симулятора в состояние |0...0>
        self.simulator.reset()

        # Шаг 2: Применить гейт X к последнему кубиту, чтобы подготовить его в состояние |1>
        self.simulator.apply_single_qubit_gate(X, self.simulator.dimension - 1)

        # Шаг 3: Применить гейты Адамара ко всем кубитам
        self.simulator.apply_n_qubit_gate(HN(self.simulator.dimension))

        # Шаг 4: Применить оракул для задачи Бернштейна-Вазирани
        self.apply_oracle(oracle)

        # Шаг 5: Применить гейты Адамара
        self.simulator.apply_n_qubit_gate(HN(self.simulator.dimension))

        # Шаг 6: Измерить все кубиты, кроме последнего вспомогательного
        measured = []
        for i in range(self.simulator.dimension - 1):
            measured.append(self.simulator.measure(i))

        return measured

    @staticmethod
    def generate_oracle(N: int, s: List[int]) -> np.ndarray:
        """
        Генерация оракула для задачи Бернштейна-Вазирани.

        Аргументы:
        - N: количество переменных.
        - s: скрытая строка, используемая для скалярного произведения в булевой функции.

        Возвращает:
        - Матрица размером 2^(N+1) x 2^(N+1), описывающая оракул.
        """
        # Размер матрицы оракула
        matrix_size = 2 ** (N + 1)

        # Инициализация нулевой матрицы
        oracle_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        # Проход по всем возможным входным данным (x) и вспомогательному биту (y)
        for x in itertools.product([0, 1], repeat=N):
            for y in [0, 1]:  # y - вспомогательный кубит
                # Индекс входа (двоичное представление x, за которым следует y)
                input_index = int(''.join(map(str, x)) + str(y), 2)

                # Вычисление булевой функции f(x) = s · x mod 2 (скалярное произведение)
                f_x = np.dot(s, x) % 2

                # Вычисление нового значения для вспомогательного бита
                output_y = y ^ f_x  # y XOR f(x)

                # Индекс выхода (двоичное представление x, за которым следует новое y)
                output_index = int(''.join(map(str, x)) + str(output_y), 2)

                # Устанавливаем элемент матрицы (input_index, output_index) в 1
                oracle_matrix[input_index][output_index] = 1

        return oracle_matrix
