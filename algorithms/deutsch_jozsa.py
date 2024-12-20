import numpy as np
from quantum_simulator.simulator import NQubitSimulator
from utils.constants import H, X


class DeutschJozsaAlgorithm:
    def __init__(self, simulator: NQubitSimulator):
        """Инициализация алгоритма с симулятором."""
        self.simulator = simulator

    def apply_oracle(self, oracle: np.ndarray):
        """Применяет заданного оракула к системе."""
        self.simulator.apply_n_qubit_gate(oracle)

    def run(self, oracle: np.ndarray) -> str:
        """Запускает алгоритм Дойча-Джозы и возвращает результат."""
        # Сброс симулятора перед началом
        self.simulator.reset()

        # Шаг 1: Установить последний кубит в |1>
        self.simulator.apply_single_qubit_gate(X, self.simulator.dimension - 1)

        # Шаг 2: Применить оператор Адамара ко всем кубитам
        for i in range(self.simulator.dimension):
            self.simulator.apply_single_qubit_gate(H, i)

        # Шаг 3: Применить оракула
        self.apply_oracle(oracle)

        # Шаг 4: Применить оператор Адамара ко всем кубитам, кроме последнего
        for i in range(self.simulator.dimension - 1):
            self.simulator.apply_single_qubit_gate(H, i)

        # Шаг 5: Измерить все входные кубиты
        for i in range(self.simulator.dimension - 1):
            if self.simulator.measure(i) == 1:
                return "Balanced"
        return "Constant"

    @staticmethod
    def generate_oracle(n: int, f) -> np.ndarray:
        """Генерация оракула для алгоритма Дойча-Джозы."""
        size = 2 ** (n + 1)  # Размер матрицы для n входных кубитов + 1 дополнительный
        oracle = np.eye(size, dtype=complex)  # Начинаем с тождественной матрицы

        for x in range(2 ** n):
            inputs = [(x >> i) & 1 for i in range(n)]  # Побитовый разбор входного числа
            output = f(inputs)  # Вызываем функцию f для получения результата
            if output:  # Если результат true, меняем соответствующий элемент оракула
                oracle[x, x] = -1  # Применяем фазовый сдвиг

        return oracle


# Пример булевых функций для генерации оракула
def bool_zero_3(x) -> bool:
    """Функция "0" для трех входных кубитов."""
    return False

def bool_or_3(x) -> bool:
    """Функция "OR" для трех входных кубитов."""
    return x[0] or x[1] or x[2]

def bool_xor_3(x) -> bool:
    """Функция "XOR" для трех входных кубитов."""
    return (x[0] != x[1]) != x[2]  # XOR между тремя входными кубитами