import numpy as np
from quantum_simulator.simulator import TwoQubitSimulator
from utils.constants import H, X

class DeutschAlgorithm:
    def __init__(self, simulator: TwoQubitSimulator):
        """Инициализация алгоритма с симулятором."""
        self.simulator = simulator

    def apply_oracle(self, oracle: np.ndarray):
        """Применяет заданного оракула к системе."""
        self.simulator.apply_two_qubit_gate(oracle)

    def run(self, oracle: np.ndarray) -> str:
        """Запускает алгоритм Дойча и возвращает результат."""
        # Сброс симулятора перед началом
        self.simulator.reset()

        # Шаг 1: Установить первый кубит в |1>
        self.simulator.apply_single_qubit_gate(X, 1)

        # Шаг 2: Применить оператор Адамара ко всем кубитам
        self.simulator.apply_two_qubit_gate(np.kron(H, H))

        # Шаг 3: Применить оракула
        self.apply_oracle(oracle)

        # Шаг 4: Применить оператор Адамара к первому кубиту
        self.simulator.apply_single_qubit_gate(H, 0)

        # Шаг 5: Измерить первый кубит
        result = self.simulator.measure(0)

        # Вернуть результат: Constant или Balanced
        return "Constant" if result == 0 else "Balanced"


DeutschAlgorithm.ORACLE1 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]], dtype=complex)  # f(x) = 0

DeutschAlgorithm.ORACLE2 = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]], dtype=complex)  # f(x) = 1

DeutschAlgorithm.ORACLE3 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]], dtype=complex)  # f(x) = x

DeutschAlgorithm.ORACLE4 = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 0]], dtype=complex)  # f(x) = !x