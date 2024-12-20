import numpy as np
from typing import List
from utils.constants import X, HN, PAULI_X, PAULI_Z, KET_0, KET_1, H, CNOT, RX
from quantum_simulator.simulator import NQubitSimulator

class QuantumTeleportation:
    def __init__(self, simulator: NQubitSimulator):
        """
        Инициализация алгоритма квантовой телепортации с симулятором.
        """
        self.simulator = simulator

    def prepare_entanglement(self):
        """
        Подготовка запутанных кубитов с использованием гейтов Адамара и CNOT.
        """
        # Применить Адамара ко второму кубиту
        self.simulator.apply_single_qubit_gate(H, 1)

        # Применить CNOT с кубитом 1 в качестве управляющего и кубитом 2 в качестве целевого
        cnot_gate = CNOT(3, 1, 2)  # используем функцию CNOT для 3 кубитов
        self.simulator.apply_n_qubit_gate(cnot_gate)

    def apply_teleportation(self):
        """
        Запуск протокола квантовой телепортации.
        """
        # Сброс симулятора
        self.simulator.reset()

        # Применить поворотный гейт к первому кубиту для изменения его состояния
        self.simulator.apply_single_qubit_gate(RX(np.pi / 4), 0)

        # Подготовка запутанных кубитов между кубитом 1 и кубитом 2
        self.prepare_entanglement()

        # Вывести текущее состояние кубитов
        for i in range(self.simulator.dimension):
            print(f'Состояние {i}: {self.simulator.get_qubit_state(i)}')

        # Применить CNOT с кубитом 0 в качестве управляющего и кубитом 1 в качестве целевого
        cnot_gate = CNOT(3, 0, 1)
        self.simulator.apply_n_qubit_gate(cnot_gate)

        # Применить Адамара к первому кубиту
        self.simulator.apply_single_qubit_gate(H, 0)

        # Измерение кубитов 0 и 1
        measurement_results = self.simulator.measure_multiple_qubits([0, 1])
        print(f'Измерения: {measurement_results}')
        print('Процесс телепортации...')

        # Применение контролируемых гейтов на основе результатов измерений
        # Если кубит 1 был измерен как 1, применить гейт X к кубиту 2
        self.simulator.controlled_by_measurement(np.eye(2), PAULI_X, measurement_results[1], 2)

        # Если кубит 0 был измерен как 1, применить гейт Z к кубиту 2
        self.simulator.controlled_by_measurement(np.eye(2), PAULI_Z, measurement_results[0], 2)

        print('Результат после телепортации:')
        # Вывод финального состояния кубитов
        for i in range(self.simulator.dimension):
            print(f'Состояние {i}: {self.simulator.get_qubit_state(i)}')
