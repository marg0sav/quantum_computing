import numpy as np
import matplotlib.pyplot as plt
from quantum_simulator.simulator import NQubitSimulator
from utils.constants import X, CNOT_func

class XErrorCorrectionSimulator:
    def __init__(self, simulator_class=NQubitSimulator, n_qubits=5):
        """
        Инициализация симулятора для коррекции X-ошибок.
        :param simulator_class: Класс симулятора (например, NQubitSimulator).
        :param n_qubits: Общее количество кубитов в симуляторе.
        """
        self.simulator_class = simulator_class
        self.n_qubits = n_qubits

    def encode(self, simulator, data_qubit_indices):
        simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[1]))
        simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[2]))

    def decode(self, simulator, data_qubit_indices):
        simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[1]))
        simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[2]))

    def introduce_errors(self, simulator, data_qubit_indices, p):
        for qubit in data_qubit_indices:
            if np.random.random() < p:
                simulator.apply_single_qubit_gate(X, qubit)

    def measure_syndrome(self, simulator, data_qubit_indices, ancilla_qubit_indices):
        simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], ancilla_qubit_indices[0]))
        simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[1], ancilla_qubit_indices[0]))
        simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], ancilla_qubit_indices[1]))
        simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[2], ancilla_qubit_indices[1]))
        s1 = simulator.measure(ancilla_qubit_indices[0])
        s2 = simulator.measure(ancilla_qubit_indices[1])
        return (s1, s2)

    def correct_errors(self, simulator, syndrome, data_qubit_indices):
        s1, s2 = syndrome
        if (s1, s2) == (1, 0):
            simulator.apply_single_qubit_gate(X, data_qubit_indices[1])
        elif (s1, s2) == (0, 1):
            simulator.apply_single_qubit_gate(X, data_qubit_indices[2])
        elif (s1, s2) == (1, 1):
            simulator.apply_single_qubit_gate(X, data_qubit_indices[0])

    def simulate_correction(self, p, n_runs):
        error_after_correction = 0
        data_qubit_indices = [0, 1, 2]
        ancilla_qubit_indices = [3, 4]

        for _ in range(n_runs):
            simulator = self.simulator_class(self.n_qubits)
            self.encode(simulator, data_qubit_indices)
            self.introduce_errors(simulator, data_qubit_indices, p)
            syndrome = self.measure_syndrome(simulator, data_qubit_indices, ancilla_qubit_indices)
            self.correct_errors(simulator, syndrome, data_qubit_indices)
            self.decode(simulator, data_qubit_indices)
            decoded = simulator.measure(0)
            if decoded != 0:
                error_after_correction += 1

        return error_after_correction / n_runs

    def simulate_no_correction(self, p, n_runs):
        error_without_correction = 0
        data_qubit_indices = [0, 1, 2]

        for _ in range(n_runs):
            simulator = self.simulator_class(self.n_qubits)
            self.encode(simulator, data_qubit_indices)
            self.introduce_errors(simulator, data_qubit_indices, p)
            self.decode(simulator, data_qubit_indices)
            decoded = simulator.measure(0)
            if decoded != 0:
                error_without_correction += 1

        return error_without_correction / n_runs

    @staticmethod
    def theoretical_p_e(p):
        return 3 * p ** 2 - 2*p ** 3

    def plot_simulation(self, p_values, simulated_p_e, p_e_no_correction, theoretical_p_e_values):
        plt.figure(figsize=(10, 6))

        # Практическая симуляция с коррекцией (сплошная линия)
        plt.plot(p_values, simulated_p_e, '-', label='Симуляция с коррекцией', color='blue')

        # Теоретическая вероятность (пунктирная линия)
        plt.plot(p_values, theoretical_p_e_values, '--', label='Теоретическая вероятность', color='green')

        # Практическая симуляция без коррекции (сплошная линия)
        plt.plot(p_values, p_e_no_correction, '-', label='Симуляция без коррекции', color='red')

        # Прямая вероятность p_e = p (пунктирная линия)
        plt.plot(p_values, p_values, '--', label='Вероятность без коррекции (p_e = p)', color='black')

        plt.title("X-ошибки")
        plt.xlabel("Вероятность битовой ошибки на кубите p")
        plt.ylabel("Вероятность ошибки системы p_e")
        plt.legend()
        plt.grid(True)
        plt.show()
