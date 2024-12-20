import random
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import RX, CNOT_func, I, H, TOFFOLI, X, PAULI_Z
from quantum_simulator.simulator import NQubitSimulator


class Shor9ErrorCorrection:
    def __init__(self):
        """
        Класс для моделирования коррекции ошибок с использованием кода Шора.
        """
        self.simulator_class = NQubitSimulator

    def shor9(self, P=0.05, debug=False):
        """
        Выполняет коррекцию ошибок с использованием кода Шора.

        :param P: Вероятность ошибки на каждом кубите.
        :param debug: Если True, выводит отладочную информацию.
        :return: Кортеж (успех, количество ошибок).
        """
        sim = self.simulator_class(9)
        sim.apply_single_qubit_gate(RX(np.pi / 5), 0)
        inital_state = sim.get_qubit_state_raw(0)

        # Кодирование
        CNOT_03 = CNOT_func(9, 0, 3)
        sim.apply_n_qubit_gate(CNOT_03)
        CNOT_06 = CNOT_func(9, 0, 6)
        sim.apply_n_qubit_gate(CNOT_06)
        sim.apply_n_gates(H, I, I, H, I, I, H, I, I)

        CNOT_01 = CNOT_func(9, 0, 1)
        CNOT_34 = CNOT_func(9, 3, 4)
        CNOT_67 = CNOT_func(9, 6, 7)
        S4_operator = np.dot(np.dot(CNOT_01, CNOT_34), CNOT_67)
        sim.apply_n_qubit_gate(S4_operator)

        CNOT_02 = CNOT_func(9, 0, 2)
        CNOT_35 = CNOT_func(9, 3, 5)
        CNOT_68 = CNOT_func(9, 6, 8)
        S5_operator = np.dot(np.dot(CNOT_02, CNOT_35), CNOT_68)
        sim.apply_n_qubit_gate(S5_operator)

        # Внесение ошибок
        error_count = 0
        for idx in range(sim.dimension):
            if random.random() <= P:
                error_count += 1
                sim.apply_single_qubit_gate(X if random.random() > 0.5 else PAULI_Z, idx)

        # Декодирование
        sim.apply_n_qubit_gate(S4_operator)
        sim.apply_n_qubit_gate(S5_operator)

        TOFFOLI_120 = TOFFOLI(9, [1, 2], 0)
        TOFFOLI_453 = TOFFOLI(9, [4, 5], 3)
        TOFFOLI_876 = TOFFOLI(9, [8, 7], 6)
        S8_operator = np.dot(np.dot(TOFFOLI_120, TOFFOLI_453), TOFFOLI_876)
        sim.apply_n_qubit_gate(S8_operator)

        sim.apply_n_gates(H, I, I, H, I, I, H, I, I)
        sim.apply_n_qubit_gate(CNOT_03)
        sim.apply_n_qubit_gate(CNOT_06)
        TOFFOLI_360 = TOFFOLI(9, [3, 6], 0)
        sim.apply_n_qubit_gate(TOFFOLI_360)

        finite_state = sim.get_qubit_state_raw(0)
        if np.isclose(finite_state['|0>'], inital_state['|0>'], 0.01) and \
                np.isclose(finite_state['|1>'], inital_state['|1>'], 0.01):
            return True, error_count
        else:
            return False, error_count

    def no_correction(self, P=0.05):
        """
        Симуляция без использования коррекции ошибок.

        :param P: Вероятность ошибки на каждом кубите.
        :return: Успех (True/False).
        """
        sim = self.simulator_class(1)
        sim.apply_single_qubit_gate(RX(np.pi / 5), 0)
        inital_state = sim.get_qubit_state_raw(0)

        if random.random() <= P:
            sim.apply_single_qubit_gate(X, 0)

        finite_state = sim.get_qubit_state_raw(0)
        return np.isclose(finite_state['|0>'], inital_state['|0>'], 0.01) and \
            np.isclose(finite_state['|1>'], inital_state['|1>'], 0.01)

    @staticmethod
    def theory(p):
        """
        Теоретическое значение вероятности ошибки.

        :param p: Вероятность ошибки.
        :return: Теоретическая вероятность ошибки.
        """
        return 1 - (1 + 8 * p) * (1 - p) ** 8

    def plot_simulation(self, p_values, total_rounds=100):
        """
        Построение графика зависимости вероятности ошибки от P.

        :param p_values: Список вероятностей ошибки.
        :param total_rounds: Количество итераций для каждого значения P.
        """
        p_e_corr_list = []
        p_e_no_corr_list = []
        theoretical_p_e_list = []

        for P in p_values:
            print(f"Simulating for p = {P:.2f}")

            # С коррекцией
            correct_count = sum(not self.shor9(P=P)[0] for _ in range(total_rounds))
            p_e_corr_list.append(correct_count / total_rounds)

            # Без коррекции
            correct_count_nc = sum(not self.no_correction(P=P) for _ in range(total_rounds))
            p_e_no_corr_list.append(correct_count_nc / total_rounds)

            # Теоретическое значение
            theoretical_p_e_list.append(self.theory(P))

        # Построение графиков
        plt.plot(p_values, p_e_corr_list, label='С коррекцией', linestyle='-', color='b', alpha=0.7)
        plt.plot(p_values, p_e_no_corr_list, label='Без коррекции', linestyle='-', color='r', alpha=0.7)

        p_values_fine = np.linspace(0, max(p_values), 100)
        theory_values = [self.theory(p) for p in p_values_fine]
        plt.plot(p_values_fine, theory_values, label='Теоретическое значение (с коррекцией)', color='green', linestyle='--')
        plt.plot(p_values_fine, p_values_fine, label='Теоретическое значение (без коррекции)', color='black', linestyle='--')

        plt.xlabel('Вероятность ошибки P')
        plt.ylabel('Общая вероятность ошибки P_e')
        plt.title('Зависимость вероятности ошибки от P')
        plt.grid(True)
        plt.legend()
        plt.show()
