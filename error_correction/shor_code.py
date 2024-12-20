import random
import numpy as np

from utils.constants import RX, CNOT_func, I, H, TOFFOLI, X, PAULI_Z
from quantum_simulator.simulator import NQubitSimulator


def shor9(P=0.05, debug=False):
    # Prepare system:
    # 9 qubits. (One of them target PHI)
    sim = NQubitSimulator(9)
    sim.apply_single_qubit_gate(RX(np.pi / 5), 0)
    inital_state = sim.get_qubit_state_raw(0)

    if debug:
        print('Initial system state:')
        # Print qubit PHI state
        for i in range(sim.dimension):
            print(f'State {i}: {sim.get_qubit_state(i)}')

    # Coding.
    # S1
    CNOT_03 = CNOT_func(9, 0, 3)
    sim.apply_n_qubit_gate(CNOT_03)

    # S2
    CNOT_06 = CNOT_func(9, 0, 6)
    sim.apply_n_qubit_gate(CNOT_06)

    # S3
    sim.apply_n_gates(H, I, I, H, I, I, H, I, I)

    # S4
    CNOT_01 = CNOT_func(N=9, c=0, t=1)
    CNOT_34 = CNOT_func(N=9, c=3, t=4)
    CNOT_67 = CNOT_func(N=9, c=6, t=7)
    S4_operator = np.dot(np.dot(CNOT_01, CNOT_34), CNOT_67)
    sim.apply_n_qubit_gate(S4_operator)

    # S5
    CNOT_02 = CNOT_func(N=9, c=0, t=2)
    CNOT_35 = CNOT_func(N=9, c=3, t=5)
    CNOT_68 = CNOT_func(N=9, c=6, t=8)
    S5_operator = np.dot(np.dot(CNOT_02, CNOT_35), CNOT_68)
    sim.apply_n_qubit_gate(S5_operator)

    if debug:
        print('Finishing coding part:')
        for i in range(sim.dimension):
            print(f'State {i}: {sim.get_qubit_state(i)}')

    error_count = 0
    # Random inverse qubit with PAULI [X or Z] (Simulate Error)
    for idx in range(0, sim.dimension):
        if random.random() <= P:
            error_count += 1
            if error_count > 1:
                return False, error_count
            if debug:
                print(f'Error in {idx}')
            pauli_x_error = random.random() > 0.5
            if pauli_x_error:
                sim.apply_single_qubit_gate(X, idx)
            else:
                sim.apply_single_qubit_gate(PAULI_Z, idx)
    if debug:
        print('Finishing error simulation part:')
        for i in range(sim.dimension):
            print(f'State {i}: {sim.get_qubit_state(i)}')

        if error_count:
            print(f'{error_count} - P: {P}')
    # Decoding
    # S6 - clone of S4
    sim.apply_n_qubit_gate(S4_operator)

    # S7 - clone of S5
    sim.apply_n_qubit_gate(S5_operator)

    # S8
    TOFFOLI_120 = TOFFOLI(N=9, controls=[1, 2], target=0)
    TOFFOLI_453 = TOFFOLI(N=9, controls=[4, 5], target=3)
    TOFFOLI_876 = TOFFOLI(N=9, controls=[8, 7], target=6)

    # Комбинируем все Toffoli-гейты в один оператор
    S8_operator = np.dot(np.dot(TOFFOLI_120, TOFFOLI_453), TOFFOLI_876)
    sim.apply_n_qubit_gate(S8_operator)

    # S9 - clone of S3
    sim.apply_n_gates(H, I, I, H, I, I, H, I, I)

    # S10 - clone of S1
    sim.apply_n_qubit_gate(CNOT_03)

    # S11 - clone of S2
    sim.apply_n_qubit_gate(CNOT_06)

    # S12
    TOFFOLI_360 = TOFFOLI(N=9, controls=[3, 6], target=0)
    sim.apply_n_qubit_gate(TOFFOLI_360)
    if debug:
        print('Finishing decoding part')
        for i in range(sim.dimension):
            print(f'State {i}: {sim.state[0]}')

    finite_state = sim.get_qubit_state_raw(0)
    if np.isclose(finite_state['|0>'], inital_state['|0>'], 0.01) and np.isclose(finite_state['|1>'],
                                                                                 inital_state['|1>'], 0.01):
        return (True, error_count)
    else:
        return (False, error_count)

def no_correction(P=0.05, debug = False):

    sim = NQubitSimulator(1)
    sim.apply_single_qubit_gate(RX(np.pi / 5), 0)
    inital_state = sim.get_qubit_state_raw(0)

    # No correction.
    # Error simulation.
    error_count = 0
    # Random inverse qubit with PAULI [X or Z] (Simulate Error)
    if random.random() <= P:
        error_count += 1
        if debug:
            print(f'Error applied')
        sim.apply_single_qubit_gate(X, 0)

    # Check.
    finite_state = sim.get_qubit_state_raw(0)
    if np.isclose(finite_state['|0>'], inital_state['|0>'], 0.0001) and np.isclose(finite_state['|1>'],
                                                                                 inital_state['|1>'], 0.0001):
        return True, error_count
    else:
        return False, error_count


def theory(p):
    return 1 - (1 + 8 * p) * (1 - p) ** 8

def shor9_withplot():
    import matplotlib.pyplot as plt

    p_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    p_e_corr_list = []
    p_e_no_corr_list = []
    theoretical_p_e_list = []
    total_rounds = 100

    for P in p_values:
        print(f"Simulating for p = {P:.2f}")

        # Симуляция с коррекцией ошибок
        correct_count = 0
        for _ in range(total_rounds):
            correct, _ = shor9(P=P)
            if not correct:
                correct_count += 1
        p_e_corr = correct_count / total_rounds
        p_e_corr_list.append(p_e_corr)

        # Симуляция без коррекции ошибок
        correct_count_nc = 0
        for _ in range(total_rounds):
            correct_nc, _ = no_correction(P=P)
            if not correct_nc:
                correct_count_nc += 1
        p_e_nc = correct_count_nc / total_rounds
        p_e_no_corr_list.append(p_e_nc)

        # Теоретическое значение
        p_e_th = theory(P)
        theoretical_p_e_list.append(p_e_th)

        print(f"p={P:.2f}: Simulated p_e_corr={p_e_corr:.4f}, Simulated p_e_no_corr={p_e_nc:.4f}, Theoretical p_e={p_e_th:.4f}")

    # Построение графика
    plt.plot(p_values, p_e_corr_list, label='С коррекцией', linestyle='-', color='b', alpha=0.7)
    plt.plot(p_values, p_e_no_corr_list, label='Без коррекции', linestyle='-', color='r', alpha=0.7)
    p_values_fine = np.linspace(0, max(p_values), 100)
    theory_values = [theory(p) for p in p_values_fine]
    plt.plot(p_values_fine, theory_values, label='Теоретическое значение', color='green', linestyle='--')
    # Прямая для теоретического значения без коррекции (p_e = p)
    plt.plot(p_values_fine, p_values_fine, label='Теоретическое значение (без коррекции)', color='black', linestyle='--')

    plt.xlabel('Вероятность ошибки P')
    plt.ylabel('Общая вероятность ошибки P_e')
    plt.title('Зависимость вероятности ошибки от P')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    shor9_withplot()