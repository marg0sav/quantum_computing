import numpy as np
from utils.qrng import QRNG
from protocols.bb84 import BB84
from algorithms.deutsch import DeutschAlgorithm
from algorithms.deutsch_jozsa import DeutschJozsaAlgorithm, bool_zero_3, bool_or_3, bool_xor_3
from algorithms.bernstein_vazirani import BernsteinVaziraniAlgorithm
from algorithms.simon import SimonAlgorithm
from protocols.teleportation import QuantumTeleportation
from quantum_simulator.simulator import SingleQubitSimulator, TwoQubitSimulator, NQubitSimulator
from error_correction.X_error_correction import XErrorCorrectionSimulator
from error_correction.Z_error_correction import PhaseErrorCorrectionSimulator
from error_correction.shor import Shor9ErrorCorrection

# Флаги для выбора режима
USE_QRNG = 0
USE_BB84 = 0
USE_DEUTSCH = 0
USE_DEUTSCH_JOZSA = 0
USE_BERNSTEIN_VAZIRANI = 0
USE_SIMON = 0
USE_TELEPORTATION = 0
USE_X_ERROR_CORRECTION = 1
USE_Z_ERROR_CORRECTION = 1
USE_SHOR_ERROR_CORRECTION = 0


def print_in_box(text):
   border = "+" + "-" * (len(text) + 2) + "+"
   print("\n\n" + border)
   print(f"| {text} |")
   print(border + "\n\n")


if __name__ == "__main__":

    if USE_QRNG:
        print_in_box("Генерация случайных чисел с использованием матрицы Адамара и поворотной матрицы")
        qsim = SingleQubitSimulator()

        total_samples = 100
        count_zeros_hadamard = 0
        count_ones_hadamard = 0
        count_zeros_rotation = 0
        count_ones_rotation = 0

        print("Генерация с использованием матрицы Адамара (50/50):")
        for idx_sample in range(total_samples):
            random_sample = QRNG.qrng(qsim)
            if random_sample == 0:
                count_zeros_hadamard += 1
            else:
                count_ones_hadamard += 1

        percentage_zeros_hadamard = (count_zeros_hadamard / total_samples) * 100
        percentage_ones_hadamard = (count_ones_hadamard / total_samples) * 100

        print("\nРезультаты генерации с матрицей Адамара:")
        print(f"Количество 0: {count_zeros_hadamard} ({percentage_zeros_hadamard:.2f}%)")
        print(f"Количество 1: {count_ones_hadamard} ({percentage_ones_hadamard:.2f}%)\n")

        print("Генерация с использованием поворотной матрицы:")
        for idx_sample in range(total_samples):
            random_sample = QRNG.qrng_with_rotation(qsim, np.pi / 3)
            if random_sample == 0:
                count_zeros_rotation += 1
            else:
                count_ones_rotation += 1

        percentage_zeros_rotation = (count_zeros_rotation / total_samples) * 100
        percentage_ones_rotation = (count_ones_rotation / total_samples) * 100

        print("\nРезультаты генерации с поворотной матрицей:")
        print(f"Количество 0: {count_zeros_rotation} ({percentage_zeros_rotation:.2f}%)")
        print(f"Количество 1: {count_ones_rotation} ({percentage_ones_rotation:.2f}%)")

    if USE_BB84:
        print_in_box("Запуск протокола BB84 для симуляции")
        print("Генерирование 96-битового ключа путем симулирования BB84...")
        key = BB84.simulate_bb84(96)
        print(f"Получен ключ {BB84.convert_to_hex(key)}.")

        message = [
            1, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 0, 1,
            1, 1, 0, 1, 1, 1, 0, 0,
            1, 0, 0, 1, 0, 1, 1, 0,
            1, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 0, 1,
            1, 1, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 1,
            1, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 0, 1,
            1, 1, 0, 1, 1, 1, 0, 0,
            1, 0, 1, 1, 1, 0, 1, 1
        ]
        print(f"Использование ключа для отправки секретного сообщения: {BB84.convert_to_hex(message)}.")
        encrypted_message = BB84.apply_one_time_pad(message, key)
        print(f"Зашифрованное сообщение: {BB84.convert_to_hex(encrypted_message)}.")
        decrypted_message = BB84.apply_one_time_pad(encrypted_message, key)
        print(f"Алиса расшифровала, получив: {BB84.convert_to_hex(decrypted_message)}.")

        print("\n\n\n\n")

        print("Генерирование 96-битового ключа с перехватом Евой...")
        key = BB84.simulate_bb84_with_eve(96)
        print(f"Получен ключ {BB84.convert_to_hex(key)}.")

        message = [
            0, 0, 1, 0, 0, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 1, 0, 0, 1,
            0, 0, 1, 0, 0, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1, 1,
            0, 1, 0, 0, 0, 1, 0, 0
        ]
        print(f"Использование ключа для отправки секретного сообщения: {BB84.convert_to_hex(message)}.")
        encrypted_message = BB84.apply_one_time_pad(message, key)
        print(f"Зашифрованное сообщение: {BB84.convert_to_hex(encrypted_message)}.")
        decrypted_message = BB84.apply_one_time_pad(encrypted_message, key)
        print(f"Расшифрованное сообщение: {BB84.convert_to_hex(decrypted_message)}.")

    if USE_DEUTSCH:
     print_in_box("Запуск алгоритма Дойча")
     simulator = TwoQubitSimulator()
     deutsch = DeutschAlgorithm(simulator)

     # Использование различных оракулов для проверки алгоритма Дойча
     result1 = deutsch.run(DeutschAlgorithm.ORACLE1)
     print(f"Oracle 1 (f(x)=1) результат: {result1}")

     result2 = deutsch.run(DeutschAlgorithm.ORACLE2)
     print(f"Oracle 2 (f(x)=0) результат: {result2}")

     result3 = deutsch.run(DeutschAlgorithm.ORACLE3)
     print(f"Oracle 3 (f(x)=x) результат: {result3}")

     result4 = deutsch.run(DeutschAlgorithm.ORACLE4)
     print(f"Oracle 4 (f(x)=!x) результат: {result4}")

    if USE_DEUTSCH_JOZSA:
     print_in_box("Запуск алгоритма Дойча-Джозы")

     # Симулятор с 4 кубитами (3 входных кубита + 1 дополнительный)
     nQubSim = NQubitSimulator(4)

     print('ORACLE1: f(x) = 0')
     print(f'Result: {DeutschJozsaAlgorithm(nQubSim).run(DeutschJozsaAlgorithm.generate_oracle(3, bool_zero_3))}')

     print('ORACLE2: f(x) = OR (3 кубита)')
     print(f'Результат: {DeutschJozsaAlgorithm(nQubSim).run(DeutschJozsaAlgorithm.generate_oracle(3, bool_or_3))}')

     print('ORACLE3: f(x) = XOR (3 кубита)')
     print(f'Результат: {DeutschJozsaAlgorithm(nQubSim).run(DeutschJozsaAlgorithm.generate_oracle(3, bool_xor_3))}')

     print("\n\n\n")

     nQubSim = NQubitSimulator(4)  # Симулятор с 4 кубитами, последний - дополнительный

     # Генерируем оракулы для каждой из функций
     oracle_zero = DeutschJozsaAlgorithm.generate_oracle(3, bool_zero_3)
     oracle_or = DeutschJozsaAlgorithm.generate_oracle(3, bool_or_3)
     oracle_xor = DeutschJozsaAlgorithm.generate_oracle(3, bool_xor_3)

     # Запускаем алгоритм Дойча-Джозы для каждого оракула
     print("ORACLE (Zero for 3 inputs):")
     result = DeutschJozsaAlgorithm(nQubSim).run(oracle_zero)
     print(f"Результат: {result}")

     print("ORACLE (OR for 3 inputs):")
     result = DeutschJozsaAlgorithm(nQubSim).run(oracle_or)
     print(f"Результат: {result}")

     print("ORACLE (XOR for 3 inputs):")
     result = DeutschJozsaAlgorithm(nQubSim).run(oracle_xor)
     print(f"Результат: {result}")

    if USE_BERNSTEIN_VAZIRANI:
     print_in_box("Запуск алгоритма Бернштейна-Вазирани")

     N = 3  # Длина скрытой двоичной строки s
     s = [1, 0, 0]  # Скрытая двоичная строка, которую мы хотим найти

     # Создаем квантовый симулятор с N + 1 кубитами (N для входа, 1 вспомогательный)
     nQubSim = NQubitSimulator(N + 1)

     # Генерируем оракул на основе скрытой строки s
     oracle = BernsteinVaziraniAlgorithm.generate_oracle(N, s)

     # Запускаем алгоритм Бернштейна-Вазирани
     bv_algorithm = BernsteinVaziraniAlgorithm(nQubSim)
     result = bv_algorithm.run(oracle)

     print(f'Скрытый двоичный вектор s: {s}')
     print(f'Измерено: {result}')

    if USE_SIMON:
     print_in_box("Запуск алгоритма Саймона")

     # Пример 1: N=2, s=11
     print("\nПример N=2, s=11:")
     result_n2 = SimonAlgorithm.example_n2_s11()
     print(f"Результат для s='11': {result_n2}")

     # Пример 2: N=3, s=100
     print("\nПример N=3, s=100:")
     result_n3 = SimonAlgorithm.example_n3_s100()
     print(f"Результат для s='100': {result_n3}")

    if USE_TELEPORTATION:
     print_in_box("Запуск квантовой телепортации")

     # Инициализация симулятора с 3 кубитами
     simulator = NQubitSimulator(3)

     # Создаем экземпляр класса QuantumTeleportation
     teleportation = QuantumTeleportation(simulator)

     # Запускаем телепортацию
     teleportation.apply_teleportation()

    if USE_X_ERROR_CORRECTION:
     print_in_box("Симуляция коррекции X-ошибок с использованием тройного повторяющегося кода")

     # Создаем экземпляр симулятора
     simulator = XErrorCorrectionSimulator()

     # Параметры симуляции
     n_qubits = 5  # 3 data qubits + 2 ancilla qubits
     n_runs = 1000  # Количество симуляций для каждого p
     p_values = np.linspace(0, 0.5, 20)  # Значения p от 0 до 0.5

     simulated_p_e = []
     p_e_no_correction = []
     theoretical_p_e_values = []

     for p in p_values:
      print(f"Simulating for p = {p:.2f}")

      # Симуляция с коррекцией ошибок
      p_e_corr = simulator.simulate_correction(p, n_runs)
      simulated_p_e.append(p_e_corr)

      # Симуляция без коррекции ошибок
      p_e_nc = simulator.simulate_no_correction(p, n_runs)
      p_e_no_correction.append(p_e_nc)

      # Теоретическое значение
      p_e_th = simulator.theoretical_p_e(p)
      theoretical_p_e_values.append(p_e_th)

      print(
       f"p={p:.2f}: Simulated p_e_corr={p_e_corr:.4f}, "
       f"Simulated p_e_no_corr={p_e_nc:.4f}, "
       f"Theoretical p_e={p_e_th:.4f}"
      )

     # Построение графика
     simulator.plot_simulation(p_values, simulated_p_e, p_e_no_correction, theoretical_p_e_values)

    if USE_Z_ERROR_CORRECTION:
     print_in_box("Симуляция коррекции фазовых ошибок с использованием тройного повторяющегося кода")

     simulator = PhaseErrorCorrectionSimulator()
     n_qubits = 5
     n_runs = 1000
     p_values = np.linspace(0, 0.5, 10)

     simulated_p_e = []
     p_e_no_correction = []
     theoretical_p_e_values = []

     for p in p_values:
      print(f"Simulating for p = {p:.2f}")
      p_e_corr = simulator.simulate_correction(p, n_runs)
      simulated_p_e.append(p_e_corr)
      p_e_nc = simulator.simulate_no_correction(p, n_runs)
      p_e_no_correction.append(p_e_nc)
      p_e_th = simulator.theoretical_p_e(p)
      theoretical_p_e_values.append(p_e_th)
      print(f"p={p:.2f}: corrected={p_e_corr:.4f}, no_correction={p_e_nc:.4f}, theory={p_e_th:.4f}")

     simulator.plot_simulation(p_values, simulated_p_e, p_e_no_correction, theoretical_p_e_values)

    if USE_SHOR_ERROR_CORRECTION:
     print_in_box("Симуляция коррекции ошибок с использованием кода Шора")

     simulator = Shor9ErrorCorrection()
     p_values = np.linspace(0, 0.1, 11)
     simulator.plot_simulation(p_values, total_rounds=100)