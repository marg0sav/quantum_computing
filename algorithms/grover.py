import numpy as np
from utils.constants import H
from quantum_simulator.simulator import NQubitSimulator


class GroverAlgorithm:
    def __init__(self, n_qubits, marked_item):
        self.n_qubits = n_qubits
        self.marked_item = marked_item
        self.simulator = NQubitSimulator(n_qubits)

    def apply_hadamard_to_all(self):
        """ Применяет оператор Адамара ко всем кубитам, создавая суперпозицию. """
        for i in range(self.n_qubits):
            self.simulator.apply_single_qubit_gate(H, i)

    def apply_oracle(self):
        """
        Оракул меняет фазу целевого состояния.
        """

        # Шаг 1: Создаём единичную матрицу размером 2^n x 2^n (где n — количество кубитов).
        control_state = np.eye(2 ** self.n_qubits)

        # Шаг 2: Изменяем фазу целевого состояния.
        # Мы берём строку и столбец, соответствующие целевому состоянию (индексу marked_item),
        # и устанавливаем значение -1 в диагональный элемент.
        control_state[self.marked_item, self.marked_item] = -1

        # Шаг 3: Применяем оператор к квантовой системе.
        # Полученная матрица control_state интерпретируется как многокубитный вентиль.
        # Мы применяем этот вентиль к текущей квантовой системе с помощью метода симулятора.
        # Это действие изменяет амплитуды волновой функции, добавляя фазовый сдвиг
        # к целевому состоянию.
        self.simulator.apply_n_qubit_gate(control_state)

    def apply_diffusion_operator(self):
        """
        Оператор диффузии для усиления амплитуды целевого состояния.
        """
        # Шаг 1: Применяем вентиль Адамара ко всем кубитам.
        self.apply_hadamard_to_all()

        # Шаг 2: Применяем фазовый переворот относительно состояния |0>.
        # Этот шаг отражает все состояния относительно начального состояния |0>.
        # В результате амплитуды всех состояний модифицируются так, чтобы усилить амплитуду целевого состояния.
        self.apply_phase_flip()

        # Шаг 3: Снова применяем вентиль Адамара ко всем кубитам.
        # Это возвращает состояние системы в исходный базис, усиливая амплитуду целевого состояния
        # и снижая амплитуды других состояний.
        self.apply_hadamard_to_all()

    def apply_phase_flip(self):
        """
        Переворачивает фазу всех состояний, кроме состояния |0>^{⊗ n}.
        Этот шаг используется как часть оператора диффузии в алгоритме Гровера.
        """

        # Шаг 1: Создаём вектор состояния |0>^{⊗ n}.
        # Это базовое состояние, где все кубиты находятся в состоянии |0>.
        # Для этого инициализируем вектор длиной 2^n с нулями.
        KET_0_n = np.zeros((2 ** self.n_qubits, 1))

        # Устанавливаем амплитуду для состояния |0>^{⊗ n} равной 1.
        # Теперь вектор KET_0_n представляет состояние |0>^{⊗ n}.
        KET_0_n[0, 0] = 1

        # Шаг 2: Создаём фазовый оператор.
        # Этот оператор отражает все состояния относительно состояния |0>^{⊗ n}.
        # Математически это можно выразить как:
        # P = 2 * |0><0| - I,
        # где |0><0| — оператор проекции на состояние |0>^{⊗ n}.
        phase_flip = 2 * np.outer(KET_0_n, KET_0_n) - np.eye(2 ** self.n_qubits)

        # Шаг 3: Применяем фазовый оператор к квантовой системе.
        # Этот оператор переворачивает фазу всех состояний, кроме состояния |0>^{⊗ n}.
        # Если текущее состояние системы |ψ⟩ = α|0> + β|1>, то после применения phase_flip:
        # |ψ⟩ -> α|0> - β|1>.
        self.simulator.apply_n_qubit_gate(phase_flip)

    def run(self):
        """ Выполнение алгоритма Гровера """
        # Инициализация суперпозиции
        self.apply_hadamard_to_all()

        # Определение количества итераций
        iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** self.n_qubits)))
        for _ in range(iterations):
            self.apply_oracle()
            self.apply_diffusion_operator()

        # Измерение
        return self.simulator.measure_multiple_qubits(range(self.n_qubits))


def run_experiment(max_qubits=5):
    """Проводит эксперимент с алгоритмом Гровера для различных размерностей устройств и целевых состояний."""
    results = []

    for n_qubits in range(2, max_qubits + 1):  # Варьируем количество кубитов от 2 до max_qubits
        # Количество возможных состояний — это 2^n_qubits
        num_states = 2 ** n_qubits

        # Выбираем случайный индекс для целевого состояния в диапазоне [0, num_states - 1]
        marked_item = np.random.randint(0, num_states)

        # Создаем экземпляр алгоритма Гровера для текущего количества кубитов и целевого состояния
        grover = GroverAlgorithm(n_qubits, marked_item)

        # Запускаем алгоритм и получаем результат измерения
        result_bits = grover.run()

        # Конвертируем результат измерения (список битов) в целое число
        result_index = int("".join(map(str, result_bits)), 2)

        # Сохраняем результат эксперимента для дальнейшего анализа
        results.append({
            "n_qubits": n_qubits,
            "marked_item": marked_item,
            "result_index": result_index,
        })

    # Итоговый вывод для анализа
    for res in results:
            print(f"Результаты для {res['n_qubits']} кубитов:")
            print(f"- Целевое состояние: {res['marked_item']}")
            print(f"- Найденный индекс: {res['result_index']}")
            print("-" * 40)

    return results


# Запуск эксперимента
experiment_results = run_experiment(max_qubits=7)