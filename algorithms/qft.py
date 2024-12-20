import typing as t
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

def QFT(
    set_initial_state: t.Callable[[QuantumCircuit, int, str], None],
    nqubits: int,
    initial_state: str = "000",
    reverse: bool = False
) -> QuantumCircuit:
    """Создает квантовую схему для QFT с заданным количеством кубитов и начальным состоянием."""
    circuit = QuantumCircuit(nqubits)  # Создаём квантовую схему
    set_initial_state(circuit, nqubits, initial_state)  # Устанавливаем начальное состояние
    rotation(circuit, nqubits, reverse)  # Добавляем операции вращения
    swap(circuit, nqubits)  # Меняем порядок кубитов для правильного результата
    return circuit


def rotation(circuit: QuantumCircuit, n: int, reverse: bool = False) -> None:
    """Добавляет операции вращения и Адамара для каждого кубита."""
    if n == 0:  # Базовый случай: если кубитов нет, ничего не делаем
        return
    n -= 1  # Переходим к текущему кубиту
    qubits = circuit.qubits
    circuit.h(qubits[n])  # Применяем вентиль Адамара к текущему кубиту
    k = -1 if reverse else 1  # Определяем направление вращения (в прямом или обратном порядке)
    for i in range(n):
        circuit.cp(k * math.pi / 2 ** (n - i), qubits[i], qubits[n])  # Добавляем контролируемые повороты
    rotation(circuit, n, reverse=reverse)  # Рекурсивно переходим к следующему кубиту

def swap(circuit: QuantumCircuit, n: int) -> None:
    """Меняет порядок кубитов для правильного отображения результата."""
    qubits = circuit.qubits
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - i - 1])  # Меняем первый и последний кубиты, двигаясь к центру


def set_initial_state(circuit: QuantumCircuit, n: int, initial_state: str) -> None:
    """
    Устанавливает начальное состояние для кубитов.

    Args:
        circuit (QuantumCircuit): Квантовая схема.
        n (int): Количество кубитов.
        initial_state (str): Двоичное представление начального состояния, например "101".
    """
    assert len(initial_state) == n, "Длина начального состояния должна соответствовать количеству кубитов"

    for i, bit in enumerate(reversed(initial_state)):  # Перебираем состояние с младшего бита
        if bit == '1':
            circuit.x(i)  # Устанавливаем кубит в состояние |1> (если бит равен 1)

# Создаем схему QFT с 3 кубитами
# Определяем начальное состояние
initial_state = "100"

# Создаём схему QFT
qft = QFT(set_initial_state, nqubits=3, initial_state=initial_state)
qft.measure_all()  # Добавляем измерения ко всем кубитам

# Выводим схему
print(qft)

# Запускаем схему на симуляторе
simulator = Aer.get_backend('aer_simulator')  # Инициализируем симулятор
compiled_circuit = transpile(qft, simulator)  # Компилируем схему для симулятора
result = simulator.run(compiled_circuit).result()  # Запускаем симуляцию

# Получаем и отображаем результаты
counts = result.get_counts()
print("Результаты измерений:", counts)
# QFT преобразует начальное состояние кубитов в суперпозицию с фазовыми амплитудами,
# зависящими от начального состояния. Эти амплитуды определяют вероятность получения каждого состояния при измерении.
