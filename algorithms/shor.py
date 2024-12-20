import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from fractions import Fraction
from math import gcd
import random


def c_amod15(a, power):
    """
    Преобразование a^2^power mod 15.
    :param a: Параметр a.
    :param power: Степень.
    :return: Вентиль преобразования.
    """
    assert a in [2, 4, 7, 8, 11, 13]
    U = QuantumCircuit(4)
    for _ in range(power):
        if a in [2, 13]:  # Пример для a = 2, 13
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        elif a in [4, 11]:  # Пример для a = 4, 11
            U.swap(1, 3)
            U.swap(0, 2)
        elif a in [7, 8]:  # Пример для a = 7, 8
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 11, 13]:  # Применяем X-операцию для некоторых значений
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U


def choose_random_a(N):
 """
 Выбирает случайное значение a, которое взаимно просто с N.
 Если gcd(a, N) != 1, то сразу возвращает делитель N.
 :param N: Число для факторизации.
 :return: a, которое взаимно просто с N (или делитель, если найден).
 """
 while True:
  a = random.randint(2, N - 1)
  common_divisor = gcd(a, N)
  if common_divisor != 1:
   print(f"Найден делитель {common_divisor} при выборе a = {a}.")
   return common_divisor  # Возвращаем делитель, если он найден
  return a


def inversed_qft(n):
 """
 Обратное квантовое преобразование Фурье.
 :param n: Количество входов квантового вентиля.
 :return: Вентиль обратного преобразования Фурье.
 """
 qc = QuantumCircuit(n)  # Создаем квантовую схему для n кубитов

 # Меняем порядок кубитов
 for qubit in range(n // 2):
  qc.swap(qubit, n - qubit - 1)  # Операция SWAP между соответствующими кубитами

 # Применяем обратное фазовое преобразование для каждого кубита
 for j in range(n):  # Перебираем все кубиты
  for m in range(j):  # Добавляем фазовые сдвиги для предыдущих кубитов
   qc.cp(-np.pi / float(2 ** (j - m)), m, j)  # Контролируемый фазовый сдвиг
  qc.h(j)  # Применяем вентиль Адамара для завершения преобразования кубита

 qc.name = "QFT-1"

 return qc


def shor_period(a, N, shots=2048):
    """
    Квантовый алгоритм Шора для нахождения периода функции a^x mod N.
    :param a: Параметр функции a.
    :param N: Параметр функции N.
    :param shots: Количество измерений в симуляции.
    """
    n_count = 8  # Количество расчетных кубитов

    qc = QuantumCircuit(QuantumRegister(8, 'x'), QuantumRegister(4, 'f(x)'), ClassicalRegister(8))

    # Инициализация расчетных кубитов в состоянии |+>
    for q in range(n_count):
        qc.h(q)

    # Вспомогательные регистры в состоянии |1>
    qc.x(3 + n_count)

    # Выполнение controlled-U операции
    for q in range(n_count):
        qc.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])

    # Выполнение QFT-1 операции
    qc.append(inversed_qft(n_count), range(n_count))

    # Измерение расчетных кубитов
    qc.measure(range(n_count), range(n_count))

    # Вывод квантовой схемы
    print("\nКвантовая схема для текущего a:")
    print(qc)

    # Симуляция
    qasm_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, qasm_sim)
    result = qasm_sim.run(t_qc, shots=shots).result()
    counts = result.get_counts()

    # Поиск наиболее вероятного состояния
    max_state = max(counts, key=counts.get)
    phase = int(max_state, 2) / (2 ** n_count)
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    if r == 0:
        raise ValueError("Невозможно найти период. Попробуйте снова.")
    return r


def factorization(N):
 """
 Алгоритм факторизации чисел (поиск простых делителей числа) с выбором a случайным образом.
 :param N: Число, необходимое факторизовать.
 """
 attempt = 0

 while True:
  attempt += 1

  # Выбираем случайное a
  a = choose_random_a(N)
  print(f"\nПопытка {attempt}: \nСлучайное а = {a}")
  # Если найден делитель на этапе выбора a, завершаем алгоритм
  if isinstance(a, int) and gcd(a, N) != 1:
   print(f"Простые делители числа {N}: {a} и {N // a}")
   return

  try:
   r = shor_period(a, N)  # Вызываем квантовый алгоритм для нахождения периода
   print(f"Период: r = {r}")
   if r % 2 != 0 or r == 0:
    print("Неправильный период. Пробуем снова.")
    continue

   # Пытаемся найти делители
   guesses = [gcd(a ** (r // 2) - 1, N), gcd(a ** (r // 2) + 1, N)]
   for guess in guesses:
    if guess in [1, N]:
     print(f"Угаданный делитель {guess} является тривиальным. Пропускаем.")
     continue
    if (N % guess) == 0:
     print(f"Простые делители числа {N}: {guess} и {N // guess}")
     print("Факторизация завершена.")
     return
    else:
     print(f"Предположение {guess} некорректно.")
  except ValueError as e:
   print(f"Ошибка: {e}. Пробуем снова.")


# Факторизация числа N
N = 15  # Число для факторизации
factorization(N)
