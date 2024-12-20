from typing import List

from quantum_simulator.interface import QuantumDevice
from quantum_simulator.interface import Qubit
from quantum_simulator.simulator import SingleQubitSimulator


class BB84():
    @staticmethod
    def sample_random_bit(device: QuantumDevice) -> bool:
     with device.using_qubit() as q:
      q.h()
      result = q.measure()
      q.reset()
     return result

    @staticmethod
    def prepare_message_qubit(message: bool, basis: bool, q: Qubit) -> None:
     if message:
      q.x()
     if basis:
      q.h()

    @staticmethod
    def measure_message_qubit(basis: bool, q: Qubit) -> bool:
     if basis:
      q.h()
     result = q.measure()
     q.reset()
     return result

    @staticmethod
    def convert_to_hex(bits: List[bool]) -> str:
     return hex(int("".join(["1" if bit else "0" for bit in bits]), 2))  # двоичной системе (основание 2)

    """
    протокол BB84 для отправки классического бита
    """

    @staticmethod
    def send_single_bit_with_bb84(
      your_device: QuantumDevice,
      alice_device: QuantumDevice
    ) -> tuple:

     [your_message, your_basis] = [
      BB84.sample_random_bit(your_device) for _ in range(2)
     ]

     alice_basis = BB84.sample_random_bit(alice_device)

     with your_device.using_qubit() as q:
      BB84.prepare_message_qubit(your_message, your_basis, q)
      # отправка
      alice_result = BB84.measure_message_qubit(alice_basis, q)

     return ((your_message, your_basis), (alice_result, alice_basis))

    @staticmethod
    def send_single_bit_with_eve_interception(
      your_device: QuantumDevice,
      eve_device: QuantumDevice,
      alice_device: QuantumDevice
    ) -> tuple:

     # Боб выбирает случайное сообщение и базис
     [your_message, your_basis] = [
      BB84.sample_random_bit(your_device) for _ in range(2)
     ]

     # Ева перехватывает сообщение
     eve_basis = BB84.sample_random_bit(eve_device)

     with your_device.using_qubit() as q:
      BB84.prepare_message_qubit(your_message, your_basis, q)
      # ОТПРАВКА КУБИТА...

      # Ева измеряет кубит в своем случайном базисе
      eve_result = BB84.measure_message_qubit(eve_basis, q)

      # Ева "восстанавливает" кубит по своему результату
      BB84.prepare_message_qubit(eve_result, eve_basis, q)

      # Ева отправляет кубит Алисе
      alice_basis = BB84.sample_random_bit(alice_device)
      alice_result = BB84.measure_message_qubit(alice_basis, q)

     return ((your_message, your_basis), (eve_result, eve_basis), (alice_result, alice_basis))

    """
    протокол BB84 для обмена ключом с Алисой
    """

    @staticmethod
    def simulate_bb84(n_bits: int) -> tuple:
     your_device = SingleQubitSimulator()
     alice_device = SingleQubitSimulator()

     key = []
     n_rounds = 0
     while len(key) < n_bits:
      n_rounds += 1
      ((your_message, your_basis), (alice_result, alice_basis)) = \
       (BB84.send_single_bit_with_bb84(your_device, alice_device))

      if your_basis == alice_basis:
       assert your_message == alice_result  # проверка на прослушку
       key.append(your_message)

     print(f"Потребовалось {n_rounds} раундов, чтобы сгенерировать {n_bits} - битовый ключ.")
     return key

    @staticmethod
    def simulate_bb84_with_eve(n_bits: int) -> tuple:
     your_device = SingleQubitSimulator()
     eve_device = SingleQubitSimulator()
     alice_device = SingleQubitSimulator()

     key = []
     n_rounds = 0
     while len(key) < n_bits:
      n_rounds += 1
      ((your_message, your_basis), (eve_result, eve_basis), (alice_result, alice_basis)) = \
       (BB84.send_single_bit_with_eve_interception(your_device, eve_device, alice_device))

      if your_basis == alice_basis:
       # Проверка на вмешательство Евы: если базисы совпадают, сообщение должно быть правильным
       if your_message != alice_result:
        print(f"Вмешательство Евы! Значение искажено.")
       else:
        key.append(your_message)

     print(f"Потребовалось {n_rounds} раундов, чтобы сгенерировать {n_bits} - битовый ключ.")
     return key

    """
    протокол BB84 для обмена ключом с Алисой
    """

    def apply_one_time_pad(message: List[bool], key: List[bool]) -> List[bool]:  # шифрование
     return [
      message_bit ^ key_bit  # XOR
      for (message_bit, key_bit) in zip(message, key)
     ]
