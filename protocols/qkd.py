from quantum_simulator.interface import Qubit
from quantum_simulator.interface import QuantumDevice

class QKD():
    @staticmethod
    def prepare_classical_message(bit: bool, q: Qubit) -> None:
        if bit:
            q.x()

    @staticmethod
    def eve_measure(q: Qubit) -> bool:
        return q.measure()

    @staticmethod
    def send_classical_bit(self, device: QuantumDevice, bit: bool) -> None:
        with device.using_qubit() as q:
            self.prepare_classical_message(bit, q)
            result = self.eve_measure(q)
            q.reset()
        assert result == bit

    @staticmethod
    def prepare_classical_message_plusminus(bit: bool, q: Qubit) -> None:
        if bit:
            q.x()
        q.h()

    @staticmethod
    def eve_measure_plusminus(q: Qubit) -> bool:
        q.h()
        return q.measure()

    @staticmethod
    def send_classical_bit_plusminus(self, device: QuantumDevice, bit: bool) -> None:
        with device.using_qubit() as q:
            self.prepare_classical_message_plusminus(bit, q)
            result = self.eve_measure_plusminus(q)
            assert result == bit

    @staticmethod
    def send_classical_bit_wrong_basis(device: QuantumDevice, bit: bool) -> None:
        with device.using_qubit() as q:
            QKD.prepare_classical_message(bit, q)
            result = QKD.eve_measure_plusminus(q)
            assert result == bit, "Две стороны не имеют одинаковое битовое значение"