from quantum_simulator.interface import QuantumDevice
import numpy as np
class QRNG():
    @staticmethod
    def qrng(device: QuantumDevice) -> bool:
        with device.using_qubit() as q:
            q.h()
            return q.measure()

    @staticmethod
    def qrng_with_rotation(device: QuantumDevice, theta:float = np.pi / 6) -> bool:
        with device.using_qubit() as q:
            q.rotation(theta)
            return q.measure()

