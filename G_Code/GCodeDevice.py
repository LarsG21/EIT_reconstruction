import collections
import threading

import serial
import time


# ser = serial.Serial('COM3', 115200, timeout=1)
#
#
# time.sleep(5)
# # ser.write(str.encode("G28\r\n"))
# print("Homing")
#
# # time.sleep(20)
#
# ser.write(str.encode("G0 Z10 F6000\r\n"))
#
# # ser.write(str.encode("G0 X0 Y0 Z0 F6000\r\n"))
#
# while True:
#     line = ser.readline()
#     print(line)
#
#     if line == b'ok\r\n':
#         break
#
# time.sleep(1)
#
# ser.write(str.encode("G28\r\n"))
#
# time.sleep(10)


class GCodeDevice:

    def __init__(self, com_port, baud_rate=115200, timeout=1):
        self.ser = serial.Serial(com_port, baud_rate, timeout=timeout)
        self._receive_buffer = collections.deque(maxlen=100)
        self._send_buffer = collections.deque(maxlen=100)

        reader = threading.Thread(target=self._read_serial)
        reader.start()

        time.sleep(1)

    def _read_serial(self):
        while True:
            line = self.ser.readline()
            self._receive_buffer.append(line)

    def home(self):
        self.ser.write(str.encode("G28\r\n"))
        while True:
            print()

        time.sleep(1)

    def move_to(self, x, y, z):
        print(f"Moving to {x}, {y}, {z}")
        self.ser.write(str.encode(f"G0 X{x} Y{y} Z{z} F6000\r\n"))
        print("Done moving")


if __name__ == '__main__':
    ender = GCodeDevice('COM3')
    time.sleep(3)
    ender.move_to(0, 0, 30)
    time.sleep(6)
    ender.move_to(10, 0, 30)
    # ender.move_relative(0, 0, -10)