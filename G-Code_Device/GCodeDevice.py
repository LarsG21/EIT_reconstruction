import collections
import threading

import serial
import time

from KeyPressModule import KeyPressModule


class GCodeDevice:

    def __init__(self, com_port, baud_rate=115200, timeout=1):
        self.ser = serial.Serial(com_port, baud_rate, timeout=timeout)
        self._receive_buffer = collections.deque(maxlen=100)
        self._send_buffer = collections.deque(maxlen=100)

        self.current_position = (0, 0, 0)  # x, y, z

        self.maximal_limits = (200, 200, 200)  # x, y, z
        self.minimal_limits = (0, 0, 0)  # x, y, z

        reader = threading.Thread(target=self._read_serial)
        reader.start()

        time.sleep(1)
        self.home()

    def _read_serial(self):
        while True:
            line = self.ser.readline()
            line = str(line, 'utf-8')
            # print(line)
            self._receive_buffer.append(line)

    def _get_last_line(self):
        return self._get_receive_buffer_at(-1)

    def _get_last_line_contains(self, string):
        for line in reversed(self._receive_buffer):
            if string in line:
                return line
        return ""

    def _get_receive_buffer_at(self, index):
        # with try catch for index out of range
        try:
            return self._receive_buffer[index]
        except IndexError:
            return ""

    def home(self):
        self.ser.write(str.encode("G28\r\n"))
        time.sleep(3)
        # while last line 2 lines do not contain "busy"
        while "busy" in self._get_receive_buffer_at(-2) or \
                "busy" in self._get_receive_buffer_at(-1):
            print("Waiting for homing to finish")
            time.sleep(1)
        print("Homing finished")
        self.current_position = (0, 0, 0)

    def check_limits_with_current_position(self, x, y, z):
        if self.current_position[0] + x > self.maximal_limits[0] or \
                self.current_position[1] + y > self.maximal_limits[1] or \
                self.current_position[2] + z > self.maximal_limits[2]:
            return False
        if self.current_position[0] + x < self.minimal_limits[0] or \
                self.current_position[1] + y < self.minimal_limits[1] or \
                self.current_position[2] + z < self.minimal_limits[2]:
            return False
        return True

    def move_to(self, x, y, z):
        if not self.check_limits_with_current_position(x, y, z):
            print("ERROR: Trying to move outside of limits")
            return
        print(f"Moving to {x}, {y}, {z}")
        self.ser.write(str.encode(f"G0 X{x} Y{y} Z{z} F6000\r\n"))
        self.current_position = (x, y, z)

    def move_relative(self, x, y, z):
        if not self.check_limits_with_current_position(x, y, z):
            print("ERROR: Trying to move outside of limits")
            return
        print(f"Moving to {x}, {y}, {z}")
        self.ser.write(str.encode(f"G91\r\n"))
        self.ser.write(str.encode(f"G0 X{x} Y{y} Z{z} F6000\r\n"))
        self.current_position = (self.current_position[0] + x,
                                 self.current_position[1] + y,
                                 self.current_position[2] + z)
        self.ser.write(str.encode(f"G90\r\n"))


def main():
    ender = GCodeDevice('COM3')
    kp = KeyPressModule()
    while True:
        key = kp.get_keypress_down()
        if key == 'w':
            ender.move_relative(0, 0, 10)
        elif key == 's':
            ender.move_relative(0, 0, -10)
        elif key == 'a':
            ender.move_relative(-10, 0, 0)
        elif key == 'd':
            ender.move_relative(10, 0, 0)
        elif key == 'h':
            ender.home()
        print(ender.current_position)
        time.sleep(0.1)



if __name__ == '__main__':
    main()
