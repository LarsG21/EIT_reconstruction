import collections
import threading

import numpy as np
import serial
import serial.tools.list_ports
import time


def calculate_moving_time(last_position: np.ndarray, center_for_moving: np.ndarray):
    """
    Calculates the time to move from the last position to the new position.
    :param last_centers:
    :param center_for_moving:
    :return:
    """
    MOVING_SEED_Z = 5  # in mm per second
    MOVING_SEED_X = 50  # in mm per second

    time_to_move = int(np.linalg.norm(last_position[0] - center_for_moving[0]) / MOVING_SEED_X +
                       np.linalg.norm(last_position[1] - center_for_moving[1]) / MOVING_SEED_Z)
    print("time_to_move", time_to_move)

    return time_to_move

def list_serial_devices():
    ports = serial.tools.list_ports.comports()
    if len(ports) == 0:
        print("No serial devices found.")
    else:
        print("List of serial devices:")
        for port in ports:
            print(f"- Port: {port.device}, Description: {port.description}, {port.manufacturer}")
    return ports


class GCodeDevice:

    def __init__(self, com_port, baud_rate=115200, timeout=1, movement_speed=6000, home_on_init=True):
        self.ser = serial.Serial(com_port, baud_rate, timeout=timeout)
        self._receive_buffer = collections.deque(maxlen=100)
        self._send_buffer = collections.deque(maxlen=100)
        self.movement_speed = movement_speed  # mm/min

        self.current_position = (0, 0, 0)  # x, y, z

        self.maximal_limits = (200, 200, 200)  # x, y, z
        self.minimal_limits = (0, 0, 0)  # x, y, z

        reader = threading.Thread(target=self._read_serial)
        reader.start()

        time.sleep(1)
        if home_on_init:
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

    def check_limits(self, x, y, z):
        if x > self.maximal_limits[0] or \
                y > self.maximal_limits[1] or \
                z > self.maximal_limits[2]:
            return False
        if x < self.minimal_limits[0] or \
                y < self.minimal_limits[1] or \
                z < self.minimal_limits[2]:
            return False
        return True

    def move_to(self, x, y, z):
        if not self.check_limits(x, y, z):
            print(f"ERROR: Trying to move outside of limits {x}, {y}, {z}")
            return
        print(f"Moving to {x}, {y}, {z}")
        self.ser.write(str.encode(f"G0 X{x} Y{y} Z{z} F{self.movement_speed}\r\n"))
        self.current_position = (x, y, z)

    def move_relative(self, x, y, z):
        if not self.check_limits_with_current_position(x, y, z):
            print("ERROR: Trying to move outside of limits")
            return
        print(f"Moving to {x}, {y}, {z}")
        self.ser.write(str.encode(f"G91\r\n"))
        self.ser.write(str.encode(f"G0 X{x} Y{y} Z{z} F{self.movement_speed}\r\n"))
        self.current_position = (self.current_position[0] + x,
                                 self.current_position[1] + y,
                                 self.current_position[2] + z)
        self.ser.write(str.encode(f"G90\r\n"))

    def calculate_moving_time(self, last_position: np.ndarray, center_for_moving: np.ndarray):
        """
        Calculates the time to move from the last position to the new position.
        :param last_position: Last position in the format [x, z] in mm
        :param center_for_moving: New position in the format [x, z] in mm
        :return:
        """
        MOVING_SEED_Z = 5  # in mm per second
        MOVING_SEED_X = 60  # in mm per second

        time_to_move = int(np.linalg.norm(last_position[0] - center_for_moving[0]) / MOVING_SEED_X +
                           np.linalg.norm(last_position[1] - center_for_moving[1]) / MOVING_SEED_Z)
        print("time_to_move", time_to_move)

        return time_to_move

def main():
    devices = list_serial_devices()
    ender = None
    for device in devices:
        if "USB-SERIAL CH340" in device.description:
            ender = GCodeDevice(device.device)
            break
    if ender is None:
        raise Exception("No Ender 3 found")
    else:
        print("Ender 3 found")
    kp = KeyPressModule()
    print(ender.maximal_limits[0])
    # X = 20
    # Z = 200
    # calculate_moving_time(np.array([0, 0]), np.array([X, Z]))
    # ender.move_to(X, 0, Z)
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
    from KeyPressModule import KeyPressModule
    main()
