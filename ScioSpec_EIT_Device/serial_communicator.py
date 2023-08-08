import time

import serial
import serial.tools.list_ports



def list_serial_devices():
    ports = serial.tools.list_ports.comports()
    if len(ports) == 0:
        print("No serial devices found.")
    else:
        print("List of serial devices:")
        for port in ports:
            print(f"- Port: {port.device}, Description: {port.description}, {port.manufacturer}")
    return ports




if __name__ == '__main__':
    ports = list_serial_devices()

    ser = serial.Serial("COM3", 115200, timeout=1)
    # send 0x18 0x01 0x84 0x18
    ser.write(b'\x18\x00\x18')
    while True:
        ser.write(b'\x18\x00\x18')
        line = ser.readline()
        # line = str(line, 'utf-8')
        print(line)
        time.sleep(0.1)
        # self._receive_buffer.append(line)
