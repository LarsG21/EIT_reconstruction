import time

import serial
import struct

def connect_to_device(port):
    try:
        handle = serial.Serial(port, baudrate=115200, timeout=1)
        return handle
    except Exception as e:
        print(f"Error connecting to {port}: {e}")
        return None

def write_data_to_device(handle, cmd):
    handle.write(cmd)
    print("Sent:", " ".join(f"{byte:02X}" for byte in cmd))

def read_ack(handle):
    read_buffer = handle.read(4)
    print("ACK-Frame:", " ".join(f"{byte:02X}" for byte in read_buffer))
    return read_buffer[2] == 0x83

def read_data(handle, bytes_to_read):
    read_buffer = handle.read(bytes_to_read)
    return read_buffer

def set_eit_setup(handle, burst_count, precision):
    cmd = bytearray([0xC4, 0x01, 0x01, 0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)
    cmd = bytearray([0xC4, 0x03, 0x02]) + struct.pack(">H", burst_count) + bytearray([0xC4])
    print(f"Cmd: {cmd}")
    write_data_to_device(handle, cmd)
    read_ack(handle)
    cmd = bytearray([0xC4, 0x05, 0x03]) + struct.pack(">H", precision) + bytearray([0xC4])
    print(f"Cmd: {cmd}")
    write_data_to_device(handle, cmd)
    read_ack(handle)

def get_eit_setup(handle):
    cmd = bytearray([0xC5, 0x01, 0x02, 0xC5])
    write_data_to_device(handle, cmd)
    read_buffer = read_data(handle, 4)
    burst_count = struct.unpack(">H", read_buffer[2:4])[0]
    print(f"Read buffer: {read_buffer}")
    print(f"Burst count: {burst_count}")

    cmd = bytearray([0xC5, 0x01, 0x03, 0xC5])
    write_data_to_device(handle, cmd)
    read_buffer = read_data(handle, 4)
    print(f"Read buffer: {read_buffer}")
    precision = struct.unpack(">H", read_buffer[2:6])[0]

    return burst_count, precision

def set_excitation_frequencies(handle, fmin, fmax, fcount, ftype):
    cmd = bytearray([0xC4, 0x0C, 0x04])
    cmd += struct.pack(">f", fmin)
    cmd += struct.pack(">f", fmax)
    cmd += struct.pack(">H", fcount)
    cmd += struct.pack("B", ftype)
    cmd += bytearray([0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)

def main():
    port = "COM3"  # Replace with the correct serial port name
    handle = connect_to_device(port)

    if not handle:
        return

    print("Connection established")
    handle.flush()

    # Initialize Setup
    # print("Initialize Setup.")
    # cmd = bytearray([0xB6, 0x01, 0x01, 0xB6])
    # write_data_to_device(handle, cmd)
    # read_ack(handle)
    # print()

    # Set EIT Setup
    print("Set EIT Setup: Burst-Count=1, Precision=1")
    set_eit_setup(handle, 7, 1)
    time.sleep(0.1)
    print(get_eit_setup(handle))

    # # Initialize Freq.Block
    # start_frequency = 100  # 100Hz
    # stop_frequency = 100e3  # 100kHz
    # frequency_count = 80
    # precision = 1
    # amplitude = 0.1  # V
    # scale = 1  # log
    # print("Set Setup-config: Frequency-block (%fHz .. %fHz, scale=%i, prec=%f, amplitude=%fV)" %
    #       (start_frequency, stop_frequency, scale, precision, amplitude))
    #
    # cmd = bytearray([0xB6, 0x16, 0x03])
    # cmd += struct.pack(">f", start_frequency)
    # cmd += struct.pack(">f", stop_frequency)
    # cmd += struct.pack(">f", frequency_count)
    # cmd.append(scale)
    # cmd += struct.pack(">f", precision)
    # cmd += struct.pack(">f", amplitude)
    # cmd.append(0xB6)
    # write_data_to_device(handle, cmd)
    # read_ack(handle)
    # print()
    #
    # # Set FrontEnd Settings
    # print("Clear Channel Stack")
    # cmd = bytearray([0xB0, 0x03, 0xFF, 0xFF, 0xFF])
    # write_data_to_device(handle, cmd)
    # read_ack(handle)
    # print()
    #
    # print("Set Channel Stack: to channel 1")
    # cmd[2] = 0x02
    # cmd[3] = 0x01
    # cmd[4] = 0x01
    # write_data_to_device(handle, cmd)
    # read_ack(handle)
    # print()
    #
    # # Start Measurement
    # number_of_specs = 1
    # print("Start Measurement")
    # cmd[0] = 0xB8
    # cmd[2] = 0x01
    # write_data_to_device(handle, cmd)
    # read_ack(handle)
    # print()
    #
    # # Receive Specs
    # print("Receiving Specs:")
    # read_buffer_size = 14
    # for j in range(number_of_specs):
    #     print("Spec#%i:" % (j + 1))
    #     print("ch\tid\tre\tim")
    #     for i in range(int(frequency_count)):
    #         read_buffer = read_data(handle, read_buffer_size)
    #         ch, id_number, re, im = struct.unpack(">BHLf", read_buffer)
    #         print(f"{ch}\t{id_number}\t{re}\t{im}")
    #     print()
    #
    # # Stop Measurement
    # print("Stop Measurement")
    # cmd[2] = 0x00
    # write_data_to_device(handle, cmd)
    # read_ack(handle)
    # print()
    #
    # handle.close()
    # print("Serial connection closed.")


if __name__ == "__main__":
    main()