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
    # print("ACK-Frame:", " ".join(f"{byte:02X}" for byte in read_buffer))
    if read_buffer[2] == 0x83:
        print("ACK: OK")
    elif read_buffer[2] == 0x01:
        print("ACK: Invalid syntax")
    elif read_buffer[2] == 0x02:
        print("ACK: Timeout error")
    elif read_buffer[2] == 0x04:
        print("ACK:  Wake-Up Message: System boot ready")
    elif read_buffer[2] == 0x11:
        print("ACK: CP-Socket: Valid TCP client-socket connection")
    elif read_buffer[2] == 0x81:
        print("ACK: Not-Acknowledge: Command has not been executed")
    elif read_buffer[2] == 0x82:
        print("ACK: Not-Acknowledge: Command could not be recognized")
    elif read_buffer[2] == 0x84:
        print("ACK:  System-Ready Message: System is operational and ready to receive data")
    return read_buffer[2] == 0x83


def read_data(handle, bytes_to_read):
    read_buffer = handle.read(bytes_to_read)
    return read_buffer


def read_data_until_command_ends(handle):
    """
    Reads data until the command ends. Command ends when the last byte is the same as the first byte.
    :param handle:
    :return:
    """
    read_buffer = bytearray()
    read_buffer += handle.read(1)
    while True:
        read_buffer += handle.read(1)
        if read_buffer[-1] == read_buffer[0]:
            break
    return read_buffer


def reset_setup(handle):
    """
    Resets the setup of the device
    :param handle:
    :return:
    """
    cmd = bytearray([0xC4, 0x01, 0x01, 0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)


def save_settings(handle):
    """
    Saves the current settings
    :param handle:
    :return:
    """
    cmd = bytearray([0x90, 0x00, 0x90])
    write_data_to_device(handle, cmd)
    read_ack(handle)


def get_options(handle, option_id):
    """
    Gets the options [CT] [LE] [OB] [CT]
    :param handle: serial handle
    :param option_id: id of the option to get
    :return: [CT] [LE] [OB] [CD] [CT]
            ACK
    """
    cmd = bytearray([0x98, 0x01, option_id, 0x98])  # Option id 0x01: Gets if timestamp is enabled
    write_data_to_device(handle, cmd)
    read_buffer = read_data_until_command_ends(handle)
    # print(f"Read buffer: {read_buffer}")
    read_ack(handle)
    return read_buffer[-2]


def set_options(handle, option_id, option_value):
    cmd = bytearray([0x97, 0x02, option_id, option_value, 0x97])  # Option id 0x01: Gets if timestamp is enabled
    write_data_to_device(handle, cmd)
    read_ack(handle)


def reset_system(handle):
    """
    complete restart of the system
    :param handle:
    :return: ACK
            Wake-Up Message
            System-Ready-Message
    """
    cmd = bytearray([0xA1, 0x00, 0xA1])
    write_data_to_device(handle, cmd)
    read_ack(handle)


def get_fe_settings(handle):
    """
    Gets the FE settings
    :param handle:
    :return:
    """
    cmd = bytearray([0xB1, 0x00, 0xB1])
    write_data_to_device(handle, cmd)
    read_buffer = read_data_until_command_ends(handle)
    print(f"Read buffer: {read_buffer}")
    if read_ack(handle):
        measurement_mode = read_buffer[3]
        measurement_channel = read_buffer[4]
        range_setting = read_buffer[5]
        # return result as dictionary
        return {"measurement_mode": measurement_mode, "measurement_channel": measurement_channel,
                "range_setting": range_setting}
    else:
        return None


def set_fe_settings(handle, measurment_mode, measurment_channel, range_setting):
    """
    Sets the FE settings
    :param handle:
    :param fe_settings:
    :return:
    """
    cmd = bytearray([0xB0, 0x03, measurment_mode, measurment_channel, range_setting, 0xB0])
    write_data_to_device(handle, cmd)
    read_ack(handle)


def start_eit(handle):
    """
    Starts the EIT measurement
    :param handle:
    :return:
    """
    cmd = bytearray([0xB8, 0x00, 0xB8])
    write_data_to_device(handle, cmd)
    read_ack(handle)


def set_eit_setup(handle, burst_count, precision):
    """
    Sets the EIT setup
    :param handle:
    :param burst_count:
    :param precision:
    :return:
    """
    cmd = bytearray([0xC4, 0x01, 0x01, 0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)
    cmd = bytearray([0xC4, 0x03, 0x02]) + struct.pack(">H", burst_count) + bytearray([0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)
    cmd = bytearray([0xC4, 0x05, 0x03]) + struct.pack(">H", precision) + bytearray([0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)


# def get_eit_setup(handle):
#     cmd = bytearray([0xC5, 0x01, 0x02, 0xC5])
#     write_data_to_device(handle, cmd)
#     read_buffer = read_data(handle, 4)
#     burst_count = struct.unpack(">H", read_buffer[2:4])[0]
#     print(f"Read buffer: {read_buffer}")
#     print(f"Burst count: {burst_count}")
#
#     cmd = bytearray([0xC5, 0x01, 0x03, 0xC5])
#     write_data_to_device(handle, cmd)
#     read_buffer = read_data(handle, 4)
#     print(f"Read buffer: {read_buffer}")
#     precision = struct.unpack(">H", read_buffer[2:6])[0]
#
#     return burst_count, precision

def set_excitation_frequencies(handle, fmin, fmax, fcount, ftype):
    """
    Sets the excitation frequencies
    :param handle:
    :param fmin:
    :param fmax:
    :param fcount:
    :param ftype:
    :return:
    """
    cmd = bytearray([0xC4, 0x0C, 0x04])
    cmd += struct.pack(">f", fmin)
    cmd += struct.pack(">f", fmax)
    cmd += struct.pack(">H", fcount)
    cmd += struct.pack("B", ftype)
    cmd += bytearray([0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)


# def set_extraction_amplitude(handle):

def main():
    port = "COM3"  # Replace with the correct serial port name
    handle = connect_to_device(port)

    if not handle:
        return

    print("Connection established")
    handle.flush()

    # reset_setup(handle)
    #
    # save_settings(handle)
    # set_options(handle, 0x01, 0x00)
    # print(get_options(handle, option_id=0x01))

    set_fe_settings(handle, 0x01, 0x03, 0x01)

    print(get_fe_settings(handle))

    # reset_system(handle)

    # start_eit(handle)

    # reset_setup(handle)
    #
    #
    # # Set EIT Setup
    # print("Set EIT Setup: Burst-Count=1, Precision=1")
    # set_eit_setup(handle, 20, 20)
    # time.sleep(0.1)
    #
    # start_eit(handle)

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
