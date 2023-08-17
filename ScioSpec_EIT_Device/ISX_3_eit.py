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


def set_eit_setup_burst(handle, burst_count):
    """
    Sets the EIT setup burst count (Number of frames generated before measurement stops automatically.)
    For continuous streaming set to 0
    :param handle:
    :param burst_count:Number of frames generated before measurement stops automatically [0-65535]
    :return:
    """
    # convert burst count to 2 bytes
    burst_count = burst_count.to_bytes(2, byteorder='big')
    burst_count_array = bytearray(burst_count)
    cmd = bytearray([0xC4, 0x03, 0x02]) + burst_count_array + bytearray([0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)

def get_eit_setup_burst(handle):
    """
    Gets the EIT setup burst count (Number of frames generated before measurement stops automatically.)
    :param handle:
    :return:
    """
    cmd = bytearray([0xC5, 0x01, 0x02, 0xC5])
    write_data_to_device(handle, cmd)
    read_buffer = read_data_until_command_ends(handle)
    print(f"Read buffer: {read_buffer}")
    if read_ack(handle):
        burst_count = int.from_bytes(read_buffer[3:5], byteorder='big')
       #  return result as dictionary
        return {"burst_count": burst_count}
    else:
        return None

def set_eit_setup_precision(handle, precision):
    """
    Sets the EIT setup precision (Precision of the measurement in ms)
    :param handle:
    :param precision: Precision of the measurement [1-10]
    :return:
    """
    cmd = bytearray([0xC4, 0x02, 0x03, precision, 0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)

def get_eit_setup_precision(handle):
    """
    Gets the EIT setup precision (Precision of the measurement in ms)
    :param handle:
    :return:
    """
    cmd = bytearray([0xC5, 0x01, 0x03, 0xC5])
    write_data_to_device(handle, cmd)
    read_buffer = read_data_until_command_ends(handle)
    print(f"Read buffer: {read_buffer}")
    if read_ack(handle):
        precision = read_buffer[3]
        # return result as dictionary
        return {"precision": precision}
    else:
        return None

def set_eit_setup_frequency(handle, f_min, f_max, f_count, f_type):
    """
    Sets the EIT setup frequency range.
    Add excitation frequency block in Hz. To build a custom excitation frequency stack the command can be send
    multiple times. The frequencies are placed in the stack as they are configured. The total number of configured
    excitation frequencies must not exceed the excitation frequency stack length of 128.
    :param handle:
    :param f_min: Minimum frequency in Hz [0.1-10M] # 4 bytes
    :param f_max: Maximum frequency in Hz [0.1-10M] # 4 bytes
    :param f_count: Number of frequencies [1-128] # 2 byte unsigned
    :param f_type: Type of frequencies [0-1] # 1 byte   0: Linear, 1: Logarithmic
    :return:
    """
    # convert fmin and fmax to 4 byte array
    bytes_f_min_array = convert_float_to_bytes(f_min)

    bytes_f_max_array = convert_float_to_bytes(f_max)

    # bytes_f_min_array = bytearray.fromhex('44 7a 00 00')
    # bytes_f_max_array = bytearray.fromhex('44 fa 00 00')
    # convert f count to 2 byte array
    bytes_f_count = f_count.to_bytes(2, byteorder='big')
    bytes_f_count_array = bytearray(bytes_f_count)
    cmd = bytearray([0xC4, 0x0C, 0x04]) + bytes_f_min_array + bytes_f_max_array + bytes_f_count_array + bytearray([f_type, 0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)


def convert_float_to_bytes(float_value):
    """
    Converts a float value to a 4 byte array by the IEEE 754 standard.
    :param float_value:
    :return:
    """
    b = struct.pack('f', float_value)
    bytes_array = bytearray(b)
    # change endianness
    bytes_array.reverse()
    return bytes_array


def get_eit_setup_frequency(handle):
    """
    Gets the EIT setup frequency range.
    :param handle:
    :return:
    """
    cmd = bytearray([0xC5, 0x01, 0x04, 0xC5])
    write_data_to_device(handle, cmd)
    read_buffer = read_data_until_command_ends(handle)
    print(f"Read buffer: {read_buffer}")
    if read_ack(handle):
        f_min = read_buffer[3]
        f_max = read_buffer[4]
        f_count = read_buffer[5]
        f_type = read_buffer[6]
        # return result as dictionary
        return {"f_min": f_min, "f_max": f_max, "f_count": f_count, "f_type": f_type}
    else:
        return None

def set_eit_amplitude(handle, amplitude):
    """
    Sets the EIT setup amplitude.
    :param handle:
    :param amplitude: could be either 32Bit float ([LE] = 05) or 64Bit float ([LE] = 09) (double)
                        excitation amplitude in Ampere
                        range 0.00001A - 0.01A
    :return:
    """
    if amplitude < 0.00001 or amplitude > 0.01:
        raise ValueError("Amplitude must be between 0.00001 and 0.01")

    amplitude_bytes = convert_float_to_bytes(amplitude)
    cmd = bytearray([0xC4, 0x05, 0x05]) + amplitude_bytes + bytearray([0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)

def get_eit_amplitude(handle):
    """
    Gets the EIT setup amplitude.
    :param handle:
    :return:
    """
    cmd = bytearray([0xC5, 0x01, 0x05, 0xC5])
    write_data_to_device(handle, cmd)
    read_buffer = read_data_until_command_ends(handle)
    print(f"Read buffer: {read_buffer}")
    if read_ack(handle):
        amplitude = read_buffer[2:6]
        print(f"Amplitude: {amplitude}")
        # return result as dictionary
        return {"amplitude": amplitude}
    else:
        return None


def add_extraction_setting(handle, electrode_in, electrode_out):
    """
    Adds an extraction sequence to the EIT setup.
    :param handle:
    :param electrode_in:
    :param electrode_out:
    :return:
    """
    cmd = bytearray([0xC4, 0x03, 0x06, electrode_in, electrode_out, 0xC4])
    write_data_to_device(handle, cmd)
    print(f"Added extraction setting from {electrode_in} to {electrode_out}")
    read_ack(handle)


def add_extraction_pattern(handle, num_electrodes):
    """
    Adds an extraction pattern to the EIT setup.
    Uses add_extraction_setting to add settings from 1 -> 2, 2 -> 3, ..., n-1 -> n
    Ends with n -> 1
    :param handle:
    :param num_electrodes:
    :return:
    """
    for i in range(1, num_electrodes):
        add_extraction_setting(handle, i, i + 1)
    add_extraction_setting(handle, num_electrodes, 1)


def get_extraction_pattern(handle):
    """
    Gets the extraction pattern from the EIT setup.
    :param handle:
    :return:
    """
    cmd = bytearray([0xC5, 0x01, 0x06, 0xC5])
    write_data_to_device(handle, cmd)
    read_buffer = read_data_until_command_ends(handle)
    print(f"Read buffer: {read_buffer}")


def set_measurement_channels(handle, channel_list):
    """
    Sets the measurement channels for the EIT setup.
    :param handle:
    :param channel_list:
    :return:
    """
    cmd = bytearray([0xC4, len(channel_list), 0x08]) + bytearray(channel_list) + bytearray([0xC4])
    write_data_to_device(handle, cmd)
    read_ack(handle)


def get_measurement_channels(handle):  # TODO: Not mentioned in the manual
    """
    Gets the measurement channels for the EIT setup.
    :param handle:
    :return:
    """
    cmd = bytearray([0xC5, 0x01, 0x08, 0xC5])
    write_data_to_device(handle, cmd)
    read_buffer = read_data_until_command_ends(handle)
    print(f"Read buffer: {read_buffer}")

def reset_eit_setup(handle):
    cmd = bytearray([0xB8, 0x03, 0x1, 0x01, 0x00, 0xB8])
    write_data_to_device(handle, cmd)
    read_ack(handle)


def start_measurement(handle, number_of_measurments):
    # cmd = bytearray([0xB8, 0x01, on_off, 0xB8])
    # bytes_nr_measurements = number_of_measurments.to_bytes(2, byteorder='big')

    # cmd = bytearray([0xB8, 0x03 , 0x01]) + bytes_nr_measurements + bytearray([0xB8])
    # cmd = bytearray([0xB8, 0x03, 0x1, 0x01, 0x00, 0xB8])
    cmd = bytearray([0xB8, 0x03, 0x02, 0x00, 0x00, 0xB8])

    write_data_to_device(handle, cmd)
    read_ack(handle)


def main():
    port = "COM3"  # Replace with the correct serial port name
    handle = connect_to_device(port)

    if not handle:
        return

    print("Connection established")
    handle.flush()

    reset_setup(handle)
    #
    # save_settings(handle)
    # set_options(handle, 0x01, 0x00)
    # print(get_options(handle, option_id=0x01))

    reset_eit_setup(handle)

    set_eit_setup_burst(handle, 0)
    print(get_eit_setup_burst(handle))
    #
    set_eit_setup_precision(handle, 0x01)
    print(get_eit_setup_precision(handle))

    set_eit_setup_frequency(handle, 1, 2, 1, 0)
    # print(get_eit_setup_frequency(handle))

    set_eit_amplitude(handle, 0.01)
    # print(get_eit_amplitude(handle))

    add_extraction_pattern(handle, 32)
    get_extraction_pattern(handle)
    # generate a list from 1 to 32
    channel_list = list(range(1, 33))
    set_measurement_channels(handle, channel_list)

    start_measurement(handle, 0x01)


if __name__ == "__main__":
    main()
