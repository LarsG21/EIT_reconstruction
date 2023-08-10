import numpy as np

from pyeit.mesh.wrapper import PyEITAnomaly_Circle


def generate_random_anomaly_list(max_number_of_anomalies, min_radius, max_radius, min_perm, max_perm, outer_circle_radius=0.8):
    """
    Generates a list of random anomalies
    :param max_number_of_anomalies: maximum number of anomalies
    :param min_radius: minimum radius of the anomaly circle
    :param max_radius: maximum radius of the anomaly circle
    :param min_perm: minimum permittivity of the anomaly circle
    :param max_perm: maximum permittivity of the anomaly circle
    :param outer_circle_radius: radius of the outer circle
    :return: list of anomalies
    """
    anomaly_list = []
    if max_number_of_anomalies != 1:
        number_of_anomalies = np.random.randint(1, max_number_of_anomalies)
    else:
        number_of_anomalies = 1
    for i in range(number_of_anomalies):
        center, r, perm = generate_random_anomaly_parameters(min_radius, max_radius, min_perm, max_perm, outer_circle_radius)
        anomaly_list.append(PyEITAnomaly_Circle(center=center, r=r, perm=perm))
    return anomaly_list


def generate_random_anomaly_parameters(min_radius, max_radius, min_perm, max_perm, outer_circle_radius=0.8):
    """
    Generates random parameters for an anomaly circle inside a Tank
    :param min_radius: minimum radius of the anomaly circle
    :param max_radius: maximum radius of the anomaly circle
    :param min_perm: minimum permittivity of the anomaly circle
    :param max_perm: maximum permittivity of the anomaly circle
    :param outer_circle_radius: radius of the outer circle
    :return: center, radius, permittivity of the anomaly circle
    """
    if max_radius + outer_circle_radius > 1:
        raise ValueError("max_radius + outer_circle_radius > 1 --> anomaly can be outside the circle")
    center = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    # allow only center position that are inside a circle of radius 1
    while np.linalg.norm(center) > outer_circle_radius:
        center = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    r = np.random.uniform(min_radius, max_radius)
    perm = np.random.uniform(min_perm, max_perm)
    return center, r, perm
