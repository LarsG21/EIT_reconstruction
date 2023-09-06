import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import find_center_of_mass


def evaluate_position_error(center_for_moving, gcode_device, img_reconstructed, relative_radius_target):
    """
    Evaluates the position error of the center of mass of the reconstructed image and the center of the target.
    :param center_for_moving:
    :param gcode_device:
    :param img_reconstructed:
    :param relative_radius_target:
    :return:
    """
    center_of_mass = find_center_of_mass(img_reconstructed)
    SCALE_FACTOR = 10
    image_show = np.zeros_like(img_reconstructed)
    img_size = image_show.shape[0]
    image_show = cv2.resize(image_show, (img_size * SCALE_FACTOR, img_size * SCALE_FACTOR),
                            interpolation=cv2.INTER_NEAREST)
    # convert to 3 channel image_show
    image_show = np.stack((image_show, image_show, image_show), axis=2)
    # add circle at center of mass
    # flip x and y of center_of_mass
    radius = int((image_show.shape[0] * relative_radius_target / 2))
    cv2.circle(image_show, (center_of_mass[0] * SCALE_FACTOR, center_of_mass[1] * SCALE_FACTOR), radius,
               (255, 0, 0), 2)
    # add circle at center of target
    # map center_for_moving to image_show size
    center_for_moving = convert_center_for_moving_to_center_in_image(center_for_moving, gcode_device, img_size)
    cv2.circle(image_show, (center_for_moving[0] * SCALE_FACTOR, center_for_moving[1] * SCALE_FACTOR), radius,
               (0, 255, 0), 2)
    print("center of mass", center_of_mass)
    print("center for moving", center_for_moving)
    distance_between_centers = np.linalg.norm(center_of_mass - center_for_moving)
    print("distance between centers", distance_between_centers)
    cv2.putText(image_show, f"Error: {round((distance_between_centers / img_reconstructed.shape[0]) * 100, 2)}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image_show, "Target", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image_show, "Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    plt.imshow(image_show)
    plt.title(f"Distance between centers: {distance_between_centers}")
    plt.show()
    return distance_between_centers


def convert_center_for_moving_to_center_in_image(center_for_moving, gcode_device, img_size):
    """
    Converts the center for moving to the center in the image.
    :param center_for_moving: center for moving in gcode device coordinates
    :param gcode_device: the gcode device
    :param img_size: the size of the image
    :return: center in image coordinates
    """
    # convert center_for_moving to center in image
    center_in_image = (center_for_moving / gcode_device.maximal_limits[0]) * img_size
    # invert y axis and x axis (0,0) for image is (max,max) for gcode device
    center_in_image = (img_size, img_size) - center_in_image
    center_in_image = center_in_image.astype(int)
    return center_in_image


def calculate_amplitude_response(center_for_moving, gcode_device, img_reconstructed, relative_radius_target,
                                 show=False):
    """
    Calculates the amplitude response of the reconstructed image.
    :param center_for_moving:
    :param gcode_device:
    :param img_reconstructed:
    :param show:
    :return:
    """
    # calculate amplitude response (ratio of pixels in the reconstructed image to the theoretical image with the anomaly)
    img_theoretical = np.zeros_like(img_reconstructed)
    center_of_anomaly = convert_center_for_moving_to_center_in_image(center_for_moving, gcode_device,
                                                                     img_reconstructed.shape[0])
    radius = int((img_reconstructed.shape[0] * relative_radius_target / 2))
    cv2.circle(img_theoretical, center_of_anomaly, radius, 1, -1)
    # calculate the intersection of the two images
    intersection = np.logical_and(img_reconstructed, img_theoretical)
    intersection = intersection.astype(np.float32)
    # get reconstructed image at parts of the intersection as mask
    img_reconstructed_intersect = img_reconstructed * intersection
    # calculate the ratio of the pixels in the intersection to the pixels in the theoretical image
    amplitude_response = np.sum(img_reconstructed_intersect) / np.sum(img_theoretical)
    if show:
        print("amplitude response: ", amplitude_response)
        cv2.imshow("theoretical", cv2.resize(img_theoretical, (512, 512)))
        cv2.imshow("reconstructed", cv2.resize(img_reconstructed, (512, 512)))
        cv2.imshow("img_reconstructed_intersect", cv2.resize(img_reconstructed_intersect, (512, 512)))
        cv2.waitKey(4000)
    plt.imshow(img_reconstructed_intersect)
    plt.title(f"Amplitude response: {amplitude_response}")
    plt.show()
    return amplitude_response
