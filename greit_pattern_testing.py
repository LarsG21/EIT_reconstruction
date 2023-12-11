import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RADIUS_TARGET_IN_MM = 20
RADIUS_TANK_IN_MM = 190

RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM

radius_target = 0.05
radius_tank = 0.9

circumference_tank = 2 * np.pi * radius_tank
# how often does the target fit into the circumference of the tank
num_targets_in_circumstance = int(circumference_tank / radius_target)
# how often does the target fit into the radius of the tank
num_targets_in_radius = int(radius_tank / radius_target)
print(f"Number of targets in circumference: {num_targets_in_circumstance}")
print(f"Number of targets in radius: {num_targets_in_radius}")


def generate_points_in_circle(radius, num_points):
    # Generate evenly spaced angles
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    print(f"Length of angles: {len(angles)}")

    # Convert polar coordinates to Cartesian coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    return x, y, radius, angles


overall_pints = 0
df_coords_complete = pd.DataFrame(columns=["x", "y", "radius", "angles"])
for i in range(1, num_targets_in_radius):
    df_coords = pd.DataFrame(columns=["x", "y"])
    radius = i * radius_target
    num_targets_in_circumstance = int(radius / radius_target) * 3
    overall_pints += num_targets_in_circumstance
    x, y, radius, angles = generate_points_in_circle(radius, num_targets_in_circumstance)
    df_coords["x"] = x
    df_coords["y"] = y
    df_coords["radius"] = radius
    df_coords["angles"] = angles
    df_coords_complete = pd.concat([df_coords_complete, df_coords], ignore_index=True)
    plt.scatter(x, y)
    plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio ensures a circular plot
    plt.title(f'Points Evenly Distributed in a Circle of Radius {radius}')
plt.show()
print(f"Overall points: {overall_pints}")
print(f"Length of df: {len(df_coords_complete)}")
# save the dataframe
df_coords_complete.to_csv("points.csv", index=False)
