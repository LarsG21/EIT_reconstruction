import torch
from matplotlib import pyplot as plt
from torchviz import make_dot

from Models import LinearModelWithDropout

VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64
# Visualize the model

model = LinearModelWithDropout(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
model.load_state_dict(torch.load(
    "../Collected_Data/Combined_dataset/Models/LinearModelDropout/old/30_08_with_noise_and_rotation_augmentation/model_2023-08-30_14-15-52_200_epochs.pth"))

x = torch.randn(1, 1024)
y = model(x)
make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]), show_attrs=False, show_saved=True).render(
    "Linear_Dropout", format="png")

torch.onnx.export(model, x, "model.onnx")
