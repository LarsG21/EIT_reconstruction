import torch
from matplotlib import pyplot as plt
from torchviz import make_dot

from Models import LinearModelWithDropout


def get_onnx_of_model():
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


VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64
model = LinearModelWithDropout(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
# Create a random input tensor for visualization
dummy_input = torch.randn(1, 1024)  # Adjust the size according to your input dimensions

# Pass the dummy input through the model to generate the computation graph
output = model(dummy_input)

# Generate the computation graph and save the plot
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("unet_graph_LinearModelWithDropout", format="png")  # Save the graph as an image file
