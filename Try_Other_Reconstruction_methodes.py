import numpy
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt

image = numpy.asarray(Image.open('Experiments/cat.jpg')).mean(axis=2)

plt.figure(figsize=[20, 10])
plt.imshow(image, cmap='gray')
plt.show()


def train_display(regressor, image, train_size=0.02, title=None):
    height, width = image.shape
    flat_image = image.reshape(-1)
    xs = numpy.arange(len(flat_image)) % width
    ys = numpy.arange(len(flat_image)) // width
    data = numpy.array([xs, ys]).T
    target = flat_image
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=train_size, random_state=42)
    mean = trainY.mean()
    regressor.fit(trainX, trainY - mean)
    new_flat_picture = regressor.predict(data) + mean
    plt.figure(figsize=[20, 10])
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.subplot(122)
    plt.imshow(new_flat_picture.reshape(height, width), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


train_display(LinearRegression(), image, title="LinearRegression")
train_display(DecisionTreeRegressor(max_depth=40), image, title="DecisionTreeRegressor")
train_display(RandomForestRegressor(n_estimators=100), image, title="RandomForestRegressor")
train_display(KNeighborsRegressor(n_neighbors=2), image, title="KNeighborsRegressor")
train_display(GradientBoostingRegressor(), image, title="GradientBoostingRegressor")
train_display(AdaBoostRegressor(), image, title="AdaBoostRegressor")
train_display(BaggingRegressor(), image, title="BaggingRegressor")
