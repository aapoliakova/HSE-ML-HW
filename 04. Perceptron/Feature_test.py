import numpy as np
from sklearn import datasets
from collections import Counter
import seaborn as sns
import matplotlib as plt

data = datasets.load_digits()
images, labels = data.images, data.target
mask = np.logical_or(labels == 1, labels == 5)
labels = labels[mask]
images = images[mask]

# Идеи для фич
# 1. Последовательность пикселей вертикально - у единицы должно быть больше
# 2. У 5рки должны быть больше разннобразных цифр, чем у единицы т к она в целом больше и ее рисуют в два штриха
#

print(images.shape)
images[images < 12] = 0
w_sym = np.abs(images - images[:, :, ::-1]).mean(axis=(1, 2))
darkness = []
for image in images:
    darkness.append(np.mean(image))
darkness = np.array(darkness)
print(darkness.shape)
print(w_sym.shape)

def longest_sequence(image):
    for column in image.T:
        max_per_column = []
        long_cur = 0
        pretend_to_be_longest = []
        for el in column:
            print(column)
            if el != 0:
                long_cur += 1
            else:
                pretend_to_be_longest.append(long_cur)
                long_cur = 0
                # print(pretend_to_be_longest)
        max_per_column.append(max(pretend_to_be_longest))



print(longest_sequence(images[0]))
# sns.scatterplot(x=w_sym, y=darkness, s=100, hue = labels).set_title('True labels')
