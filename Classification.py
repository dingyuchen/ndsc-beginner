#Classification
import csv
import pickle
import pandas as pd

imgToCat = {}
train = open('./train.csv', "r")
trainData = train.read()
trainData = trainData.split('\n')
trainData = [line.split(',') for line in trainData][1:-1]
print(trainData[-1])

for i in range(1, len(trainData) - 1):
    imgName = trainData[i][-1].split('/')[-1]
    if '.jpg' not in imgName: 
        imgName += '.jpg'
    category = int(trainData[i][-2])
    imgToCat[imgName] = category


# print(imgToCat)

from PIL import Image
import numpy as np
import os
from random import shuffle
# import matplotlib.pyplot as plt

DIR = './ndsc/fashion_image/'

def label_img(filename):
    numericalCategory = imgToCat[filename]
    
    output = np.zeros(58)
    output[numericalCategory] = 1
    return output

IMG_SIZE = 300

count = 200

def load_training_data(dir):
    train_data = []
    for img in os.listdir(dir):
        try:
            label = label_img(img)
            path = os.path.join(dir, img)
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            train_data.append([np.array(img), label])
            if len(train_data) > count:
                break
        except Exception as e:
            pass
            
    shuffle(train_data)
    return train_data

train_data = []
test_data = []
# try:
#     with open("./data.pkl", "rb") as f:
#         train_data = pickle.load(f)
# except FileNotFoundError:
#     fashion_data = load_training_data(DIR)
#     DIR = './ndsc/mobile_image/'
#     mobile_data = load_training_data(DIR)
#     DIR = './ndsc/beauty_image/'
#     beauty_data = load_training_data(DIR)
#     train_data = fashion_data[:count // 2] + beauty_data[:count // 2] + mobile_data[:count // 2]
#     test_data = fashion_data[count // 2:] + beauty_data[count // 2:] + mobile_data[count // 2:]
#     with open("./data.pkl", "wb") as f:
#         pickle.dump(train_data, f)

fashion_data = load_training_data(DIR)
mobile_data = load_training_data('./ndsc/mobile_image/')
beauty_data = load_training_data('./ndsc/beauty_image/')
train_data = fashion_data[:(count // 2)] + beauty_data[:(count // 2)] + mobile_data[:(count // 2)]
test_data = fashion_data[(count // 2):] + beauty_data[(count // 2):] + mobile_data[(count // 2):]

train_data = fashion_data + beauty_data + mobile_data

# plt.imshow(train_data[43][0], cmap = 'gist_gray')
print(len(train_data))
trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in train_data])

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np 



try:
    with open("./model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(58, activation = 'softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    # print(len(trainImages))
    # print(len(trainLabels))
    model.fit(trainImages, trainLabels, batch_size = 50, epochs = 5, verbose = 1)
    with open("./model.pkl", "wb") as f:
            pickle.dump(model, f)


# Test on Test Set
TEST_DIR = './test'
# def load_test_data():
#     test_data = []
#     for img in os.listdir(TEST_DIR):
#         label = label_img(img)
#         path = os.path.join(TEST_DIR, img)
#         if "DS_Store" not in path:
#             img = Image.open(path)
#             img = img.convert('L')
#             img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
#             test_data.append([np.array(img), label])
#     shuffle(test_data)
#     return test_data
 
# plt.imshow(test_data[10][0], cmap = 'gist_gray')

testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
testLabels = np.array([i[1] for i in test_data])

loss, acc = model.evaluate(testImages, testLabels, verbose = 0)
print(acc * 100)



test_csv = pd.read_csv("./test.csv", index_col="itemid")
def lookup_photo(id):
    # return the full path to the image
    return test_csv.loc[id][1]


def decode(datum):
    return np.argmax(datum)

item_ids = []
def create_photos(pd_df):
    photos = []

    for index, row in pd_df.iterrows():
        try:
            img = lookup_photo(index)
            path = os.path.join("./ndsc/", img)
            img = Image.open(path)
            img = img.convert("L")

            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            photos.append([np.array(img)])
            item_ids.append(index)
            print(path)
            # break
        except FileNotFoundError:
            continue

    shuffle(photos)
    return photos


test_set = create_photos(test_csv)
test_set = np.array([i[0] for i in test_set]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

result = model.predict(test_set)

result = [decode(num_arr) for num_arr in result]

output = pd.DataFrame(columns=['itemid', 'Category'])

for i in range(0, len(result)):
    output.loc[i] = [item_ids[i], result[i]]

output.to_csv("./output.csv", index=False)

