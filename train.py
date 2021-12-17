import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

import cv2
import os


class TumorDetectionNet:

    @staticmethod
    def __train_network(df, model_filename):
        train, test = train_test_split(df, train_size=0.95, random_state=0)
        train_new, valid = train_test_split(train, train_size=0.90, random_state=0)

        print(f"train set shape: {train_new.shape}")
        print(f"test set shape: {test.shape}")
        print(f"validation set shape: {valid.shape}")

        train_datagen = ImageDataGenerator(rescale=1. / 255., rotation_range=40, width_shift_range=0.2,
                                           height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                           horizontal_flip=True, vertical_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
        train_gen = train_datagen.flow_from_dataframe(dataframe=train_new, x_col='filepaths', y_col='labels',
                                                      target_size=(150, 150), batch_size=16, class_mode='binary',
                                                      shuffle=True)
        val_gen = train_datagen.flow_from_dataframe(valid, target_size=(150, 150), x_col='filepaths', y_col='labels',
                                                    class_mode='binary', batch_size=16, shuffle=True)
        test_gen = test_datagen.flow_from_dataframe(test, target_size=(150, 150), x_col='filepaths', y_col='labels',
                                                    class_mode='binary', batch_size=16, shuffle=False)
        base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', input_shape=(200, 200, 3),
                                                             include_top=False)

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        plot_model(model, to_file='images/model_plot.png', show_shapes=True, show_layer_names=True)

        input()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(model_filename, save_best_only=True, verbose=0),
        ]

        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        history = model.fit(train_gen, validation_data=val_gen, epochs=40, callbacks=[callbacks], verbose=1)
        return history

    @staticmethod
    def __plot_train_history(train_history):
        acc = train_history.history['accuracy']
        val_acc = train_history.history['val_accuracy']
        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        epochs_range = range(1, len(train_history.epoch) + 1)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train Set')
        plt.plot(epochs_range, val_acc, label='Val Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Train Set')
        plt.plot(epochs_range, val_loss, label='Val Set')
        plt.legend(loc="best")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Loss')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def __prepare_data(path_to_dataset):
        no_brain_tumor = path_to_dataset + 'no/'
        yes_brain_tumor = path_to_dataset + 'yes/'

        dirlist = [no_brain_tumor, yes_brain_tumor]
        classes = ['No', 'Yes']
        filepaths = []
        labels = []
        for i, j in zip(dirlist, classes):
            filelist = os.listdir(i)
            for f in filelist:
                filepath = os.path.join(i, f)
                filepaths.append(filepath)
                labels.append(j)
        print('filepaths: ', len(filepaths), '   labels: ', len(labels))
        print("=====================================")

        files = pd.Series(filepaths, name='filepaths')
        label = pd.Series(labels, name='labels')
        df = pd.concat([files, label], axis=1)
        df = pd.DataFrame(np.array(df).reshape(253, 2), columns=['filepaths', 'labels'])
        print(df.head())
        print(df['labels'].value_counts())
        print("=====================================")

        return df

    @staticmethod
    def __show_images(df):
        plt.figure(figsize=(16, 8))
        for i in range(30):
            random = np.random.randint(1, len(df))
            plt.subplot(5, 6, i + 1)
            plt.imshow(cv2.imread(df.loc[random, "filepaths"]))
            plt.title(df.loc[random, "labels"], size=14, color="black")
            plt.xticks([])
            plt.yticks([])

        plt.show()

    def train(self, path_to_dataset="archive/", model_filename="Tumor_classifier_model.h5"):
        df = self.__prepare_data(path_to_dataset)
        self.__show_images(df)
        train_history = self.__train_network(df, model_filename)
        self.__plot_train_history(train_history)