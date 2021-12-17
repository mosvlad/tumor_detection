import cv2

import numpy as np

from tensorflow.keras.preprocessing import image
from keras.models import load_model


class Evaluator:

    @staticmethod
    def __load_image(img_path, w=200, h=200):
        img = image.load_img(img_path, target_size=(w, h))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        return img_tensor

    @staticmethod
    def __show_result(img_path, probability):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (800, 800))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (25, 25)
        font_scale = 1
        color = (10, 10, 255)
        thickness = 1

        img = cv2.putText(img, 'tumor probability:' + str(probability[0][0] * 100.0)[0:5], org, font,
                          font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("tumor probability", img)
        cv2.waitKey(0)

    def evaluate(self, model_path, image_path):
        model = load_model(model_path)
        print(model.summary())

        img_path = image_path
        input_image = self.__load_image(img_path)
        tumor_prob = model.predict(input_image)

        self.__show_result(img_path, tumor_prob)
