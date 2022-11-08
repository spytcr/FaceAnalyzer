import cv2 as cv
import numpy as np
import math


class LandmarksDetector:
    def __init__(self, model):
        self.model = model
        self.detector = cv.dnn.readNetFromTensorflow(model)

    def detect(self, image, facebox):
        x, y, w, h = facebox[0:4].astype(np.int32)
        face = image[y:y + h, x:x + w]  # Face getting from image
        if face.shape[1] == 0:
            return []
        face = cv.resize(face, (64, 64))  # Face resizing to model input size
        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # Face converting to model input color
        blob = cv.dnn.blobFromImage(face, 1, (64, 64), 0)
        self.detector.setInput(blob)
        landmarks = self.detector.forward()  # Landmarks getting
        landmarks = np.array(landmarks).flatten()
        landmarks = np.reshape(landmarks, (-1, 2))  # Landmarks formatting
        landmarks *= (w, h)
        landmarks += (x, y)  # Landmark alignment
        return landmarks

    @staticmethod
    def vectorize(landmarks):
        x_mean = np.mean(landmarks[:, 0])
        y_mean = np.mean(landmarks[:, 1])
        result = np.empty(shape=landmarks.shape[0] * 4, dtype=np.float32)
        for i, (x, y) in enumerate(landmarks):
            dist = np.linalg.norm((y - y_mean, x - x_mean))  # Distance of each landmark from center of gravity
            angle = (math.atan2(y - y_mean, x - x_mean) * 360) / (2 * math.pi)  # Angle of each landmark
            result[4 * i:4 * (i + 1)] = x, y, dist, angle
        return result.reshape(1, -1)


