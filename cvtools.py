import numpy as np
import cv2 as cv
from landmarks_detector import LandmarksDetector
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


class CVTools:
    def __init__(self):
        self.detector = cv.FaceDetectorYN.create(
            model='models/face_detection_yunet_2022mar.onnx',
            # https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx
            config='',
            input_size=[320, 320],
            score_threshold=0.9,  # Filter out faces of confidence < conf_threshold.
            nms_threshold=0.3,  # Suppress bounding boxes of iou >= nms_threshold.
            top_k=5000,  # Keep top_k bounding boxes before NMS.
            backend_id=0,  # DNN_BACKEND_OPENCV, DNN_BACKEND_CUDA
            target_id=0  # DNN_TARGET_CPU, DNN_TARGET_CUDA, DNN_TARGET_CUDA_FP16
        )
        self.recognizer = cv.FaceRecognizerSF.create(
            model='models/face_recognition_sface_2021dec.onnx',
            # https://github.com/opencv/opencv_zoo/blob/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
            config='',
            backend_id=0,
            target_id=0
        )
        self.landmarks_detector = LandmarksDetector(
            # https://github.com/junhwanjang/face_landmark_dnn/blob/master/landmark_model/Mobilenet_v1.hdf5
            model='models/landmark_model.pb'
        )
        self.data = None

    def get_faces(self, image):
        h, w, _ = image.shape
        self.detector.setInputSize((w, h))
        faces = self.detector.detect(image)[1]
        return faces if faces is not None else []

    def get_embedder(self, image, face):
        aligned = self.recognizer.alignCrop(image, face[::-1])
        embedder = self.recognizer.feature(aligned)
        return embedder

    def get_landmark(self, image, face):
        return self.landmarks_detector.detect(image, face)

    def vectorize_landmark(self, landmark):
        return self.landmarks_detector.vectorize(landmark)

    @staticmethod
    def get_model(titles, data):
        encoder = LabelEncoder()
        labels = encoder.fit_transform(titles)
        model = SVC(C=1.0, kernel='linear', probability=True)
        model.fit(data, labels)
        return encoder, model

    def learn_model(self, path, people_names, embedders, emotions_titles, vectorized_landmarks):
        names_encoder, names_model = self.get_model(people_names, embedders)
        emotions_encoder, emotions_model = self.get_model(emotions_titles, vectorized_landmarks)
        with open(path, 'wb') as f:
            data = {
                'names_encoder': names_encoder,
                'names_model': names_model,
                'emotions_encoder': emotions_encoder,
                'emotions_model': emotions_model
            }
            self.data = data
            pickle.dump(data, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.data = data

    @staticmethod
    def predict_model(encoder, model, data):
        predicts = model.predict_proba(data)[0]
        i = np.argmax(predicts)
        title = encoder.classes_[i]
        probability = predicts[i]
        return title, probability

    def predict_name_and_emotion(self, embedder, vectorized_landmark):
        name = self.predict_model(self.data['names_encoder'], self.data['names_model'], embedder)
        emotion = self.predict_model(self.data['emotions_encoder'], self.data['emotions_model'], vectorized_landmark)
        return name, emotion

    @staticmethod
    def draw_faces(image, faces, color):
        for face in faces:
            box = face[0:4].astype(np.int32)
            cv.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)

    @staticmethod
    def write_texts(image, faces, texts, color):
        for face, text in zip(faces, texts):
            cv.putText(image, text, (int(face[0]), int(face[1]) - 10), cv.FONT_HERSHEY_COMPLEX, 0.45, color, 2)

    @staticmethod
    def draw_landmarks(image, landmarks, color):
        for landmark in landmarks:
            for point in landmark:
                x, y = point.astype(np.int32)
                cv.circle(image, (x, y), 2, (255, 255, 255), -1, cv.LINE_AA)

    @staticmethod
    def read_image(path):
        return cv.imread(path)

    @staticmethod
    def save_image(path, image):
        cv.imwrite(path, image)

    @staticmethod
    def get_stream():
        return cv.VideoCapture(0)
