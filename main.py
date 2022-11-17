import os
import sys
import time

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLineEdit, QColorDialog, \
    QFileDialog, QWidget, QProgressBar, QTableWidgetItem, QInputDialog
import xlsxwriter

from cvtools import CVTools
from database import Database
from ui.db_manager import Ui_Form
from ui.main_window import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()

        self.setupUi(self)

        self.database = Database('db.sqlite')
        self.cvtools = CVTools()
        self.db_manager = DbManager(self.database)

        self.tab_widget.currentChanged.connect(self.tab_bar_clicked)

        self.learning_viewer = ImageViewer(self.lbl_image_name1, self.image1)
        self.btn_load_images1.clicked.connect(self.load_learning_images)
        self.btn_left_image1.clicked.connect(self.learning_image_left)
        self.btn_right_image1.clicked.connect(self.learning_image_right)
        self.cb_name.currentTextChanged.connect(self.name_selected)
        self.cb_emotion.currentTextChanged.connect(self.emotion_selected)
        self.btn_save_data.clicked.connect(self.save_data)
        self.btn_learn.clicked.connect(self.learn)
        self.btn_delete_image.clicked.connect(self.delete_image)
        self.prepare_learning_content()

        self.recognition_viewer = ImageViewer(self.lbl_image_name2, self.image2)
        self.btn_load_images2.clicked.connect(self.load_recognition_images)
        self.btn_take_photo.clicked.connect(self.take_photo)
        self.btn_left_image2.clicked.connect(self.recognition_image_left)
        self.btn_right_image2.clicked.connect(self.recognition_image_right)
        self.btn_save_images.clicked.connect(self.save_recognition_images)
        self.btn_save_statistic.clicked.connect(self.save_statistics)
        self.btn_run_stream.clicked.connect(self.run_stream)
        self.btn_stop_stream.clicked.connect(self.stop_stream)
        self.stream_manager = StreamManager()
        self.stream_manager.update_frame.connect(self.update_stream)

        self.btn_db_manager.clicked.connect(self.load_db_manager)
        self.cb_show_rect.setChecked(self.database.get_bool('show_rect'))
        self.cb_show_rect.stateChanged.connect(lambda x: self.show_changed(x, 'show_rect'))
        self.btn_rect_color.clicked.connect(lambda: self.color_clicked('rect_color'))
        self.cb_show_text.setChecked(self.database.get_bool('show_text'))
        self.cb_show_text.stateChanged.connect(lambda x: self.show_changed(x, 'show_text'))
        self.btn_text_color.clicked.connect(lambda: self.color_clicked('text_color'))
        self.cb_show_landmarks.setChecked(self.database.get_bool('show_landmarks'))
        self.cb_show_landmarks.stateChanged.connect(lambda x: self.show_changed(x, 'show_landmarks'))
        self.btn_landmarks_color.clicked.connect(lambda: self.color_clicked('landmarks_color'))

    def add_learning_images(self, data):
        images = []
        for path, name, emotion in data:
            if os.path.exists(path):
                image = self.cvtools.read_image(path)
                faces = self.cvtools.get_faces(image)
                r, g, b, _ = QColor(self.database.get_preference('rect_color')).getRgb()
                self.cvtools.draw_faces(image, faces, (b, g, r))
                images.append(LearningImage(path, image, name, emotion, len(faces) == 1))
        self.learning_viewer.add_images(images)
        self.check_learning_images()

    def check_learning_images(self):
        self.learning_content_2.setHidden(len(self.learning_viewer.images) == 0)
        if len(self.learning_viewer.images) > 0:
            self.set_learning_content()

    def set_learning_content(self):
        self.learning_viewer.set_image()
        image = self.learning_viewer.get_current_image()
        self.learning_tools.setHidden(not image.valid)
        self.lbl_learning_error.setHidden(image.valid)
        if image.valid:
            self.cb_name.setCurrentText(image.name)
            self.cb_emotion.setCurrentText(image.emotion)

    def learning_image_left(self):
        self.learning_viewer.pref_image()
        self.set_learning_content()

    def learning_image_right(self):
        self.learning_viewer.next_image()
        self.set_learning_content()

    def name_selected(self, text):
        if len(self.learning_viewer.images) > 0:
            image = self.learning_viewer.get_current_image()
            image.name = text

    def emotion_selected(self, text):
        if len(self.learning_viewer.images) > 0:
            image = self.learning_viewer.get_current_image()
            image.emotion = text

    def save_data(self):
        self.database.clear_data()
        for image in self.learning_viewer.images:
            if image.valid:
                self.database.insert_data((image.path, image.name, image.emotion))

    def delete_image(self):
        self.learning_viewer.delete_image()
        self.check_learning_images()

    def learn(self):
        self.save_data()
        data = self.database.get_data()
        names = [el[1] for el in data]
        emotions = [el[2] for el in data]
        if len(set(names)) < 2:
            self.show_warning('Для построения модели должно быть хотя-бы 2 разных человека')
        elif len(set(emotions)) < 2:
            self.show_warning('Для построения модели должно быть хотя-бы 2 разных эмоции')
        else:
            model_path = QFileDialog.getSaveFileName(self, 'Сохраните модель', '', 'Model file (*.pickle)')[0]
            if model_path != '':
                self.database.update_preference('model_path', model_path)
                progress_bar = QProgressBar(self)
                progress_bar.setMinimum(0)
                progress_bar.setMaximum(0)
                self.statusbar.addWidget(progress_bar)
                embedders, landmarks = [], []
                for path in [el[0] for el in data]:
                    image = self.cvtools.read_image(path)
                    face = self.cvtools.get_faces(image)[0]
                    embedder = self.cvtools.get_embedder(image, face)
                    embedders.append(embedder.flatten())
                    landmark = self.cvtools.vectorize_landmark(self.cvtools.get_landmark(image, face))
                    landmarks.append(landmark.flatten())
                self.cvtools.learn_model(model_path, names, embedders, emotions, landmarks)
                self.statusbar.removeWidget(progress_bar)

    def load_learning_images(self):
        paths = self.get_images()
        self.add_learning_images(zip(paths, [self.database.UNKNOWN_NAME] * len(paths),
                                     [self.database.UNKNOWN_EMOTION] * len(paths)))

    def take_photo(self):
        stream = self.cvtools.get_stream()
        time.sleep(0.25)
        ret, frame = stream.read()
        stream.release()
        if frame is None:
            self.show_warning('Не удалось сделать фотографию, проверьте веб-камеру')
        else:
            path = self.save_image()
            if path != '':
                self.cvtools.save_image(path, frame)
                self.add_learning_images([(path, self.database.UNKNOWN_NAME, self.database.UNKNOWN_EMOTION)])

    def add_recognition_images(self, data):
        images = []
        for path in data:
            if os.path.exists(path):
                image = self.cvtools.read_image(path)
                images.append(RecognitionImage(path, image, self.get_image_content(image)))
        self.recognition_viewer.add_images(images)
        self.set_recognition_content()

    def get_image_content(self, image):
        faces = self.cvtools.get_faces(image)
        content, landmarks = [], []
        for face in faces:
            embedder = self.cvtools.get_embedder(image, face)
            landmark = self.cvtools.get_landmark(image, face)
            landmarks.append(landmark)
            vectorized = self.cvtools.vectorize_landmark(landmark)
            content.append(self.cvtools.predict_name_and_emotion(embedder, vectorized))
        if self.database.get_bool('show_rect'):
            r, g, b, _ = QColor(self.database.get_preference('rect_color')).getRgb()
            self.cvtools.draw_faces(image, faces, (b, g, r))
        if self.database.get_bool('show_text'):
            r, g, b, _ = QColor(self.database.get_preference('text_color')).getRgb()
            self.cvtools.write_texts(image, faces, [el[0][0] + ', ' + el[1][0] for el in content], (b, g, r))
        if self.database.get_bool('show_landmarks'):
            r, g, b, _ = QColor(self.database.get_preference('landmarks_color')).getRgb()
            self.cvtools.draw_landmarks(image, landmarks, (b, g, r))
        return content

    def set_recognition_content(self):
        self.image_content_2.setHidden(len(self.recognition_viewer.images) == 0)
        if len(self.recognition_viewer.images) > 0:
            self.recognition_viewer.set_image()
            image = self.recognition_viewer.get_current_image()
            self.lbl_image_content.setText('На картинке изображены: ' + RecognitionImage.format_content(image.content))

    def recognition_image_left(self):
        self.recognition_viewer.pref_image()
        self.set_recognition_content()

    def recognition_image_right(self):
        self.recognition_viewer.next_image()
        self.set_recognition_content()

    def load_recognition_images(self):
        self.add_recognition_images(self.get_images())

    def get_images(self):
        return QFileDialog.getOpenFileNames(self, 'Выберите фотографии', '', 'Image files (*.png *.jpg *.jpeg)')[0]

    def save_image(self):
        return QFileDialog.getSaveFileName(self, 'Сохранить фотографию', '', 'Image (*.png)')[0]

    def save_recognition_images(self):
        folder = QFileDialog.getExistingDirectory(self, 'Выберите папку для сохранения')
        for image in self.recognition_viewer.images:
            path = folder + '/' + os.path.split(image.path)[1]
            self.cvtools.save_image(path, image.image)

    def run_stream(self):
        self.btn_run_stream.setEnabled(False)
        self.btn_stop_stream.setEnabled(True)
        self.stream_manager.stream = self.cvtools.get_stream()
        self.stream_manager.start()

    @pyqtSlot(object)
    def update_stream(self, frame):
        if frame is not None:
            content = RecognitionImage.format_content(self.get_image_content(frame))
            self.lbl_stream_content.setText('В кадре: ' + content)
            image = cv_image_to_qt(frame, self.stream.width() - 1, self.stream.height() - 1)
            self.stream.setPixmap(image)
        else:
            self.show_warning('Не удалось получить трансляцию, проверьте видеокамеру')
            self.stop_stream()

    def stop_stream(self):
        self.btn_run_stream.setEnabled(True)
        self.btn_stop_stream.setEnabled(False)
        self.stream_manager.stop()
        self.stream_manager.stream.release()
        self.stream_manager.stream = None

    def prepare_learning_content(self):
        self.cb_name.clear()
        self.cb_emotion.clear()
        self.cb_name.addItems([el[1] for el in self.database.get_people()])
        self.cb_emotion.addItems([el[1] for el in self.database.get_emotions()])
        self.learning_viewer.images.clear()
        self.add_learning_images(self.database.get_data())

    def prepare_recognition_content(self):
        data_path = self.database.get_preference('model_path')
        if data_path is None or not os.path.exists(data_path):
            self.tab_widget.setCurrentIndex(0)
            self.show_warning('Для распознавания нужно обучить модель')
        else:
            self.cvtools.load_model(data_path)
            self.image_content_2.setHidden(True)
            self.recognition_viewer.images.clear()

    def save_statistics(self):
        path = QFileDialog.getSaveFileName(self, 'Сохраните статистику', '', 'Excel file (*.xlsx)')[0]
        if path != '':
            workbook = xlsxwriter.Workbook(path)
            worksheet = workbook.add_worksheet('Статистика')
            headers = ['Фотография', 'Люди', 'Эмоции']
            for i, el in enumerate(headers):
                worksheet.write(0, i, el)
            i = 1
            names, emotions = {}, {}
            for image in self.recognition_viewer.images:
                for content in image.content:
                    worksheet.write(i, 0, image.path)
                    worksheet.write(i, 1, content[0][0])
                    worksheet.write(i, 2, content[1][0])
                    if content[0][0] not in names:
                        names[content[0][0]] = 0
                    names[content[0][0]] += 1
                    if content[1][0] not in emotions:
                        emotions[content[1][0]] = 0
                    emotions[content[1][0]] += 1
                    i += 1
            i = 0
            for (k1, v1), (k2, v2) in zip(names.items(), emotions.items()):
                worksheet.write(i, 4, k1)
                worksheet.write(i, 5, v1)
                worksheet.write(i, 7, k2)
                worksheet.write(i, 8, v2)
                i += 1
            names_chart, emotions_chart = workbook.add_chart({'type': 'pie'}), workbook.add_chart({'type': 'pie'})
            names_chart.add_series({'values':  f'=Статистика!F1:F{i}',
                                    'categories': f'=Статистика!E1:E{i}'})
            emotions_chart.add_series({'values':  f'=Статистика!I1:I{i}',
                                       'categories': f'=Статистика!H1:H{i}'})
            worksheet.insert_chart(f'E{i + 1}', names_chart)
            worksheet.insert_chart(f'E{i + 16}', emotions_chart)
            workbook.close()

    def load_db_manager(self):
        self.db_manager.show()

    def show_changed(self, state, name):
        self.database.update_preference(name, str(state == Qt.Checked))

    def color_clicked(self, name):
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self.database.update_preference(name, color.name())

    def show_warning(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def tab_bar_clicked(self, index):
        if index == 0:
            self.prepare_learning_content()
        elif index == 1:
            self.prepare_recognition_content()

    def closeEvent(self, event):
        self.database.close()
        super().closeEvent(event)


class DbManager(QWidget, Ui_Form):
    def __init__(self, database):
        super().__init__()

        self.setupUi(self)

        self.database = database
        self.load_people()
        self.load_emotions()
        self.btn_add_person.clicked.connect(self.add_person)
        self.btn_edit_person.clicked.connect(self.edit_person)
        self.btn_delete_person.clicked.connect(self.delete_person)
        self.btn_add_emotion.clicked.connect(self.add_emotion)
        self.btn_edit_emotion.clicked.connect(self.edit_emotion)
        self.btn_delete_emotion.clicked.connect(self.delete_emotion)

    def add_person(self):
        name, ok_pressed = QInputDialog.getText(self, 'Добавить человека', 'Введите имя')
        if ok_pressed:
            self.database.add_person(name)
        self.load_people()

    def edit_person(self):
        row = self.table_people.currentRow()
        if row != -1:
            name, ok_pressed = QInputDialog.getText(self, 'Изменить имя', 'Введите имя', QLineEdit.Normal,
                                                    self.table_people.item(row, 1).text())
            if ok_pressed:
                self.database.update_person(int(self.table_people.item(row, 0).text()), name)
        self.load_people()

    def delete_person(self):
        row = self.table_people.currentRow()
        if row != -1 and self.show_delete_dialog():
            self.database.delete_person(int(self.table_people.item(row, 0).text()))
        self.load_people()

    def add_emotion(self):
        title, ok_pressed = QInputDialog.getText(self, 'Добавить эмоцию', 'Введите название')
        if ok_pressed:
            self.database.add_emotion(title)
        self.load_emotions()

    def edit_emotion(self):
        row = self.table_emotions.currentRow()
        if row != -1:
            title, ok_pressed = QInputDialog.getText(self, 'Изменить название', 'Введите название', QLineEdit.Normal,
                                                     self.table_emotions.item(row, 1).text())
            if ok_pressed:
                self.database.update_emotion(int(self.table_emotions.item(row, 0).text()), title)
        self.load_emotions()

    def delete_emotion(self):
        row = self.table_emotions.currentRow()
        if row != -1 and self.show_delete_dialog():
            self.database.delete_emotion(int(self.table_emotions.item(row, 0).text()))
        self.load_emotions()

    def show_delete_dialog(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('Удалить запись')
        msg.setText('Вы уверены?')
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return msg.exec() == QMessageBox.Ok

    def load_people(self):
        headers = ['ID', 'Имя']
        data = list(filter(lambda x: x[1] != self.database.UNKNOWN_NAME, self.database.get_people()))
        self.load_data(self.table_people, headers, data)

    def load_emotions(self):
        headers = ['ID', 'Название']
        data = list(filter(lambda x: x[1] != self.database.UNKNOWN_EMOTION, self.database.get_emotions()))
        self.load_data(self.table_emotions, headers, data)

    @staticmethod
    def load_data(table, headers, data):
        table.setColumnCount(len(headers))
        table.setRowCount(0)
        table.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(data):
            table.setRowCount(table.rowCount() + 1)
            for j, el in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(el)))


class Image:
    def __init__(self, path, image):
        self.path = path
        self.image = image


class RecognitionImage(Image):
    def __init__(self, path, image, content):
        super().__init__(path, image)
        self.content = content

    @staticmethod
    def format_content(content):
        fstr = '{} с вероятностью {}%'
        res = '\n'.join(fstr.format(el[0][0], int(el[0][1] * 100)) + ', ' + fstr.format(el[1][0], int(el[1][1] * 100))
                        for el in content)
        return res


class LearningImage(Image):
    def __init__(self, path, image, name, emotion, valid):
        super().__init__(path, image)
        self.name = name
        self.emotion = emotion
        self.valid = valid


def cv_image_to_qt(image, w, h):
    converted_image = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                                   QtGui.QImage.Format.Format_RGB888).rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(converted_image)
    return pixmap.scaled(w, h, Qt.KeepAspectRatio)


class ImageViewer:
    def __init__(self, lbl_name, lbl_image):
        self.lbl_name = lbl_name
        self.lbl_image = lbl_image
        self.images = []
        self.current = 0

    def set_image(self):
        self.lbl_name.setText(self.get_current_image().path)
        image, w, h = self.get_current_image().image, self.lbl_image.width() - 1, self.lbl_image.height() - 1
        self.lbl_image.setPixmap(cv_image_to_qt(image, w, h))

    def get_current_image(self):
        return self.images[self.current]

    def next_image(self):
        if self.current < len(self.images) - 1:
            self.current += 1

    def pref_image(self):
        if self.current > 0:
            self.current -= 1

    def delete_image(self):
        del self.images[self.current]
        if self.current > 0:
            self.current -= 1

    def add_images(self, images):
        self.images = images + self.images


class StreamManager(QThread):
    update_frame = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.stream = None
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.stream.read()
            self.update_frame.emit(frame)
            if frame is None:
                break

    def stop(self):
        self.running = False


def exception_hook(*args):
    sys.__excepthook__(*args)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.excepthook = exception_hook
    sys.exit(app.exec())
