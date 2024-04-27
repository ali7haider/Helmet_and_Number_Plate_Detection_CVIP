import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import time
from PyQt5.QtCore import Qt
from main_ui import Ui_MainWindow  # Import the generated class
from math import floor, ceil
from datetime import timedelta


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up the user interface from the generated class
        self.setupUi(self)

        # Set flags to remove the default title bar
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Connect the maximizeRestoreAppBtn button to the maximize_window method
        self.maximizeRestoreAppBtn.clicked.connect(self.maximize_window)

        # Connect the closeAppBtn button to the close method
        self.closeAppBtn.clicked.connect(self.close_event)

        # Connect the minimizeAppBtn button to the showMinimized method
        self.minimizeAppBtn.clicked.connect(self.showMinimized)
        self.btnBrowse.clicked.connect(self.upload_video)
        self.cap = None

        self.model = YOLO('best.pt')

        self.class_list = []
        with open("classes.txt", "r") as my_file:
            data = my_file.read()
            self.class_list = data.split("\n")

    def maximize_window(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
    def upload_video(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, 'Open video', '', 'Video Files (*.mp4)')
        if file_path:
            self.video_file_path = file_path
            self.txtBrowsePath.setText(file_path)
            self.process_video(file_path)

    def process_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Get total duration of the video
        if self.cap is not None and self.cap.isOpened():
            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_seconds = total_frames / fps
            total_time = str(timedelta(seconds=total_seconds))
            self.lblTotalTime.setText(total_time)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (700, 400))
            self.original_frame = frame.copy()

            results= self.model.predict(frame)
            a=results[0].boxes.data
            self.px=pd.DataFrame(a).astype("float")
            
            rider_list = []
            helmet_list = []
            number_list = []
            for index,row in self.px.iterrows():
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[5])
                c=self.class_list[d]

                if c == 'helmet':
                    helmet_list.append(row)
                elif c == 'motorcyclist':
                    rider_list.append(row)
                elif c == 'license_plate':
                    number_list.append(row)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

            for rider in rider_list:
                x1r, y1r, x2r, y2r, _ ,_= rider
                no_helmet = True
                for helmet in helmet_list:
                    x1h, y1h, x2h, y2h, _ ,_= helmet
                    if self.inside_box([x1r, y1r, x2r, y2r], [x1h, y1h, x2h, y2h]):
                        no_helmet = False
                        break

                if no_helmet:
                    for number in number_list:
                        x1n, y1n, x2n, y2n, _ ,_= number
                        if self.inside_box([x1r, y1r, x2r, y2r], [x1n, y1n, x2n, y2n]):
                            plate_region = self.original_frame[int(y1n):int(y2n), int(x1n):int(x2n)]
                            cv2.imwrite(f"license_plates/{time.time()}_plate.jpg", plate_region)

            self.display_frame(frame)
            self.update_progress_bar()  # Update progress bar after processing each frame
            # Update running time
            if self.cap.isOpened():
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                current_seconds = current_frame / fps
                current_time = str(timedelta(seconds=current_seconds)).split('.', 1)[0]  # Remove milliseconds
                self.lblRunningTime.setText(current_time)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


    def close_event(self):
        # Release OpenCV capture and destroy any OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()
        # Close the application
        self.close()
    def inside_box(self, big_box, small_box):
        x1 = small_box[0] - big_box[0]
        y1 = small_box[1] - big_box[1]
        x2 = big_box[2] - small_box[2]
        y2 = big_box[3] - small_box[3]
        return not bool(min([x1, y1, x2, y2, 0]))
    def update_progress_bar(self):
        if self.cap is not None:
            print("Video capture object exists.")
            if self.cap.isOpened():
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                progress_percentage = (current_frame / total_frames) * 100
                
                # Check if the fractional part of progress_percentage is greater than or equal to 0.5
                fractional_part = progress_percentage - floor(progress_percentage)
                if fractional_part >= 0.5:
                    progress_percentage = ceil(progress_percentage)  # Round up
                else:
                    progress_percentage = floor(progress_percentage)  # Round down
                
                self.videoPrograssBar.setValue(int(progress_percentage))
            else:
                print("Error: Video capture is not open.")
        else:
            print("Error: Video capture is not initialized.")


    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
         # Scale the QPixmap to fit the QLabel while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
        self.videoLabel.setPixmap(scaled_pixmap)
        # Update progress bar
        # self.update_progress_bar()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
