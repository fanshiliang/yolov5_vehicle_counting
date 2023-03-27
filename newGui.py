import sys
import threading

import cv2
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, \
	QGridLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

from detector import Detector
import tracker


class VideoLoaderThread(QThread):
	video_loaded = pyqtSignal(object)
	
	def __init__(self, parent=None, detector = None):
		super().__init__(parent)
		self.detector = detector
	
	def run(self):
		# open file dialog to select video file
		file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "Video Files (*.mp4 *.avi)")
		if not file_path:
			return
		
		# set media content to selected video file
		media_content = QMediaContent(QUrl.fromLocalFile(file_path))
		
		# set media content to video player and play
		video_player = QMediaPlayer()
		video_player.setMedia(media_content)
		
		# create video capture
		cap = cv2.VideoCapture(file_path)
		
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			
			# 缩小尺寸，1920x1080->960x540
			im = cv2.resize(im, (960, 540))
			list_bboxs = []
			bboxes = self.detector.detect(im)
			
			# 如果画面中 有bbox
			if len(bboxes) > 0:
				list_bboxs = tracker.update(bboxes, im)
				
				# 画框
				# 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
				output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=1)
				pass
			else:
				# 如果画面中 没有bbox
				output_image_frame = im
			pass
			
			# emit signal with frame and detected objects
			self.video_loaded.emit((output_image_frame, bboxes))
		
		cap.release()
		video_player.stop()
		
class MyGUI(QWidget):
	def __init__(self):
		super().__init__()
		self.initUI()
		self.grid_layout = None
		self.detector = None
		self.tracker = None
	
	def initUI(self):
		# create the layout
		self.grid_layout = QGridLayout(self)
		self.detector = Detector()
		self.tracker = tracker.Tracker()
		
		# create the buttons and add to button layout
		button_labels = ["line1", "line2", "line3", "line4", "line5", "line6", "line7", "line8", "line9", "line10"]
		
		button_shortcuts = ["Ctrl+O", "Space", "P", "S", "Left", "Right", "Comma", "Period", "Up", "Down"]
		button_icons = ["open.png", "play.png", "pause.png", "stop.png", "rewind.png", "fast-forward.png",
		                "previous-frame.png", "next-frame.png", "increase-speed.png", "decrease-speed.png"]
		# Create a grid layout for the buttons and add them to the layout
		button_layout = QGridLayout()
		for i, label in enumerate(button_labels):
			# Create the button and set its properties
			button = QPushButton(label)
			button.setToolTip(f"Click to {label}")
			button.setShortcut(button_shortcuts[i])
			button.setIcon(QIcon(f"icons/{button_icons[i]}"))
			
			# Create a label to hold the text
			text_label = QLabel(f"{label}")
			text_label.setAlignment(Qt.AlignRight)
			
			# Add the button and label to the layout
			button_layout.addWidget(text_label, i, 0)
			# button_layout.addWidget(button, i, 1)
		
		# Add the button layout to the main layout
		self.grid_layout.addLayout(button_layout, 0, 0)
		
		# create the video display and add to video layout
		self.video_player = QMediaPlayer()
		self.video_widget = QVideoWidget()
		self.video_widget.setFixedSize(600, 400)
		self.video_player.setVideoOutput(self.video_widget)
		self.grid_layout.addWidget(self.video_widget, 0, 2, 10, 1)
		
		# create the statistic display and add to stat layout
		stat_display = QLabel("Statistic Display")
		self.grid_layout.addWidget(stat_display, 10, 0, 1, 3)
		
		# set the window properties
		self.setGeometry(100, 100, 800, 600)
		self.setWindowTitle('My GUI')
		
		# connect the video_loaded signal to update the video display
		self.video_loader_thread = VideoLoaderThread()
		self.video_loader_thread.video_loaded.connect(self.update_video_display(self.detector))
		self.video_loader_thread.start()
		
		self.setLayout(self.grid_layout)
	
	def update_video_display(self, video_frame):
		frame, detected_objects = video_frame
		
		# convert the frame to QImage and display it on the video widget
		height, width, channel = frame.shape
		bytes_per_line = 3 * width
		q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
		self.video_widget.setPixmap(QPixmap.fromImage(q_img))
		
		# create layout for detected objects and add to main layout
		self.detected_objects_label = QLabel("Detected Objects:")
		self.grid_layout.addWidget(self.detected_objects_label, 2, 0)
		self.detected_objects_text = QLabel("")
		self.grid_layout.addWidget(self.detected_objects_text, 2, 2)
		
		# set the main layout and window properties
		self.setLayout(self.grid_layout)
		self.setGeometry(200, 200, 900, 500)
		self.setWindowTitle("My Video Player")
	
	def load_video_file(self):
		# create video capture
		file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "Video Files (*.mp4 *.avi)")
		if not file_path:
			return
		
		# set media content to selected video file
		media_content = QMediaContent(QUrl.fromLocalFile(file_path))
		self.video_player.setMedia(media_content)
		
		# enable video controls and play button
		self.play_button.setEnabled(True)
		self.pause_button.setEnabled(True)
		self.stop_button.setEnabled(True)
	
	def load_video_file_in_thread(self):
		self.video_loader_thread = VideoLoaderThread()
		self.video_loader_thread.video_loaded.connect(self.handle_loaded_video)
		self.video_loader_thread.start()
	
	def handle_loaded_video(self, frame_and_detected_objects):
		frame, detected_objects = frame_and_detected_objects
		self.display_frame(frame)
		self.display_detected_objects(detected_objects)
	
	def display_frame(self, frame):
		# convert frame from BGR to RGB
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# convert frame to QImage
		qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
		# set QImage to video widget
		self.video_widget.setPixmap(QPixmap.fromImage(qimage))
	
	def display_detected_objects(self, detected_objects):
		detected_objects_str = "\n".join(detected_objects)
		self.detected_objects_text.setText(detected_objects_str)
	
	def play_video(self):
		self.video_player.play()
	
	def pause_video(self):
		self.video_player.pause()
	
	def stop_video(self):
		self.video_player.stop()
	
	def set_video_position(self, position):
		self.video_player.setPosition(position)
			

if __name__ == '__main__':
	app = QApplication(sys.argv)
	gui = MyGUI()
	gui.show()
	sys.exit(app.exec_())
