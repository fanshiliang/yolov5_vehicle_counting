import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt


class MyWidget(QWidget):
	def __init__(self):
		super().__init__()
		
		# 创建布局
		layout = QHBoxLayout(self)
		
		# 创建右侧视频播放器
		self.video_player = QLabel(self)
		layout.addWidget(self.video_player, 1)
		
		# 创建左侧按钮和统计信息区域
		left_layout = QVBoxLayout()
		layout.addLayout(left_layout, 1)
		
		# 创建10个按钮并将它们添加到左侧布局中
		for i in range(10):
			button = QPushButton("Button {}".format(i + 1), self)
			button.clicked.connect(lambda _, idx=i: self.draw_line(idx))
			left_layout.addWidget(button)
		
		# 创建统计信息区域
		stats_layout = QHBoxLayout()
		left_layout.addLayout(stats_layout)
		stats_layout.addWidget(QLabel("小汽车："))
		stats_layout.addWidget(QLabel("大客车："))
		stats_layout.addWidget(QLabel("自行车："))
		stats_layout.addWidget(QLabel("行人："))
		
		# 加载YOLO模型和类别列表
		self.net = cv2.dnn.readNet("yolo_weights/yolov3.weights", "yolo_cfg/yolov3.cfg")
		with open("yolo_cfg/coco.names", "r") as f:
			self.classes = [line.strip() for line in f.readlines()]
		
		# 初始化线条坐标列表
		self.lines = []
		
		# 初始化视频播放器
		self.cap = cv2.VideoCapture("test.mp4")
		self.timer = self.startTimer(1000 // self.cap.get(cv2.CAP_PROP_FPS))
	
	def timerEvent(self, event):
		# 从视频中读取帧并进行YOLO检测
		ret, frame = self.cap.read()
		if not ret:
			self.killTimer(self.timer)
			return
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		self.net.setInput(blob)
		outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
		boxes, confidences, class_ids = self.postprocess(frame, outs)
		
		# 在帧上绘制检测结果
		self.draw_boxes(frame, boxes, confidences, class_ids)
		
		# 将帧转换为Qt图像并在右侧视频播放器上显示
		rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytesPerLine = ch * w
		qt_image = QImage(rgb_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
		self.video_player.setPixmap(QPixmap.fromImage(qt_image))
		
		width = frame.shape[1]
		height = frame.shape[0]
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)
					x = center_x - w // 2
					y = center_y - h // 2
					class_ids.append(class_id)
					confidences.append(float(confidence))
					boxes.append([x, y, w, h])
		indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		boxes = [boxes[i[0]] for i in indices]
		class_ids = [class_ids[i[0]] for i in indices]
		confidences = [confidences[i[0]] for i in indices]
		return boxes, confidences, class_ids
	
	def draw_boxes(self, frame, boxes, confidences, class_ids):
		# 在帧上绘制检测结果
		colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
		for i, box in enumerate(boxes):
			x, y, w, h = box
			color = colors[class_ids[i] % len(colors)]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			label = "{}: {:.2f}".format(self.classes[class_ids[i]], confidences[i])
			cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	
	def draw_line(self, idx):
		# 在视频帧上绘制一条线
		self.lines.append(idx)
		frame_idx = len(self.lines)
		color = QColor(255, 0, 0)
		pen = QPen(color, 3, Qt.SolidLine)
		painter = QPainter(self.video_player.pixmap())
		painter.setPen(pen)
		painter.drawLine(0, 0, self.video_player.width(), self.video_player.height())
	
if __name__ == '__main__':
	app = QApplication(sys.argv)
	widget = MyWidget()
	widget.show()
	sys.exit(app.exec_())

