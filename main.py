
""" P15/1662/2019
	ASAVA MAJANI LESLIE
	CROWD MASK DETECTION

"""

import sys
import random
from PyQt5.QtWidgets import QMainWindow,QLabel,QApplication,QPushButton,QFileDialog,QProgressBar,QLineEdit,QMessageBox,QWidget,QGraphicsOpacityEffect,QComboBox, QFormLayout
from PyQt5.QtGui import QPixmap,QFont,QMovie,QPainter,QBrush,QPen
from PyQt5.QtCore import Qt, QThread,pyqtSignal,QByteArray, pyqtSlot, QTimer
from PyQt5 import QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import argparse
import os

import pickle
import pyqtgraph as pg
import numpy as np
#import keras
import cv2
import time

#X = 400
#Y = 150
X = 100
Y = 20
WIDTH = 1200
HEIGHT = 800

class ImageThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray)
	done = pyqtSignal()
	def run(self):
		# construct the argument parser and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-i", "--image",
			help="path to input image")
		ap.add_argument("-f", "--face", type=str,
			default="face_detector",
			help="path to face detector model directory")
		ap.add_argument("-m", "--model", type=str,
			default="mask_detector.model",
			help="path to trained face mask detector model")
		ap.add_argument("-c", "--confidence", type=float, default=0.5,
			help="minimum probability to filter weak detections")
		args = vars(ap.parse_args())

		# load our serialized face detector model from disk
		print("[INFO] loading face detector model...")
		prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
		weightsPath = os.path.sep.join([args["face"],
			"res10_300x300_ssd_iter_140000.caffemodel"])
		net = cv2.dnn.readNet(prototxtPath, weightsPath)

		# load the face mask detector model from disk
		print("[INFO] loading face mask detector model...")
		model = load_model(args["model"])

		# load the input image from disk, clone it, and grab the image spatial
		# dimensions
		#image = cv2.imread(args["image"])
		image = cv2.imread(controller.image_path)
		orig = image.copy()
		(h, w) = image.shape[:2]

		# construct a blob from the image
		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
			(104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		print("[INFO] computing face detections...")
		net.setInput(blob)

		controller.with_masks = 0
		controller.without_masks = 0
		controller.total = 0

		detections = net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				print(detections.shape[2])
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = image[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				face = np.expand_dims(face, axis=0)

				# pass the face through the model to determine if the face
				# has a mask or not
				(mask, withoutMask) = model.predict(face)[0]

				# determine the class label and color we'll use to draw
				# the bounding box and text
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

				if label == "Mask":
					controller.with_masks += 1
				else:
					controller.without_masks += 1

				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				# display the label and bounding box rectangle on the output
				# frame
				cv2.putText(image, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

		"""# show the output image
		cv2.imshow("Output", image)
		print(type(image))
		cv2.waitKey(0)"""
		self.change_pixmap_signal.emit(image)
		controller.model_ran = True

class VideoThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray)
	done = pyqtSignal()
	def run(self):
		def rescale_frame(frame, percent=75):
			width = int(frame.shape[1] * percent/ 100)
			height = int(frame.shape[0] * percent/ 100)
			dim = (width, height)
			return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
		def detect_and_predict_mask(frame, faceNet, maskNet):
			# grab the dimensions of the frame and then construct a blob
			# from it
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
				(104.0, 177.0, 123.0))

			# pass the blob through the network and obtain the face detections
			faceNet.setInput(blob)
			detections = faceNet.forward()

			controller.with_masks = 0
			controller.without_masks = 0
			controller.total = 0

			# initialize our list of faces, their corresponding locations,
			# and the list of predictions from our face mask network
			faces = []
			locs = []
			preds = []

			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the detection
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the confidence is
				# greater than the minimum confidence
				if confidence > args["confidence"]:
					# compute the (x, y)-coordinates of the bounding box for
					# the object
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# ensure the bounding boxes fall within the dimensions of
					# the frame
					(startX, startY) = (max(0, startX), max(0, startY))
					(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

					# extract the face ROI, convert it from BGR to RGB channel
					# ordering, resize it to 224x224, and preprocess it
					face = frame[startY:endY, startX:endX]
					if face.any():
						face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
						face = cv2.resize(face, (224, 224))
						face = img_to_array(face)
						face = preprocess_input(face)

						# add the face and bounding boxes to their respective
						# lists
						faces.append(face)
						locs.append((startX, startY, endX, endY))

			# only make a predictions if at least one face was detected
			if len(faces) > 0:
				# for faster inference we'll make batch predictions on *all*
				# faces at the same time rather than one-by-one predictions
				# in the above `for` loop
				faces = np.array(faces, dtype="float32")
				preds = maskNet.predict(faces, batch_size=32)

			# return a 2-tuple of the face locations and their corresponding
			# locations
			return (locs, preds)

		# construct the argument parser and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-f", "--face", type=str,
			default="face_detector",
			help="path to face detector model directory")
		ap.add_argument("-m", "--model", type=str,
			default="mask_detector.model",
			help="path to trained face mask detector model")
		ap.add_argument("-c", "--confidence", type=float, default=0.5,
			help="minimum probability to filter weak detections")
		args = vars(ap.parse_args())

		# load our serialized face detector model from disk
		print("[INFO] loading face detector model...")
		prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
		weightsPath = os.path.sep.join([args["face"],
			"res10_300x300_ssd_iter_140000.caffemodel"])
		faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

		# load the face mask detector model from disk
		print("[INFO] loading face mask detector model...")
		maskNet = load_model(args["model"])

		# initialize the video stream and allow the camera sensor to warm up
		print("[INFO] starting video stream...")
		vs = cv2.VideoCapture(controller.video_path)
		time.sleep(2.0)

		# loop over the frames from the video stream
		while vs.isOpened():
			try:
				# grab the frame from the threaded video stream and resize it
				# to have a maximum width of 400 pixels
				ret, frame = vs.read()
				#frame = cv2.resize(frame, (400,400))

				# detect faces in the frame and determine if they are wearing a
				# face mask or not
				(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

				# loop over the detected face locations and their corresponding
				# locations
				for (box, pred) in zip(locs, preds):
					# unpack the bounding box and predictions
					(startX, startY, endX, endY) = box
					(mask, withoutMask) = pred

					# determine the class label and color we'll use to draw
					# the bounding box and text
					label = "Mask" if mask > withoutMask else "No Mask"
					color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
						
					# include the probability in the label
					label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

					# display the label and bounding box rectangle on the output
					# frame
					if controller.video_detect:
						cv2.putText(frame, label, (startX, startY - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
						cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			
				self.change_pixmap_signal.emit(frame)
			except:
				vs = cv2.VideoCapture(controller.video_path)

class WebcamThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray)
	done = pyqtSignal()
	def run(self):
		def detect_and_predict_mask(frame, faceNet, maskNet):
			# grab the dimensions of the frame and then construct a blob
			# from it
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
				(104.0, 177.0, 123.0))

			# pass the blob through the network and obtain the face detections
			faceNet.setInput(blob)
			detections = faceNet.forward()


			# initialize our list of faces, their corresponding locations,
			# and the list of predictions from our face mask network
			faces = []
			locs = []
			preds = []

			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the detection
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the confidence is
				# greater than the minimum confidence
				if confidence > args["confidence"]:
					# compute the (x, y)-coordinates of the bounding box for
					# the object
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# ensure the bounding boxes fall within the dimensions of
					# the frame
					(startX, startY) = (max(0, startX), max(0, startY))
					(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

					# extract the face ROI, convert it from BGR to RGB channel
					# ordering, resize it to 224x224, and preprocess it
					face = frame[startY:endY, startX:endX]
					if face.any():
						face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
						face = cv2.resize(face, (224, 224))
						face = img_to_array(face)
						face = preprocess_input(face)

						# add the face and bounding boxes to their respective
						# lists
						faces.append(face)
						locs.append((startX, startY, endX, endY))

			# only make a predictions if at least one face was detected
			if len(faces) > 0:
				# for faster inference we'll make batch predictions on *all*
				# faces at the same time rather than one-by-one predictions
				# in the above `for` loop
				faces = np.array(faces, dtype="float32")
				preds = maskNet.predict(faces, batch_size=32)

			# return a 2-tuple of the face locations and their corresponding
			# locations
			return (locs, preds)

		# construct the argument parser and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-f", "--face", type=str,
			default="face_detector",
			help="path to face detector model directory")
		ap.add_argument("-m", "--model", type=str,
			default="mask_detector.model",
			help="path to trained face mask detector model")
		ap.add_argument("-c", "--confidence", type=float, default=0.5,
			help="minimum probability to filter weak detections")
		args = vars(ap.parse_args())

		# load our serialized face detector model from disk
		print("[INFO] loading face detector model...")
		prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
		weightsPath = os.path.sep.join([args["face"],
			"res10_300x300_ssd_iter_140000.caffemodel"])
		faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

		# load the face mask detector model from disk
		print("[INFO] loading face mask detector model...")
		maskNet = load_model(args["model"])

		# initialize the video stream and allow the camera sensor to warm up
		print("[INFO] starting video stream...")
		vs = VideoStream(0).start()
		time.sleep(2.0)

		# loop over the frames from the video stream
		while True:
			with_masks = 0
			without_masks = 0
			total = 0
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
			frame = vs.read()
			frame = imutils.resize(frame, width=400)

			# detect faces in the frame and determine if they are wearing a
			# face mask or not
			(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

			# loop over the detected face locations and their corresponding
			# locations
			for (box, pred) in zip(locs, preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred

				# determine the class label and color we'll use to draw
				# the bounding box and text
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

				if label == "Mask":
					with_masks += 1
				else:
					without_masks += 1
					
				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				# display the label and bounding box rectangle on the output
				# frame
				if controller.webcam_detect:
					cv2.putText(frame, label, (startX, startY - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
			controller.with_masks = with_masks
			controller.without_masks = without_masks

			self.change_pixmap_signal.emit(frame)
			controller.model_ran = True


class ChartCanvas(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(10.8, 2.8), dpi = 75)
        fig.set_facecolor("#313131")
        super().__init__(fig)
        self.setParent(parent)

class WidgetGroup(QWidget):
	def __init__(self, parent,x,y,title,color):
		super().__init__()
		self.parent = parent
		self.color = color
		self.title = title
		self.width, self.height = 260,320
		self.x,self.y = x,y
		self.createUI()
		self.panel.move(x, y)
		self.layout = QFormLayout(self.panel)

	def createUI(self):
		self.panel = QLabel(self.parent)
		self.panel.resize(self.width,self.height)
		self.panel.setStyleSheet("background-color:white")

		self.title_label = QLabel(self.parent)
		self.title_label.resize(self.width,40)
		self.title_label.setText(self.title)
		self.title_label.setStyleSheet("background-color:%s; padding-left:10px" %self.color)
		self.title_label.move(self.x,self.y-35)

		

class WindowTemplate():
	def __init__(self,parent):
		super().__init__()
		self.parent = parent
		self.createUI()

	def on_home_btn_click(self):
		controller.current_screen = "home"
		self.parent.switch_to_home.emit()
		self.parent.close()

	def on_select_image_file_btn_click(self):
		controller.current_screen = "image"
		self.parent.switch_to_image.emit()
		self.parent.close()

	def on_select_video_file_btn_click(self):
		controller.current_screen = "video"
		self.parent.switch_to_video.emit()
		self.parent.close()

	def on_start_webcam_btn_click(self):
		controller.current_screen = "webcam"
		self.parent.switch_to_webcam.emit()
		self.parent.close()

	def on_analyze_data_btn_click(self):
		controller.current_screen = "analysis"
		self.parent.switch_to_analysis.emit()
		self.parent.close()


	def update_active_screen(self):
		self.format_side_bar()
		button = self.home_btn
		if controller.current_screen == "home":
			button = self.home_btn
		elif controller.current_screen == "image":
			button = self.select_image_file_btn
		elif controller.current_screen == "video":
			button = self.select_video_file_btn
		elif controller.current_screen == "webcam":
			button = self.start_webcam_btn
		elif controller.current_screen == "analysis":
			button = self.analyze_data_btn
		elif controller.current_screen == "desired":
			button = self.desired_btn
		button.setStyleSheet("QPushButton"
                             "{"
                             "background-color : #484E55;color:#FFFEF6;border: solid #4B80F5;border-width: 0px 0px 0px 7px;font-size:8.7pt;"
                             "}")

	def format_side_bar(self):
		start_x = 0
		start_y = 120
		button_width = 227
		button_height = 53
		for button in self.button_list:
			button.resize(button_width,button_height)
			button.move(start_x,start_y)
			button.setStyleSheet("QPushButton::hover"
                             "{"
                             "background-color : #00617F;color:#FFFEF6;border:None"
                             "}"
                             "QPushButton"
                             "{"
                             "font-size:8.7pt;background-color:#252525;color:#FFFEF6; border : None"
                             "}"
                             )
			start_y+=53

	def createUI(self):
		self.parent.setWindowTitle("Speech Emotion Recognition")
		self.footer_opacity_effect = QGraphicsOpacityEffect()
		self.footer_opacity_effect.setOpacity(0.8)
		self.panel_opacity_effect = QGraphicsOpacityEffect()
		self.panel_opacity_effect.setOpacity(0.8)

		self.background = QLabel(self.parent)
		self.background.resize(WIDTH, HEIGHT)
		self.background.setStyleSheet("background-color:#333333")
		
		self.buttons_panel = QLabel(self.parent)
		self.buttons_panel.move(0,80)
		self.buttons_panel.resize(237,HEIGHT-87)
		self.buttons_panel.setStyleSheet("background-color : #252525")
		#self.buttons_panel.setGraphicsEffect(self.panel_opacity_effect)

		self.widget_panel = QLabel(self.parent)
		self.widget_panel.move(227,80)
		self.widget_panel.resize(965,HEIGHT-87)
		self.widget_panel.setStyleSheet("background-color : #484E55")
		#self.widget_panel.setGraphicsEffect(self.panel_opacity_effect)


		self.home_btn = QPushButton(self.parent)
		self.home_btn.setText("Home")
		self.home_btn.clicked.connect(self.on_home_btn_click)

		self.select_image_file_btn = QPushButton(self.parent)
		self.select_image_file_btn.setText("Select Image File")
		self.select_image_file_btn.clicked.connect(self.on_select_image_file_btn_click)

		self.select_video_file_btn = QPushButton(self.parent)
		self.select_video_file_btn.setText("Select Video File")
		self.select_video_file_btn.clicked.connect(self.on_select_video_file_btn_click)

		self.start_webcam_btn = QPushButton(self.parent)
		self.start_webcam_btn.setText("Start Webcam")
		self.start_webcam_btn.clicked.connect(self.on_start_webcam_btn_click)

		self.analyze_data_btn = QPushButton(self.parent)
		self.analyze_data_btn.setText("Analyze Data")
		self.analyze_data_btn.clicked.connect(self.on_analyze_data_btn_click)

		"""self.footer_panel = QLabel(self.parent)
		self.footer_panel.move(0,HEIGHT-50)
		self.footer_panel.resize(WIDTH,50)
		self.footer_panel.setStyleSheet("background-color : rgb(20,20,20)")
		self.footer_panel.setGraphicsEffect(self.footer_opacity_effect)"""
		self.python=QLabel(self.parent)
		self.python_pixmap=QPixmap('Python-Logo-Png.png')
		#python_pixmap = python_pixmap.scaledToHeight(35)
		self.python.setPixmap(self.python_pixmap)
		self.python.resize(230,60)
		self.python.move(640,HEIGHT-55)

		self.button_list = [self.home_btn,self.select_image_file_btn,self.select_video_file_btn,self.start_webcam_btn,self.analyze_data_btn]
		self.update_active_screen()
		


class HomeScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_image = pyqtSignal()
	switch_to_video = pyqtSignal()
	switch_to_webcam = pyqtSignal()
	switch_to_analysis = pyqtSignal()	
	switch_to_desired = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
		self.template = WindowTemplate(self)

		self.logo_label = QLabel(self)
		self.logo_label.setPixmap(QtGui.QPixmap("covid-logo.png").scaledToHeight(240))
		self.logo_label.resize(500,240)
		self.logo_label.move(420,130)
		self.logo_label.setAlignment(Qt.AlignCenter)

		self.title_label = QLabel(self)
		self.title_label.setText("Crowd Face Mask Detection")
		self.title_label.setStyleSheet("color:white; font-size:28pt; font-weight: bold")
		self.title_label.move(400,400)

		self.slogan_label = QLabel(self)
		self.slogan_label.setText(" ~ No one's safe until we're all safe ~ ")
		self.slogan_label.setStyleSheet("color:white; font-size:15pt")
		self.slogan_label.move(490,480)
		

class ImageFileScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_image = pyqtSignal()
	switch_to_video = pyqtSignal()
	switch_to_webcam = pyqtSignal()
	switch_to_analysis = pyqtSignal()	
	switch_to_desired = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.media_label.setPixmap(qt_img)
    
	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		p = convert_to_Qt_format.scaled(800, 470, Qt.KeepAspectRatio)
		return QPixmap.fromImage(p)

	def run_model(self):
		self.thread = ImageThread()
		# connect its signal to the update_image slot
		self.thread.change_pixmap_signal.connect(self.update_image)
		# start the thread
		self.thread.start()

	def open_image_file_dialog(self):
		#self.emotion_label.setText("")
		options = QFileDialog.Options()
		fileName, _ = QFileDialog.getOpenFileName(self,"Select Image File", "","Image Files (*.*);", options=options)
		if fileName:
			controller.image_path = fileName

			self.file_path_entry.setText(fileName)
			img = cv2.imread(fileName)
			self.dimensions = img.shape

			self.input_pixmap=QPixmap(fileName)
			if self.dimensions[0] >= self.dimensions[1]:
				self.input_pixmap = self.input_pixmap.scaledToHeight(460)
			else:
				self.input_pixmap = self.input_pixmap.scaledToWidth(790)
			self.media_label.setPixmap(self.input_pixmap)


	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
		self.template = WindowTemplate(self)

		self.title_label = QLabel(self)
		self.title_label.setText("Choose image file")
		self.title_label.setStyleSheet("color:white; font-size:13pt")
		self.title_label.move(280,120)

		self.file_path_entry = QLineEdit(self)
		self.file_path_entry.resize(560,40)
		self.file_path_entry.move(280,170)
		self.file_path_entry.setStyleSheet("border-radius:10px; font-size: 10pt; padding-left: 10px")

		self.file_path_btn = QPushButton(self)
		self.file_path_btn.resize(120,35)
		self.file_path_btn.move(900,172)
		self.file_path_btn.setText("Browse")
		self.file_path_btn.setStyleSheet("border-radius:10px; background-color:darkgray")
		self.file_path_btn.clicked.connect(self.open_image_file_dialog)

		self.media_label = QLabel(self)
		self.media_label.resize(800,470)
		self.media_label.move(280,250)
		self.media_label.setStyleSheet("background-color: black; border-radius: 5px")
		self.media_label.setAlignment(Qt.AlignCenter)

		self.run_model_button = QPushButton(self)
		self.run_model_button.setText("Run Model")
		self.run_model_button.resize(130,40)
		self.run_model_button.move(650,737)
		self.run_model_button.setStyleSheet("background-color: green; color: white; border-radius: 15px; font-size: 10pt")
		self.run_model_button.clicked.connect(self.run_model)

class VideoFileScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_image = pyqtSignal()
	switch_to_video = pyqtSignal()
	switch_to_webcam = pyqtSignal()
	switch_to_analysis = pyqtSignal()	

	def __init__(self):
		super().__init__()
		self.createUI()

		

	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.media_label.setPixmap(qt_img)

	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		p = convert_to_Qt_format.scaled(830, 510, Qt.KeepAspectRatio)
		return QPixmap.fromImage(p)

	def run_model(self):
		if not controller.video_detect:
			controller.video_detect = True
			self.run_model_button.setText("Stop Model")
		else:
			controller.video_detect = False
			self.run_model_button.setText("Run Model")

	def open_video_file_dialog(self):
		#self.emotion_label.setText("")
		options = QFileDialog.Options()
		fileName, _ = QFileDialog.getOpenFileName(self,"Select Video File", "","Video Files (*.mp4);", options=options)
		if fileName:
			controller.video_path = fileName

			self.file_path_entry.setText(fileName)
			self.thread = VideoThread()
			# connect its signal to the update_image slot
			self.thread.change_pixmap_signal.connect(self.update_image)
			# start the thread
			self.thread.start()
		

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
		self.template = WindowTemplate(self)

		self.title_label = QLabel(self)
		self.title_label.setText("Choose video file")
		self.title_label.setStyleSheet("color:white; font-size:13pt")
		self.title_label.move(280,120)

		self.file_path_entry = QLineEdit(self)
		self.file_path_entry.resize(560,40)
		self.file_path_entry.move(280,170)
		self.file_path_entry.setStyleSheet("border-radius:10px; font-size: 10pt; padding-left: 10px")

		self.file_path_btn = QPushButton(self)
		self.file_path_btn.resize(120,35)
		self.file_path_btn.move(900,172)
		self.file_path_btn.setText("Browse")
		self.file_path_btn.setStyleSheet("border-radius:10px; background-color:darkgray")
		self.file_path_btn.clicked.connect(self.open_video_file_dialog)

		self.media_label = QLabel(self)
		self.media_label.resize(800,470)
		self.media_label.move(280,250)
		self.media_label.setStyleSheet("background-color: black; border-radius: 5px")
		self.media_label.setAlignment(Qt.AlignCenter)

		self.run_model_button = QPushButton(self)
		self.run_model_button.setText("Run Model")
		self.run_model_button.resize(130,40)
		self.run_model_button.move(650,737)
		self.run_model_button.setStyleSheet("background-color: green; color: white; border-radius: 15px; font-size: 10pt")
		self.run_model_button.clicked.connect(self.run_model)

class WebcamScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_image = pyqtSignal()
	switch_to_video = pyqtSignal()
	switch_to_webcam = pyqtSignal()
	switch_to_analysis = pyqtSignal()	

	def __init__(self):
		super().__init__()
		self.createUI()
		self.thread = WebcamThread()
		# connect its signal to the update_image slot
		self.thread.change_pixmap_signal.connect(self.update_image)
		# start the thread
		self.thread.start()

	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.media_label.setPixmap(qt_img)

	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		p = convert_to_Qt_format.scaled(830, 510, Qt.KeepAspectRatio)
		return QPixmap.fromImage(p)

	def run_model(self):
		if not controller.webcam_detect:
			controller.webcam_detect = True
			self.run_model_button.setText("Stop Model")
		else:
			controller.webcam_detect = False
			self.run_model_button.setText("Run Model")
		

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
		self.template = WindowTemplate(self)

		self.title_label = QLabel(self)
		self.title_label.setText("Live video stream")
		self.title_label.setStyleSheet("color:white; font-size:13pt")
		self.title_label.move(290,120)

		self.media_label = QLabel(self)
		self.media_label.resize(830,510)
		self.media_label.move(280,180)
		self.media_label.setStyleSheet("background-color: black; border-radius: 5px")
		self.media_label.setAlignment(Qt.AlignCenter)

		self.run_model_button = QPushButton(self)
		self.run_model_button.setText("Run Model")
		self.run_model_button.resize(130,40)
		self.run_model_button.move(650,722)
		self.run_model_button.setStyleSheet("background-color: green; color: white; border-radius: 15px; font-size: 10pt")
		self.run_model_button.clicked.connect(self.run_model)



class DataAnalysisScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_image = pyqtSignal()
	switch_to_video = pyqtSignal()
	switch_to_webcam = pyqtSignal()
	switch_to_analysis = pyqtSignal()	
	switch_to_desired = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()
		self.plot_chart()
		self.display_statistics()

	def plot_chart(self):
		if controller.model_ran:
			self.pie_chart.ax.clear()
			x = [controller.with_masks, controller.without_masks]
			y = ["With Masks", "Without Masks"]

			x_new = []
			y_new = []
			index = 0
			for i in x:
				if i:
					x_new.append(i)
					y_new.append(y[index])

				index += 1

			_, patches, autotexts = self.pie_chart.ax.pie(x_new,labels = y_new, colors = ["g","r"], autopct='%1.0f%%')
			try:
				for patch, autotext in patches, autotexts:
					autotext.set_color('white')
					patch.set_color('white')
			except:
				pass
			self.pie_chart.ax.legend()
			print(x,y)

	def display_statistics(self):
		#max_value = max(controller.file_list)
		#max_index = controller.file_list.index(max_value)

		message = "People detected : %s\n"%(controller.without_masks + controller.with_masks)
		message += "With Masks     : %s\n"%controller.with_masks
		message += "Without Masks  : %s\n"%controller.without_masks

		self.statistics_label.setText(message)

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
		self.template = WindowTemplate(self)

		self.title_label = QLabel(self)
		self.title_label.setText("Data Analysis")
		self.title_label.setStyleSheet("color:white; font-size:13pt")
		self.title_label.move(290,120)

		self.timer=QTimer()
		self.timer.timeout.connect(self.plot_chart)
		self.timer.start(1000)

		# Instantiate the pie chart canvas
		self.pie_chart = ChartCanvas(self)

		# Position the chart
		self.pie_chart.move(290,190)
		self.pie_chart.resize(620,500)

		self.statistics_label = QLabel(self)
		self.statistics_label.resize(230,500)
		self.statistics_label.move(930,190)
		self.statistics_label.setStyleSheet("background-color: #313131; border-radius: 10px; color: white; padding-left: 10px; font-size: 12pt; margin-top: 20px")

class Controller():
	def __init__(self):
		self.current_screen = ""
		self.image_path = ""
		self.video_path = ""
		self.webcam_detect = False
		self.video_detect = True
		self.model_ran = False

		self.with_masks = 0
		self.without_masks = 0
		self.total = 0

	def show_home(self):
		self.home_screen = HomeScreen()
		self.home_screen.switch_to_home.connect(self.show_home)
		self.home_screen.switch_to_image.connect(self.show_image)
		self.home_screen.switch_to_video.connect(self.show_video)
		self.home_screen.switch_to_webcam.connect(self.show_webcam)
		self.home_screen.switch_to_analysis.connect(self.show_analysis)	
		self.home_screen.show()

	def show_image(self):
		self.image_screen = ImageFileScreen()
		self.image_screen.switch_to_home.connect(self.show_home)
		self.image_screen.switch_to_image.connect(self.show_image)
		self.image_screen.switch_to_video.connect(self.show_video)
		self.image_screen.switch_to_webcam.connect(self.show_webcam)
		self.image_screen.switch_to_analysis.connect(self.show_analysis)	
		self.image_screen.show()

	def show_video(self):
		self.video_screen = VideoFileScreen()
		self.video_screen.switch_to_home.connect(self.show_home)
		self.video_screen.switch_to_image.connect(self.show_image)
		self.video_screen.switch_to_video.connect(self.show_video)
		self.video_screen.switch_to_webcam.connect(self.show_webcam)
		self.video_screen.switch_to_analysis.connect(self.show_analysis)	
		self.video_screen.show()

	def show_webcam(self):
		self.webcam_screen = WebcamScreen()
		self.webcam_screen.switch_to_home.connect(self.show_home)
		self.webcam_screen.switch_to_image.connect(self.show_image)
		self.webcam_screen.switch_to_video.connect(self.show_video)
		self.webcam_screen.switch_to_webcam.connect(self.show_webcam)
		self.webcam_screen.switch_to_analysis.connect(self.show_analysis)	
		self.webcam_screen.show()

	def show_analysis(self):
		self.analysis_screen = DataAnalysisScreen()
		self.analysis_screen.switch_to_home.connect(self.show_home)
		self.analysis_screen.switch_to_image.connect(self.show_image)
		self.analysis_screen.switch_to_video.connect(self.show_video)
		self.analysis_screen.switch_to_webcam.connect(self.show_webcam)
		self.analysis_screen.switch_to_analysis.connect(self.show_analysis)	
		self.analysis_screen.show()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	app.setStyle('Fusion')
	controller = Controller()
	controller.show_home()
	sys.exit(app.exec_())