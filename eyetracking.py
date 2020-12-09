from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import dlib
import cv2
import VideoCapture
import math
import time


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def pt_distance(coord1, coord2):
	(x1, y1) = coord1
	(x2, y2) = coord2
	return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

class EyeTracker:
	def __init__(self, vid, ear_thresh=.18, blink_consec=2):
		self.vid = vid
		self.ear_thresh = ear_thresh
		self.blink_consec = blink_consec
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		(self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		self.r_left_corner_ind = 36
		self.r_right_corner_ind = 39
		self.l_left_corner_ind = 42
		self.l_right_corner_ind = 45
		self.left_counter = 0
		self.right_counter = 0
		self.left_blinked = False
		self.right_blinked = False
		self.img = []
		self.face_points = []
		self.data = []
		self.good_data = False

	def get_image(self, img):
		return self.img

	def get_faces(self):
		if(self.img == []):
			print("No image")
			return []
		return self.detector(self.img, 0)

	def gen_face_points(self):
		faces = self.get_faces()
		if(faces == []):
			print("No faces")
			return []
		for face in faces:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			self.face_points = self.predictor(self.img, face)
			self.face_points = face_utils.shape_to_np(self.face_points)

	def get_face_points(self):
		if(self.face_points == []):
			print("No face points")
		return self.face_points

	def get_data(self):
		if(not self.good_data):
			return []
		return self.data

	def eye_aspect_ratio(self,eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates

		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])
		# A = pt_distance(eye[1], eye[5])
		# B = pt_distance(eye[2], eye[4])
		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])
		# C = pt_distance(eye[0], eye[3])
		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)
		# return the eye aspect ratio
		return ear

	def nothing(self,x):
		pass

	def mainloop(self):
		ret, frame = self.vid.get_frame()
		self.img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		# cv2.createTrackbar('threshold', 'image', 50, 255, self.nothing)
		if(ret):
			self.gen_face_points()
			if(self.face_points == []):
				return
			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			leftEye = self.face_points[self.lStart:self.lEnd]
			rightEye = self.face_points[self.rStart:self.rEnd]

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			leftEAR = self.eye_aspect_ratio(leftEye)
			rightEAR = self.eye_aspect_ratio(rightEye)

			#check for blinking
			#left blibk
			if leftEAR < self.ear_thresh and rightEAR >= self.ear_thresh:
				self.left_counter += 1
				return
			else:
				if self.left_counter >= self.blink_consec:
					self.left_blinked = True
				# reset the eye frame counter
				self.left_counter = 0

			#right blink
			if rightEAR < self.ear_thresh and leftEAR >= self.ear_thresh:
				self.right_counter += 1
				return
			else:
				if self.right_counter >= self.blink_consec:
					self.right_blinked = True
				# reset the eye frame counter
				self.right_counter = 0

			black_frame = np.zeros_like(self.img).astype(np.uint8)
			cv2.fillPoly(black_frame , [leftEyeHull], (255, 255, 255))
			cv2.fillPoly(black_frame , [rightEyeHull], (255, 255, 255))
			mask = black_frame == 255
			w_mask = np.array(255 * (mask == 0)).astype(np.uint8)
			targetROI = (self.img * mask) + w_mask
			#
			# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			# cv2.putText(targetROI, "Blinks: {}".format(TOTAL), (10, 30),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(targetROI, "EAR: {:.2f}".format(leftEAR), (300, 30),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# show the frame
			blur = cv2.GaussianBlur(targetROI, (5,5), 0)
			# thresh_val = r = cv2.getTrackbarPos('threshold', 'thresh')
			_, threshold = cv2.threshold(blur, 30,255, cv2.THRESH_BINARY_INV)
			contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			# if len(contours) != 2:
			# 	print("Too many contours")
			# 	continue

			centers = []
			eyes = []
			areas = []
			for cnt in contours:
				area = cv2.contourArea(cnt)
				areas.append(area)
				if(area < 20):
					continue
				cv2.drawContours(frame, [cnt], -1, (0,0,255), 1)
				M = cv2.moments(cnt)
				if M["m00"] == 0:
					break
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				centers.append([cX, cY])
				eyes.append(cnt)

			if len(centers) != 2:
				self.good_data = False
				print("Bad center")
				return
			#first center is right eye (leftmost eye in image)
			left_iris = eyes[1]
			right_iris = eyes[0]
			if centers[0][0] > centers[1][0]:
				tmp = centers[0]
				centers[0] = centers[1]
				centers[1] = tmp
				left_iris = eyes[0]
				right_iris = eyes[1]

			rmin_left_dist = 0
			rmin_right_dist = 0
			first = True
			for pt in right_iris:
				(x, y) = pt[0]
				(xl, yl) = self.face_points[self.r_left_corner_ind]
				(xr, yr) = self.face_points[self.r_right_corner_ind]
				ldist = distance(x, y, xl, yl)
				rdist = distance(x, y, xr, yr)
				if(first):
					rmin_left_dist = ldist
					rmin_right_dist = rdist
					first = False
					continue
				if ldist < rmin_left_dist:
					rmin_left_dist = ldist
				if rdist < rmin_right_dist:
					rmin_right_dist = rdist


			lmin_left_dist = 0
			lmin_right_dist = 0
			first = True
			for pt in left_iris:
				(x, y) = pt[0]
				(xl, yl) = self.face_points[self.l_left_corner_ind]
				(xr, yr) = self.face_points[self.l_right_corner_ind]
				ldist = distance(x, y, xl, yl)
				rdist = distance(x, y, xr, yr)
				if(first):
					lmin_left_dist = ldist
					lmin_right_dist = rdist
					first = False
					continue
				if ldist < lmin_left_dist:
					lmin_left_dist = ldist
				if rdist < lmin_right_dist:
					lmin_right_dist = rdist

			self.dists = [rmin_left_dist, rmin_right_dist, lmin_right_dist, lmin_left_dist]
			self.centers = centers

			self.data = [self.centers, self.face_points, self.dists]
			self.good_data = True
			cv2.imshow("image", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break




# # #
vs = VideoCapture.MyVideoCapture()
et = EyeTracker(vid = vs)
while True:
	start_time = time.time()
	et.mainloop()
	# print("--- %s seconds ---" % (time.time() - start_time))
