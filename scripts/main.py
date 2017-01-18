""" 
====================================================
		Real Time Heart Rate Detector 
====================================================

The Haar cascade dataset used is the Extended Yale Database B Cropped

  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html


Summary:
	This utility overlays the heart rate extracted from input video.
	Order of operations:
		-> Run real time facial tracking and recognition using Haar cascades
		and SVM 
		-> Integrate cropped faces over red channel to produce 1d scalar 
		-> FFT 1d red brightness scalar for sets of 30 frames
		-> Find peak of FFT within range of expected heart rates

To Run:
	* To run it without options
		python main.py

	* Or running with options (By default, scale_multiplier = 4):

		python main.py [scale_multiplier=<full screensize divided by scale_multiplier>]

	* Say you want to run with 1/2 of the full sreen size, specify that scale_multiplier = 4:

		python main.py 4


Usage: 
		press 'q' or 'ESC' to quit the application


		
Adapted from code by Chenxing Ouyang <c2ouyang@ucsd.edu>
Chenxing's code does the face detection python implementation and provided the Haar cascade database

Written by Amos Manneschmidt

"""

import cv2
import os
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import utils as ut
import svm
import sys
import logging
import warnings
from pdb import set_trace as br

print(__doc__)
###############################################################################
# Building SVC from database
integrated_red=np.arange((150))
integrated_blue=integrated_red
integrated_green=integrated_red
#ani.save('test_sub.mp4')
plt.ion()

x = np.linspace(0, 1, 150)
y=integrated_red
# You probably won't need this if you're embedding things in a tkinter plot...

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-')
line2, = ax.plot(x, y, 'b-')
line3, = ax.plot(x, y, 'g-')# Returns a tuple of line objects, thus the comma
# for phase in np.linspace(0, 10*np.pi, 500):
	# line1.set_ydata(np.sin(x + phase))
fig.canvas.draw()

FACE_DIM = (50,50) # h = 50, w = 50
# Load training data from face_profiles/
face_profile_data, face_profile_name_index, face_profile_names  = ut.load_training_data("../face_profiles/")

print "\n", face_profile_name_index.shape[0], " samples from ", len(face_profile_names), " people are loaded"

# Build the classifier
clf, pca = svm.build_SVC(face_profile_data, face_profile_name_index, FACE_DIM)


###############################################################################
# Facial Recognition In Live Tracking
DISPLAY_FACE_DIM = (200, 200) # the displayed video stream screen dimention 
SKIP_FRAME = 1      # the fixed skip frame
frame_skip_rate = 1 # skip SKIP_FRAME frames every other frame
SCALE_FACTOR = 4 # used to resize the captured frame for face detection for faster processing speed
face_cascade = cv2.CascadeClassifier("../classifier/haarcascade_frontalface_default.xml") #create a cascade classifier
sideFace_cascade = cv2.CascadeClassifier('../classifier/haarcascade_profileface.xml')

# Certainly simpler that a full argument parser. Will probably update later
if len(sys.argv) == 2:
	SCALE_FACTOR = float(sys.argv[1])
elif len(sys.argv) >2:
	logging.error("main.py ")
# dictionary mapping used to keep track of head rotation maps
rotation_maps = {
	"left": np.array([-30, 0, 30]),
	"right": np.array([30, 0, -30]),
	"middle": np.array([0, -30, 30]),
}
def smooth(x,window_len=9,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2):-(window_len/2)]


def get_rotation_map(rotation):
	""" Takes in an angle rotation, and returns an optimized rotation map """
	if rotation > 0: return rotation_maps.get("right", None)
	if rotation < 0: return rotation_maps.get("left", None)
	if rotation == 0: return rotation_maps.get("middle", None)

current_rotation_map = get_rotation_map(0) 


webcam = cv2.VideoCapture(0)

ret, frame = webcam.read() # get first frame
frame_scale = (frame.shape[1]/SCALE_FACTOR,frame.shape[0]/SCALE_FACTOR)  # (y, x)

cropped_face = []
num_of_face_saved = 0


while ret:
	key = cv2.waitKey(1)
	# exit on 'q' 'esc' 'Q'
	if key in [27, ord('Q'), ord('q')]: 
		break
	# resize the captured frame for face detection to increase processing speed
	resized_frame = cv2.resize(frame, frame_scale)

	processed_frame = resized_frame
	# Skip a frame if the no face was found last frame
	xo=0
	yo=0
	wo=0
	ho=0
	if frame_skip_rate == 0:
		faceFound = False
		for rotation in current_rotation_map:

			rotated_frame = ndimage.rotate(resized_frame, rotation)
			gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

			# return tuple is empty, ndarray if detected face
			faces = face_cascade.detectMultiScale(
				gray,
				scaleFactor=1.3,
				minNeighbors=3,
				minSize=(25, 25),
				flags=cv2.CASCADE_SCALE_IMAGE
			) 

			# If frontal face detector failed, use profileface detector
			faces = faces if len(faces) else sideFace_cascade.detectMultiScale(                
				gray,
				scaleFactor=1.3,
				minNeighbors=3,
				minSize=(25, 25),
				flags=cv2.CASCADE_SCALE_IMAGE
			)

			# for f in faces:
			# 	x, y, w, h = [ v*SCALE_FACTOR for v in f ] # scale the bounding box back to original frame size
			# 	cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
			# 	cv2.putText(frame, "DumbAss", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

			if len(faces):
				for f in faces:
					# Crop out the face
					x, y, w, h = [ v for v in f ] # scale the bounding box back to original frame size
					cropped_face = rotated_frame[y: y + int(.8*h+.2*ho), x: x + int(.8*w+.2*wo)]   # img[y: y + h, x: x + w]
					cropped_face = cv2.resize(cropped_face, DISPLAY_FACE_DIM, interpolation = cv2.INTER_AREA)
					integrated_red=np.roll(integrated_red,1)
					integrated_blue=np.roll(integrated_blue,1)
					integrated_green=np.roll(integrated_green,1)
					integrated_red[0]=np.median(cropped_face[:,:,2])
					integrated_blue[0]=np.median(cropped_face[:,:,1])
					integrated_green[0]=np.median(cropped_face[:,:,0])
					normalized_red=np.true_divide(integrated_red,np.sqrt(integrated_green*integrated_blue))
					normalized_blue=np.true_divide(integrated_blue,np.sqrt(integrated_green*integrated_red))
					normalized_green=np.true_divide(integrated_green,np.sqrt(integrated_red*integrated_blue))
					# brightness=smooth(np.abs(normalized_green*normalized_red*normalized_blue))
					# print(brightness,21)
					# pltq=np.append(np.fft.rfft(integrated_red),np.zeros(integrated_red.shape[0]/2-1))
					# renormalized_red=normalized_red-brightness
					# renormalized_blue=normalized_blue-brightness
					# renormalized_green=normalized_green-brightness
					pltq=integrated_blue
					pltq=integrated_red-smooth(pltq)
					# pltq=smooth(renormalized_red-(np.sqrt(np.abs(renormalized_green*renormalized_blue))),5)
					pltq2=smooth(pltq)
					pltq3=smooth(np.abs(np.fft.rfft(pltq2)),5)
					pltq3=np.append(pltq3,np.zeros(integrated_red.shape[0]-pltq3.shape[0]))/5
					
					# pltq3=normalized_green
					center=np.mean(np.abs(pltq))
					rad=np.max(np.abs(pltq))-np.min(np.abs(pltq))
					plt.axis([0,1,center-rad,center+rad])
					line1.set_ydata(pltq)
					line2.set_ydata(pltq2)
					line3.set_ydata(pltq3)
					# Name Prediction
					face_to_predict = cv2.resize(cropped_face, FACE_DIM, interpolation = cv2.INTER_AREA)
					face_to_predict = cv2.cvtColor(face_to_predict, cv2.COLOR_BGR2GRAY)
					xo=x
					yo=y
					wo=w
					ho=h
					# Display frame
					cv2.rectangle(rotated_frame, (x,y), (x+int(.8*w+.2*wo),y+int(.8*h+.2*ho)), (0,255,0))
				# rotate the frame back and trim the black paddings
				processed_frame = ut.trim(ut.rotate_image(rotated_frame, rotation * (-1)), frame_scale)

				# reset the optmized rotation map
				current_rotation_map = get_rotation_map(rotation)
				faceFound = True
				break
		if faceFound: 
			frame_skip_rate = 0
			# print "Face Found"
		else:
			frame_skip_rate = SKIP_FRAME
			# print "Face Not Found"
	else:
		frame_skip_rate -= 1
		# print "Face Not Found"
	# print "Frame dimension: ", processed_frame.shape
	if len(cropped_face):
		cv2.imshow("Cropped Face", cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY))
		# face_to_predict = cv2.resize(cropped_face, FACE_DIM, interpolation = cv2.INTER_AREA)
		# face_to_predict = cv2.cvtColor(face_to_predict, cv2.COLOR_BGR2GRAY)
		# name_to_display = svm.predict(clf, pca, face_to_predict, face_profile_names)
	
	cv2.putText(processed_frame, "Press ESC or 'q' to quit.", (5, 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
	cv2.imshow("Real Time Facial Recognition", processed_frame)
	# get next frame
	ret, frame = webcam.read()
webcam.release()
cv2.destroyAllWindows()