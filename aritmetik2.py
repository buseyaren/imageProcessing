from collections import OrderedDict
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

#costruct the argument parser and parse tha arguments
ap=argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args=vars(ap.parse_args())

#initialize dlib's face detector(HoG-based) and then create
#the facial landmarks predictor
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["shape_predictor"])

#load the input image, redize it, and convet it grayscale
image=cv2.imread(args["image"])
image=imutils.resize(image, width=500)
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect faces in the grayscale image
rects=detector(gray,1)

#loop over the face detections
for(i, rect) in enumerate(rects):
    #determine the facial landmarks for the face region, then
    #convert the facial landmarks(x,y)-coordinates to a Numpy
    #array
    shape=predictor(gray,rect)
    shape=face_utils.shape_to_np(shape)

    #convert slib's rectangle to a OpenCv-style bounding box
    #[i.e., (x,y,w,h)] then draw the face bounding box
    (x,y,w,h)=face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y),(x+w,y+h), (0,255,0),2)
    #show the face number
    cv2.putText(image, "Face #{}".format(i+1),(x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    #loop over the(x,y)-cordinates for the facial landmarks
    #and draw them on the image
    for (x,y) in shape:
        cv2.circle(image,(x,y),3,(0,255,0),-1)
#show the output image with the face detections+facial labdmarks
cv2.imshow("Output",image)
cv2.waitKey(0)


def rect_to_bb(rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

