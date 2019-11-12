# Gaze-Detection-with-fixation-time-using-webcam

Introducation:
Its a complex code to detect eye pupil, get the gaze point and fixation time of it. to say in simple words, fixation time is the total time a person spends to look at anything at a glance. The code gives the output of fixation time and the points. its not the coordinate point. but if we use fixed head position, we might get same coordinate for any point of the screen at any different test. the fixation time and point are shown so that we know for how long the user is looking at what point. different filters like stabilizing filter, averaging filter, median filter etc are used to get non-fluctuate value

Requirements & repository:
-> any webcame
-> python3 IDE
-> OpenCV2
-> Numpy
-> dlib
-> shape predictor 68 face landmarks (link: https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)

How to use:
Run the code using python3 IDE and keeping "shape_predictor_68_face_landmarks.dat" at the same folder as the code. the leftmost point of the left eye and rightmost point of the right eye should be close to the border of the rectangular box. only then the code will run. and give the output

Issues:
-> cant run in low light
-> depth sensing code is not there, so the head must have to be fixed
-> cant wear spectacles as the reflected light from the monitor screen become the obstacles for the code to detect pupil

