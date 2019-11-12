import cv2
import numpy as np
import dlib
import array
import time
from math import hypot

cap = cv2.VideoCapture(0)

#egula file theke face ar eye detection er header file. amar janar kno dorkar nai
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#ei array te random value gula saved ase ar cnt counter ta refresh er jonno rakhsi
arr_left = array.array('i',[])  
arr_right = array.array('i',[])
decision_left =  array.array('i',[])
decision_right =  array.array('i',[])
fixation_time = []

cnt = 0
flag = 1
k_cnt = 0
cnt_fin = 0

#screen e kun font e print hobe sheta. onno font er cheye eta beshi popular
font = cv2.FONT_HERSHEY_SIMPLEX

start_time = time.time()

while True:
	#ekhane rgb value gula re gray te convertion hocche
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = detector(gray)
	for face in faces:
		landmarks = predictor(gray, face)

		#gaze detection
		#egula hoilo point. ekta picture ase jetar upor depended. manusher face er bivinno part k point kore. left eye holo 36-41 ar right eye 42-47
		#polyline holo jeta print kortese amar chokh detect korte partese ki na. red hyperbola ta
		left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
					(landmarks.part(37).x, landmarks.part(37).y),
					(landmarks.part(38).x, landmarks.part(38).y),
					(landmarks.part(39).x, landmarks.part(39).y),
					(landmarks.part(40).x, landmarks.part(40).y),
					(landmarks.part(41).x, landmarks.part(41).y)], np.int32)
		right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
					(landmarks.part(43).x, landmarks.part(43).y),
					(landmarks.part(44).x, landmarks.part(44).y),
					(landmarks.part(45).x, landmarks.part(45).y),
					(landmarks.part(46).x, landmarks.part(46).y),
					(landmarks.part(47).x, landmarks.part(47).y)], np.int32)
		#print (landmarks.part(37).x, landmarks.part(37).y, landmarks.part(46).x, landmarks.part(46).y)
		if landmarks.part(37).x >= 220 and landmarks.part(37).x <= 250 and landmarks.part(37).y >=130 and landmarks.part(37).y <= 150 and landmarks.part(46).x >=375 and landmarks.part(46).x <=405 and landmarks.part(46).y>=145 and landmarks.part(46).y<=165:
			cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
			cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 2)
		
			#minimum ar maximum value nicchi jeta amar chokher initial position bujhabe
			left_min_x = np.min(left_eye_region[:, 0])
			left_max_x = np.max(left_eye_region[:, 0])
			left_min_y = np.min(left_eye_region[:, 1])
			left_max_y = np.max(left_eye_region[:, 1])

			right_min_x = np.min(right_eye_region[:, 0])
			right_max_x = np.max(right_eye_region[:, 0])
			right_min_y = np.min(right_eye_region[:, 1])
			right_max_y = np.max(right_eye_region[:, 1])

			left_eye = frame[left_min_y : left_max_y, left_min_x : left_max_x]
			right_eye = frame[right_min_y : right_max_y, right_min_x : right_max_x]
			
			#gray te convert kore threshold fix korsi jst jate exact eye pupil location ta pai
			gray_left_eye = cv2.cvtColor(left_eye,cv2.COLOR_BGR2GRAY)
			gray_right_eye = cv2.cvtColor(right_eye,cv2.COLOR_BGR2GRAY)

			_, threshold_eye_left = cv2.threshold(gray_left_eye, 70, 255, cv2.THRESH_BINARY_INV)
			_, threshold_eye_right = cv2.threshold(gray_right_eye, 70, 255, cv2.THRESH_BINARY_INV)


			height, width = threshold_eye_left.shape
			
			threshold_left_side = threshold_eye_left[0 : height, 0 : int(width/2)]
			threshold_left_side_white = cv2.countNonZero(threshold_left_side)
			
			height, width = threshold_eye_left.shape
			threshold_right_side = threshold_eye_right[0 : height, int(width/2) : width]
			threshold_right_side_white = cv2.countNonZero(threshold_right_side)

			cv2.putText(frame, str(threshold_left_side_white), (50,100), font, 2, (0,0,255), 3)
			cv2.putText(frame, str(threshold_right_side_white), (50, 150), font, 2, (0, 0, 255), 3)
			
			#egula resize korsi karon echara eye chara o face er kichu ongsho chole ashe
			left_eye = cv2.resize(left_eye, None, fx=5, fy=5)
			right_eye = cv2.resize(right_eye, None, fx=5, fy=5)
			
			#array khulsi jeta random value gula save kore, sort kore, minimum ar maximum ta pathabe
			arr_left.append(threshold_left_side_white)
			arr_right.append(threshold_right_side_white)

			arr_left = sorted(arr_left)
			arr_right = sorted(arr_right)

			left_len = len(arr_left)
			right_len = len(arr_right)

			end_time = time.time()

			#print(arr_left[2], arr_left[left_len-1],left_len,end_time-start_time)
			if flag==1:
				decision_left.append(arr_left[int((left_len-1)/2)])
				decision_right.append(arr_right[int((right_len-1)/2)])

				decision_left_len = len(decision_left)
				decision_right_len = len(decision_right)

				keeper_left = arr_left[int((left_len-1)/2)]
				keeper_right = arr_right[int((right_len-1)/2)]

				flag = 0
				cnt_fin = cnt_fin + 1
			else:
				if (keeper_left-arr_left[int((left_len-1)/2)]) > 10 or (keeper_left-arr_left[int((left_len-1)/2)]) < -10 or (keeper_right-arr_right[int((right_len-1)/2)]) > 10 or (keeper_right-arr_right[int((right_len-1)/2)]) < -10:
					k_cnt = k_cnt + 1
					if k_cnt == 3:
						end_time = time.time()
						if int(end_time-start_time)>0:
							fixation_time.append([average_left, average_right, int(end_time-start_time)])			
						del decision_left[:]
						del decision_right[:]
						decision_left = array.array('i',[arr_left[int((left_len-1)/2)]])
						decision_right = array.array('i',[arr_right[int((right_len-1)/2)]])

						decision_left_len = len(decision_left)
						decision_right_len = len(decision_right)
						print(threshold_left_side_white, threshold_right_side_white, '|', arr_left[int((left_len-1)/2)],arr_right[int((right_len-1)/2)],'|',decision_left[int((decision_left_len-1)/2)],decision_right[int((decision_right_len-1)/2)],'|',average_left,average_right, '|',fixation_time[(len(fixation_time)-1)])

						keeper_left = arr_left[int((left_len-1)/2)]
						keeper_right = arr_right[int((right_len-1)/2)]
						p = k_cnt
						k_cnt = 0
						cnt_fin = 1
						end_time = 0
						start_time = 0
						start_time = time.time()
				else:
					decision_left.append(arr_left[int((left_len-1)/2)])
					decision_right.append(arr_right[int((right_len-1)/2)])

					keeper_left = arr_left[int((left_len-1)/2)]
					keeper_right = arr_right[int((right_len-1)/2)]

					decision_left_len = len(decision_left)
					decision_right_len = len(decision_right)
					cnt_fin = cnt_fin + 1
			average_left = int(sum(decision_left) / len(decision_left))
			average_right = int(sum(decision_right) / len(decision_right))
			
			print(threshold_left_side_white, threshold_right_side_white, '|', arr_left[int((left_len-1)/2)],arr_right[int((right_len-1)/2)],'|',decision_left[int((decision_left_len-1)/2)],decision_right[int((decision_right_len-1)/2)],'|',average_left,average_right)
			cnt = cnt + 1
			if cnt == 10:
				a = arr_left[int((left_len-1)/2)]
				b = arr_right[int((right_len-1)/2)]
				del arr_left[:]
				del arr_right[:]
				arr_left = array.array('i',[a])  
				arr_right = array.array('i',[b])  
				cnt = 1
			if cnt_fin == 16:
				a = decision_left[int((decision_left_len-1)/2)]
				b = decision_right[int((decision_right_len-1)/2)]

				del decision_left[:]
				del decision_right[:]

				decision_left = array.array('i',[a,a])  
				decision_right = array.array('i',[b,b])

				decision_left_len = len(decision_left)
				decision_right_len = len(decision_right)

				keeper_left = a
				keeper_right = b
				cnt_fin = 1
				k_cnt = 0

	#ekhane rectangle ta holo jeta eye er position er jnno fix dhorsi
	cv2.rectangle(frame,(200,120),(440,180),(0,255,0),3)
	cv2.imshow("Frame", frame)
	
	#ar eta temon ksu na. control + 'c' dile jate program terminate hoy oita
	key = cv2.waitKey(1)
	if key == 27:
		break

#kaj shesher por data release ar camera ke rest
cap.release()
cv2.destroyAllWindows()
