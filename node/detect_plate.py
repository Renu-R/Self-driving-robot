#! /usr/bin/env python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def getreckt(sorted_contours, img):
  smallest_item= sorted_contours[-2]
  largest_item= sorted_contours[-1]

  plt.show()

  X = []
  Y =[]

  x,y,w,h= cv2.boundingRect(largest_item)

  xcoordinate1= x
  X.append(xcoordinate1) 
  xcoordinate2= x + w
  X.append(xcoordinate2) 
  ycoordinate1= y
  Y.append(ycoordinate1)
  ycoordinate2= y + h
  Y.append(ycoordinate2)

  x2,y2,w2,h2= cv2.boundingRect(smallest_item)
  x2coordinate1= x2
  X.append(x2coordinate1)  
  x2coordinate2= x2 + w2
  X.append(x2coordinate2)
  y2coordinate1= y2
  Y.append(y2coordinate1) 
  y2coordinate2= y2 + h2
  Y.append(y2coordinate2)

  X.sort()
  Y.sort()

  return ((X,Y))

def get_plate(img):
  img = img[300:, 0:550]
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower = np.array([120,120,58])
  upper = np.array([255,255,204])
  
  mask = cv2.inRange(hsv, lower, upper)

  threshold = 20
  _, img_bin = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
  eroded = cv2.erode(img_bin, None, iterations=3)
  dilate = cv2.dilate(eroded, None, iterations = 2)


  edges= cv2.Canny(dilate, 50,200)
  dilated_edge = cv2.dilate(edges, None, iterations = 4)
  hierarchy, contours,_= cv2.findContours(dilated_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  sorted_contours= sorted(contours, key=cv2.contourArea, reverse= False)

  if len(sorted_contours) >= 2:
    X,Y = getreckt(sorted_contours, img)

  else:
    return np.zeros(1)

  if(X[2]-X[1]>=110 and X[2]-X[1]<=160 and Y[3]-Y[0]>=160 and Y[3]-Y[0]<=200):

    xp = [X[1], X[2]]
    yp = [Y[0],Y[3]]
    crop_img = img[yp[0]:yp[1], xp[0]:xp[1]]
    return crop_img
  else:
    return np.zeros(1)

def get_parking(crop_img):
  crop_img =  crop_img[25:,:]
  hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
  lower = np.array([0,0,0])
  upper = np.array([255,255,12])
  mask = cv2.inRange(hsv, lower, upper)
  mask = cv2.dilate(mask, None, iterations = 2)

  
  points = cv2.findNonZero(mask)
  rect = cv2.minAreaRect(points)
  box = cv2.boxPoints(rect)

  plate = crop_img[int(box[0][1]):]
  plate_num = get_platenum(plate)
  
  return (mask, plate_num)

def get_platenum (img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower = np.array([120,120,58])
  upper = np.array([255,255,204])
  mask = cv2.inRange(hsv, lower, upper)

  return mask

def get_chars_pl(img):
  chars = []
  #blur = cv2.GaussianBlur(img,(3,3),0)
  threshold = 5
  _, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
  hierarchy, cont, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  sorted_contours = sorted(cont, key= lambda c : c[c[:, :, 0].argmin()][0][0])
  #cv2.drawContours(img, cont, -1, color=(255, 255, 255), thickness=cv2.FILLED)
  for contour in sorted_contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
    ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
    ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
    ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

    roi_corners = np.array([box], dtype=np.int32)

    #cv2.polylines(img, roi_corners, 1, (255, 0, 0), 3)

    cropped_image = img[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]

    try:
      chars.append(cv2.resize(cropped_image,(35,45),interpolation = cv2.INTER_NEAREST))
    except:
      break
    #cv2.imwrite('crop.jpg', cropped_image)
  if len(chars) == 4:
    return chars
  else:
    return None

def get_chars_pk(img):
  chars = []
  blur = cv2.GaussianBlur(img,(3,3),0)
  threshold = 5
  _, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
  hierarchy, cont,_ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  sorted_contours = sorted(cont, key= lambda c : c[c[:, :, 0].argmin()][0][0])
  #cv2.drawContours(img, cont, -1, color=(255, 255, 255), thickness=cv2.FILLED)
  for contour in sorted_contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
    ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
    ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
    ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

    roi_corners = np.array([box], dtype=np.int32)

    #cv2.polylines(img, roi_corners, 1, (255, 0, 0), 3)

    cropped_image = img[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]

    try:
      chars.append(cv2.resize(cropped_image,(35,45),interpolation = cv2.INTER_NEAREST))
    except:
      break
    #cv2.imwrite('crop.jpg', cropped_image)
  if len(chars) == 2:
    return chars
  else:
    return None


def get_all_chars(frame):
	plate = get_plate(frame)
	if not np.all(plate == 0):
		cv2.imshow("plate",plate)
		cv2.waitKey(2)
		parking_id, plate_num = get_parking(plate)

		chars1 = get_chars_pk(parking_id)
		chars2 = get_chars_pl(plate_num)

		if( chars1 != None and chars2 != None):
			return (chars1[1], chars2)
	
	return (None, None)