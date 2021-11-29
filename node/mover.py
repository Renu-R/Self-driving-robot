#! /usr/bin/env python

import rospy
import roslib
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
import roslib
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import String
from detect_plate import get_all_chars


class PIDController:

    def __init__(self):
        self.pub = rospy.Publisher('R1/cmd_vel', Twist, 
              queue_size=1)
        self.sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,self.callback)
        self.bridge = CvBridge()
        self.move = Twist()
        self.cX = 0
        self.cY = 0
        self.lastError = 0
        self.p = float(0.0075)
        self.d = float(0.01)
        self.score = rospy.Publisher('/license_plate',String,queue_size = 1)
        self.Scoreclock = rospy.Subscriber('/clock',String,self.setTime)
        self.initialTime = rospy.Time()
        self.time = rospy.Time()
        self.ifFirstCycle = True
        self.notStopped = True
        self.freeze = False

    def setTime(self, clockTime):
        self.time = clockTime.clock
        # print type(self.time)
        if self.ifFirstCycle == True and self.score.get_num_connections() > 0:
            self.initialTime = self.time
            self.score.publish(str('TeamRed,multi21,0,XR58'))
            print "in First Cycle"
            self.ifFirstCycle = False
            self.notStopped = True
        difference = (self.time - self.initialTime).to_sec()

        if difference > 10:
            freeze = False

        if difference > 25 and self.notStopped:
            self.score.publish(str('TeamRed,multi21,-1,XR58'))
            self.notStopped = False
            print "stopped"



        

    def HorizontalCrop(self,image):
      #rows,cols,channels = image.shape
      return image[400:500]

    def VerticalCrop(self,image):
      #rows,cols,channels = image.shape
      return image[300:400,550:680]

    def Centroid(self,gray):
    #  _, img_bin = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)
      #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      M = cv2.moments(gray)
      cX = -1
      cY = -1
      try:
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
      except ZeroDivisionError:
        pass
      
      return (cX,cY)

    def ImageSum(self,img_bin):
      """detects if the amount of white pixel inside the image is above a threshold. Put in vertical or horizontal slice and it can detect if there is road at horizon or turns"""
      threshold = 10000

      #gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
      #_, img_bin = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)

      return np.sum(img_bin) 
    #  inverted = img_bin - 255

    def HorizonDetector(self,image):
      """Detects if there is road on the horizon, a criteria for intersection (none at turns)"""
      return self.ImageSum(self.VerticalCrop(image))
        

    def speed(self, angular):
        if self.freeze:
            self.move.linear.x = 0
            self.move.angular.z = 0
        else:
            self.move.linear.x = 0.08
            self.move.angular.z = angular

        self.pub.publish(self.move)

    def TurnDetector(self,image):
      """Detects if there is a horizontal turn"""
      return self.ImageSum(self.HorizontalCrop(image)) 

    def intersectionDetector(self,image):
      """0 means no intersection, 1 means turning intersection, 2 means symmetric intersection"""
      HorizonThreshold = 50000
      TurnThreshold = 5000000
      CentroidThreshold = 15

      Turn = self.TurnDetector(image)
      Horizon = self.HorizonDetector(image)
      Cx,Cy = self.Centroid(self.HorizontalCrop(image))

      center = image.shape[1]/2

      ifCenter = (Cx<center+CentroidThreshold) and (Cx>center - CentroidThreshold)

      if(Turn<TurnThreshold):
        return 0
      #all intersections have turn, or else its road
      elif(ifCenter):
        return 2
      #if the centroid is centered, then it is symmetric intersection
      elif(Horizon>HorizonThreshold):
        return 1
      #if there is turn and horizon, its turning intersection
      else:
        return 0

    def callback(self, data): 
        #self.score.publish(str('TeamRed,multi21,0,XR58'))
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e

        chars = get_all_chars(cv_image)
        
        (rows, col, channel) = cv_image.shape
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        HSV = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)

        HSVThreshold = cv2.inRange(HSV,(0,0,75),(255,4,89)) 

        intersection = self.intersectionDetector(HSVThreshold)
        if intersection == 2:
            mask = np.ones((HSVThreshold.shape[0],HSVThreshold.shape[1]),dtype="uint8")
            start = (900,0)
            end = (HSVThreshold.shape[1],HSVThreshold.shape[0])
            mask = cv2.rectangle(mask,start,end,(0,0,0),-1)
            HSVThreshold = cv2.bitwise_and(HSVThreshold,mask,mask = None)
        elif intersection == 1:
            mask = np.ones((HSVThreshold.shape[0],HSVThreshold.shape[1]),dtype="uint8")
            start = (900,0)
            end = (HSVThreshold.shape[1],HSVThreshold.shape[0])
#            mask = cv2.rectangle(mask,start,end,(0,0,0),-1)
            mask = cv2.rectangle(mask,(0,0),(300,HSVThreshold.shape[0]),(0,0,0),-1)
            HSVThreshold = cv2.bitwise_and(HSVThreshold,mask,mask = None)

        threshold = 64
        _, img_bin = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        threshold_cropped = HSVThreshold[rows-300:rows]

        M = cv2.moments(threshold_cropped)

        try:
            self.cX = int(M["m10"]/M["m00"])
            self.cY = int(M["m01"]/M["m00"])
        except ZeroDivisionError:
            print "Off Track"

        error = self.cX - col/2
        errorDifference = error - self.lastError
        # print "p:",self.p
        angSpeed = -self.p * error - self.d * errorDifference
        self.speed(angSpeed)

        self.lastError = error

        cv2.circle(HSVThreshold,(self.cX,self.cY), 20,(100,62,255),-1)


        #cv2.imshow("HSVThreshold",HSVThreshold)
        #cv2.waitKey(2)

        cv2.imshow("color",cv_image)
        cv2.waitKey(2)

        cv2.circle(threshold_cropped,(self.cX,10), 20,(100,62,255),-1)

        #cv2.imshow("threshold_cropped",threshold_cropped)
        #cv2.waitKey(2)

        #cv2.imshow("HorizontalCrop",self.HorizontalCrop(HSVThreshold))
        #cv2.waitKey(2)


def main():
    rospy.init_node('controller',anonymous=True)
    PID = PIDController()

    # rospy.spin()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "KeyboardInterrupt"

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()