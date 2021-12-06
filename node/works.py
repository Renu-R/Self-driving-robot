#!/usr/bin/env python
#print "hola"
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
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import time

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend



from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)



class PIDController:



    def __init__(self):
        self.number_model = models.load_model('/home/fizzer/ros_ws/src/controller_pkg/node/NumModel')
        self.letter_model = models.load_model('/home/fizzer/ros_ws/src/controller_pkg/node/my_model_letters.h5')
        self.pub = rospy.Publisher('R1/cmd_vel', Twist, 
              queue_size=1)
        self.sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,self.callback)
        self.bridge = CvBridge()
        self.move = Twist()
        self.cX = 0
        self.cY = 0
        self.lastError = 0
        self.p = float(0.0075)
        self.d = float(0.003)
        self.d = float(0.01)
        self.score = rospy.Publisher('/license_plate',String,queue_size = 1)
        self.Scoreclock = rospy.Subscriber('/clock',String,self.setTime)

        self.initialTime = rospy.Time()
        self.time = rospy.Time()
        self.ifFirstCycle = True
        self.notStopped = True
        self.freeze = True
        self.publish = False
        "whether to publish when frozen or not"

        self.atRedLine = False
        self.pedOnRoad = False

        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.newRedLine = False
        self.lastAbove = True
        "safer to detect pedestrian to be above road last frame when everything start so we dont move immediately"

        self.superspeed = False
        self.slowspeed = False

        self.moveTime = rospy.Time()
        self.stopTime  = rospy.Time()
        self.count = 0
        self.ParkingIDs = []
        self.num1 = []
        self.num2 = []
        self.letter1 = []
        self.letter2 = []
        self.publishedPlate = True
        self.plateTime = rospy.Time()
        self.goin = False
        self.platesonright = False

        time.sleep(1)

    def setTime(self, clockTime):
        self.time = clockTime.clock
        # print type(self.time)
        if self.ifFirstCycle == True and self.score.get_num_connections() > 0:
            self.initialTime = self.time
            self.score.publish(str('AllOfHerRainNews,multi21,0,XR58'))
            print "in First Cycle"
            self.ifFirstCycle = False
            self.notStopped = True
            self.freeze = False
            self.publish = True
        difference = (self.time - self.initialTime).to_sec()


        if difference > 418 and self.notStopped:
            self.score.publish(str('AllOfHerRainNews,multi21,-1,XR58'))
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
            print "in freezing"
            self.move.linear.x = 0
            self.move.angular.z = 0
            if self.publish:
                print "publish Freeze"
                self.pub.publish(self.move)

        else:
            # self.move.linear.x = 0.15
            self.move.linear.x = 0.1
            self.move.angular.z = angular

            if self.superspeed:
                self.move.linear.x = 0.2
                self.move.angular.z = angular*1.5
            elif self.slowspeed:
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
      elif (ifCenter and self.goin):
        print "we here!"
      elif(ifCenter):
        return 2
      #if the centroid is centered, then it is symmetric intersection
      elif self.goin:
        if (Horizon>HorizonThreshold and Turn<TurnThreshold):
            return 1
      elif(Horizon>HorizonThreshold ): #and Turn<TurnThreshold
        return 1
      #if there is turn and horizon, its turning intersection
      else:
        return 0

    def redLineDetection(self, image):
        hsvCross = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower = np.array([0,97,125])
        upper = np.array([10,255,255])
        mask = cv2.inRange(hsvCross,lower,upper)

        cutMask = mask[500:,:]
        _,bin = cv2.threshold(cutMask,200,1,cv2.THRESH_BINARY)
        sum = np.sum(bin)

        if(sum>13000):
            return True
        else:
            return False

    def pedDetection(self,image):
        image = image[100:750,:]
        HSVFramePed = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower = np.array([72,0,0])
        upper = np.array([136,185,113])
        PedMask = cv2.inRange(HSVFramePed,lower,upper)
        foreGround = self.backSub.apply(PedMask)
        eroded = cv2.erode(foreGround,None,iterations=3)
        dilated = cv2.dilate(eroded,None,iterations=3)

        """default values:"""
        maxDifference = -100
        difference = -100
        "negative so if theres no line it'll detect pedestrian in road"


        sum = np.sum(dilated)

        M = cv2.moments(dilated)

        centroidX = 0
        centroidY = 0
        noObject = False

        if np.sum(dilated) < 10000:
            noObject = True

        if not noObject:
            try:
              centroidX = int(M["m10"]/M["m00"])
              centroidY = int(M["m01"]/M["m00"])
            except ZeroDivisionError:
              noObject = True
            
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower = np.array([0,0,215])
        upper = np.array([255,25,255])
        mask = cv2.inRange(hsv,lower,upper)

        edge = cv2.Canny(mask,50,200)

        lines = cv2.HoughLines(edge,1,np.pi/200,190,None,0,0)

        row = edge.shape[0]
        col = edge.shape[1]
        lineImage = np.ones((row,col))

        #  crossWalk = crosswalkImg.copy()
        above = False



        if True:
            if not noObject:
                if lines is not None:
                    for i in range(0, len(lines)):
                        rho = lines[i][0][0]
                        theta = lines[i][0][1]
                        slope = -np.cos(theta)/np.sin(theta)
                        offset = rho/np.sin(theta)

                        difference = slope*centroidX + offset - centroidY

                        # print(difference)

                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                        # lineImage = cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

                        print "difference =",difference
                        # print "numLInes = ", len(lines)

                        if difference > maxDifference:
                            maxDifference = difference

                        # if difference > 30 and not offTrack:
                        #     above = True
            if maxDifference > 10 or noObject:
                above = True

            print "above",above
            
            # cv2_imshow(mask)
            # cv2_imshow(edge)
            # cv2.circle(lineImage,(centroidX,centroidY),10,(0,0,255),-1)
            # plt.imshow(lineImage)
            # plt.show()

          
          
        return (dilated,above)


    def extractSingleOutputNum(self,inputVector):
        index = np.argmax(inputVector)

        asciiNum = index

        asciiNum += (48)

        return chr(asciiNum)

    def extractSingleOutputLet(self,inputVector):
        index = np.argmax(inputVector)

        asciiNum = index

        asciiNum += (65)

        return chr(asciiNum)

    def mostFrequent(self,List):
        return max(set(List), key = List.count)

    def callback(self, data): 
        #self.score.publish(str('TeamRed,multi21,0,XR58'))
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e

        chars = get_all_chars(cv_image, self.platesonright)

        if chars != None:
            parkingID = chars[0][1]
            platenum1 = chars[1][2]
            platenum2 = chars[1][3]
            platelet1 = chars[1][0]
            platelet2 = chars[1][1]

            shape = parkingID.shape
            parkingID = parkingID.reshape(1,shape[0],shape[1],1)
            platenum1 = platenum1.reshape(1,shape[0],shape[1],1)
            platenum2 = platenum2.reshape(1,shape[0],shape[1],1)
            platelet1 = platelet1.reshape(1,shape[0],shape[1],1)
            platelet2 = platelet2.reshape(1,shape[0],shape[1],1)

            global sess1
            global graph1
            with graph1.as_default():
               set_session(sess1)
               parkPredict = self.extractSingleOutputNum(self.number_model.predict(parkingID))

               platenum1predict = self.extractSingleOutputNum(self.number_model.predict(platenum1))
               platenum2predict = self.extractSingleOutputNum(self.number_model.predict(platenum2))
               platelet1predict = self.extractSingleOutputLet(self.letter_model.predict(platelet1))
               platelet2predict = self.extractSingleOutputLet(self.letter_model.predict(platelet2))

               self.ParkingIDs.append(parkPredict)
               self.num1.append(platenum1predict)
               self.num2.append(platenum2predict)
               self.letter1.append(platelet1predict)
               self.letter2.append(platelet2predict)

            print "parkingID:",parkPredict
            print "platelet1:",platelet1predict
            print "platelet2:", platelet2predict
            print "platenum1:",platenum1predict
            print "platenum2:", platenum2predict
            
            #
            self.publishedPlate = False
            self.plateTime = self.time

               
        timesincePlate = (self.time - self.plateTime).to_sec()
        #print self.plateTime

        if self.publishedPlate == False and timesincePlate > 1:
            #print "in publishedPlate"
            self.publishedPlate = True
            parkPredict = self.mostFrequent(self.ParkingIDs)

            if int(parkPredict) == 1:
                self.goin = True
                self.platesonright = True

            #if int(parkPredict) == 7:
                #self.goin = False

            plateletter1predict = self.mostFrequent(self.letter1)
            plateletter2predict = self.mostFrequent(self.letter2)
            platenum1predict = self.mostFrequent(self.num1)
            platenum2predict = self.mostFrequent(self.num2)
            self.score.publish(str('AllOfHerRainNews,multi21,'+str(parkPredict)+','+str(plateletter1predict)+str(plateletter2predict)+str(platenum1predict)+str(platenum2predict)))
            print "sent"
            print self.goin
            self.ParkingIDs = []
            self.num1 = []
            self.num2 = []
            self.letter1 = []
            self.letter2 = []


        self.atRedLine = self.redLineDetection(cv_image)

        if self.atRedLine:
            if self.newRedLine:
                self.backSub = cv2.createBackgroundSubtractorMOG2()
                self.newRedLine = False
                self.stopTime = self.time

            print "at red line"
            self.freeze = True
            self.publish = True
            # self.pedOnRoad = self.pedDetection(cv_image)

            self.pedOnRoad = True
            backgroundSub,above = self.pedDetection(cv_image)

            timeSinceStop = self.time - self.stopTime

            if above and not self.lastAbove and timeSinceStop.to_sec() > 2:
                self.moveTime = self.time

            self.lastAbove = above

            cv2.imshow("backgroundsub",backgroundSub)
            cv2.waitKey(1)

            print np.sum(backgroundSub)

            print "background subtracted"

            if timeSinceStop.to_sec() > 4:
                self.moveTime = self.time
                """If the car has been stopped for over 6 seconds it is likely pedestrian is stuck, etc, so we move"""

        else:
            self.freeze = False
            self.newRedLine = True
            cv2.imshow("backgroundsub",cv_image)
            cv2.waitKey(2)

        timeSinceLastFreeze = self.time - self.moveTime
        print "freeze",self.freeze

        # print timeSinceLastFreeze

        timeSinceStart = self.moveTime - self.initialTime
        

        # if (timeSinceLastFreeze).to_sec() < 15 and (timeSinceLastFreeze - timeSinceStart).to_sec > 1:
        if (timeSinceLastFreeze).to_sec() < 2.5 and timeSinceStart.to_sec() > 1:
            self.superspeed = True
            self.freeze = False
        else:
            self.superspeed = False
            """move for 10 seconds and not freeze for the next 10 seconds (so it wont stop at next red line)"""

        (rows, col, channel) = cv_image.shape
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        HSV = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)

        HSVThreshold = cv2.inRange(HSV,(0,0,75),(255,4,89)) 

        intersection = self.intersectionDetector(HSVThreshold)

        self.slowspeed = False

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

        # inverted = np.invert(threshold_cropped)

        M = cv2.moments(threshold_cropped)

        try:
            self.cX = int(M["m10"]/M["m00"])
            self.cY = int(M["m01"]/M["m00"])
            #print self.cX
        except ZeroDivisionError:
            print "Off Track"

        error = self.cX - col/2
        errorDifference = error - self.lastError
        #print "p:",self.p
        # print "p:",self.p
        angSpeed = -self.p * error - self.d * errorDifference
        self.speed(angSpeed)

        self.lastError = error

        cv2.circle(HSVThreshold,(self.cX,self.cY), 20,(100,62,255),-1)

        cv2.imshow("color",cv_image)
        cv2.waitKey(2)

        #print intersection   

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