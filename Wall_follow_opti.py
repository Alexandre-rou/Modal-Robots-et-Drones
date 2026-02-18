#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

#PID CONTROL PARAMS
kp = 1
kd = 0.09
ki = 0
LookAhead=0.1
thetalook=50
rightDist=0.9
servo_offset = 0.0
prev_error = 0.0 
error = 0.0
integral = 0.0

#WALL FOLLOW PARAMS

VELOCITY = 2.00 # meters per second
CAR_LENGTH = 0.50 # Traxxas Rally is 20 inches or 0.5 meters

class WallFollow:
    """ Implement Wall Following on the car
    """
    def __init__(self):
        #Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/nav'
        self.lidar_sub=rospy.Subscriber('/scan',LaserScan,self.scan_callback)
        self.drive_pub = rospy.Publisher('/nav', AckermannDriveStamped, queue_size=10)

    def getRange(self, data, angle_deg):
        # data: single message from topic /scan
        # angle: between -45 to 225 degrees, where 0 degrees is directly to the right
        # Outputs length in meters to object with angle in lidar scan field of view
        #make sure to take care of nans etc.
        
        global LookAhead
        ascan=data.angle_min
        theta_rad=math.radians(angle_deg)
        index_b=int((math.radians(0)-data.angle_min-math.radians(90))/data.angle_increment)
        index_a=int((theta_rad-data.angle_min-math.radians(90))/data.angle_increment)
        b=data.ranges[index_b]
        a=data.ranges[index_a]
        if math.isnan(a) or math.isinf(a): a=data.range_max
        if math.isnan(b) or math.isinf(b) : b=data.range_max
        alpha=math.atan2((a*math.cos(theta_rad)-b),(a*math.sin(theta_rad)))
        Dt=b*math.cos(alpha)
        
        Dtplus1=Dt+LookAhead*math.sin(alpha)
        return Dtplus1
    
    def getRangeLeft(self, data, angle_deg):
        # data: single message from topic /scan
        # angle: between -45 to 225 degrees, where 0 degrees is directly to the right
        # Outputs length in meters to object with angle in lidar scan field of view
        #make sure to take care of nans etc.
        
        global LookAhead
        ascan=data.angle_min
        theta_rad=math.radians(angle_deg)
        index_b=int((math.radians(0)-data.angle_min-math.radians(90))/data.angle_increment)
        index_a=int((theta_rad-data.angle_min-math.radians(90))/data.angle_increment)
        b=data.ranges[index_b]
        a=data.ranges[index_a]
        if math.isnan(a) or math.isinf(a): a=data.range_max
        if math.isnan(b) or math.isinf(b) : b=data.range_max
        alpha=math.atan2((a*math.cos(theta_rad)-b),(a*math.sin(theta_rad)))
        Dt=b*math.cos(alpha)
        
        Dtplus1=Dt+LookAhead*math.sin(alpha)
        return Dtplus1


    def pid_control(self, error):
        global integral
        global prev_error
        global kp
        global ki
        global kd
        integral+=error
        angle = kp*error+kd*(error-prev_error)+ki*integral
        prev_error=error
        absangle=abs(angle)
        
        if absangle<0.0873:
            velocity=4
        elif absangle<0.1746:
            velocity=2
        else:
            velocity=1
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)
        
    def followRight(self, data, rightDist):
        #Follow right wall as per the algorithm 
        
        realDist=self.getRange(data,thetalook)
        error=rightDist-realDist
        
        self.pid_control(error)
        
    

    def scan_callback(self, data):
        global rightDist
        self.followRight(data,rightDist)


def main(args):
    rospy.init_node("WallFollow_node", anonymous=True)
    wf = WallFollow()
    rospy.sleep(0.1)
    rospy.spin()
    

if __name__=='__main__':
	main(sys.argv)