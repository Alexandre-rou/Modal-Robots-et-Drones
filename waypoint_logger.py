#!/usr/bin/env python

import rospy
import numpy as np
import csv
import os
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point

class WaypointLogger:
    def __init__(self):
        rospy.init_node('waypoint_logger', anonymous=True)

        # Parameters
        self.filename = '/home/alek/track_waypoints.csv'
        self.min_distance = 0.1  # Logging every 10 cm
        
        # State variables
        self.last_x = None
        self.last_y = None
        
        # preparing csv file
        self.file = open(self.filename, 'w')
        self.writer = csv.writer(self.file)
        # Header : x, y, yaw, velocity
        self.writer.writerow(['x', 'y', 'yaw', 'velocity'])
        
        # Odometry Subscriber 
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        rospy.loginfo(f"Waypoint logging file {self.filename}...")
        rospy.on_shutdown(self.shutdown)

    def odom_callback(self, msg):
        # extracting position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # calculating distance to last logged point
        if self.last_x is not None:
            dist = np.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
            if dist < self.min_distance:
                return # Trop proche, on ignore ce point

        # extracting yaw
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        yaw = euler_from_quaternion(quaternion)[2]
        
        # extracting velocity
        velocity = msg.twist.twist.linear.x
        
        # logging waypoint
        self.writer.writerow([x, y, yaw, velocity])
        
        # updating state variables
        self.last_x = x
        self.last_y = y
        rospy.loginfo(f"Logged_point : x={x:.2f}, y={y:.2f}")

    def shutdown(self):
        rospy.loginfo("Closing...")
        self.file.close()

if __name__ == '__main__':
    try:
        WaypointLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass