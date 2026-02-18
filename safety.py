#!/usr/bin/env python
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
import time
class Safety(object):
    
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        rospy.init_node('safety_node')
        self.speed = 0
        self.brake = rospy.Publisher('/brake', AckermannDriveStamped, queue_size=10)
        self.brake_bool = rospy.Publisher('/brake_bool', Bool, queue_size=10)
        self.scan=rospy.Subscriber('/scan',LaserScan,self.scan_callback)
        self.odom=rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        
    def odom_callback(self, odom_msg):
        self.speed = odom_msg.twist.twist.linear.x
        

    def scan_callback(self, scan_msg):
        stop= AckermannDriveStamped()
        stop.drive.speed=0
        stop.drive.acceleration=0.0
        
        angle=scan_msg.angle_min
        
        for r in scan_msg.ranges:
            if math.isinf(r)== True:
                r=0
            angle= (angle + scan_msg.angle_increment)
            ang=angle%3.14159265
            if ang < 0.27 or ang > 2.87:
                if self.speed==0:
                    TTC=math.inf
                else: 
                    TTC=r/(self.speed*math.cos(ang))
                

                if TTC<1 and TTC>0:
                    print(self)
                    self.brake.publish(stop)
                    brake_bool_msg=Bool()
                    brake_bool_msg.data=True
                    self.brake_bool.publish(brake_bool_msg)
                    print(TTC)
                    print(self.speed)
                    print(ang)
                    time.sleep(0.1)
                    
      


def main():

    sn = Safety()
    rospy.spin()
if __name__ == '__main__':
    main()

