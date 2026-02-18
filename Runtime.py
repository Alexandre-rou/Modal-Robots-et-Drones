#!/usr/bin/env python3

import rospy
import math
import time
from geometry_msgs.msg import PoseStamped

start_x = 0.1
start_y = 0.0
prev_x = 0.0
prev_y = 0.0
prev_time = 0.0

def callback(data):
    global prev_x
    global prev_y
    global prev_time
    cur_x = data.pose.position.x
    cur_y = data.pose.position.y
    t = rospy.Time.from_sec(time.time())
    cur_time = t.to_sec() #floating point
    dist = math.sqrt((cur_x-start_x)**2 + (cur_y-start_y)**2)

    if (dist < 1):
        if (prev_x < start_x) and (cur_x > start_x):
            print("New Lap")
            if (prev_time != 0.0):
                print(f"Lap_time: {cur_time - prev_time:.3f}s")
            prev_time = cur_time
    prev_x  = cur_x
    prev_y = cur_y
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('laptime', anonymous=True)

    rospy.Subscriber("/gt_pose",  PoseStamped, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
