#!/usr/bin/env python

import sys
import numpy as np
import time
import math

#ROS Imports
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


#PID CONTROL PARAMS
kp = 0.6
kd = 0.09
ki = 0
LookAhead=0.1
thetalook=50
rightDist=0.9
servo_offset = 0.0
prev_error = 0.0 
error = 0.0
integral = 0.0
car_width=0.2
max_v=6.5
min_v=1.2



class reactive_follow_gap:
    def __init__(self):
        
        #Topics & Subscriptions,Publishers
        lidarscan_topic = '/scan'
        drive_topic = '/nav'

        self.lidar_sub = rospy.Subscriber(lidarscan_topic,LaserScan,self.lidar_callback,queue_size=1)

        self.drive_pub = rospy.Publisher(drive_topic,AckermannDriveStamped,queue_size=10)
        
       
        
        # State variables
        self.last_angle = 0.0 
        self.mean_period=3
        self.range_history=[]
    """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
            My implementation:
            1.Mean over MEAN_PERIOD Scans before the current reading
            2.Cap high values 
        """    
    def preprocess_lidar(self, ranges):
  
        # Replace NaNs or Infs with 0 or a max range
        ranges = np.nan_to_num(ranges, nan=0.0)

        # Maintain a rolling window of scans
        self.range_history.append(ranges)

        if len(self.range_history)>self.mean_period:
            self.range_history.pop(0)

        mean_ranges=np.mean(self.range_history,0)
        
  #      mean_ranges=np.where(mean_ranges>3.0,3.0,mean_ranges)
        

        return mean_ranges
    

    """ Return the start index & end index of the max gap in free_space_ranges """
    def find_max_gap(self, free_space_ranges):
    

        max_start = 0
        max_len = 0
        current_start = 0
        current_len = 0
        threshold = np.quantile(free_space_ranges,0.75)
       

        for i, val in enumerate(free_space_ranges):
            if val>=threshold:
                if current_len==0:
                    current_start=i
                current_len+=1
                if current_len>max_len:
                    max_len=current_len
                    max_start=current_start
            else : current_len=0

        return max_start, max_start + max_len - 1
        
    """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	Naive: Choose the furthest point within ranges and go there
        """
    
    def extend_disparities(self, ranges, threshold=0.5):
    
        proc_ranges = np.copy(ranges)
        width_indices = 40 # indices to cover (depends on car width)
    
        for i in range(1, len(ranges)):
            diff = ranges[i] - ranges[i-1]
            if abs(diff) > threshold:
                #Found a disparity
                if diff > 0: # i-1 is closer
                    start = i
                    end = min(len(ranges), i + width_indices)
                    proc_ranges[start:end] = np.minimum(proc_ranges[start:end], ranges[i-1])
                else: # i is closer
                    start = max(0, i - width_indices)
                    end = i
                    proc_ranges[start:end] = np.minimum(proc_ranges[start:end], ranges[i])
        return proc_ranges

    def find_best_point(self, start_i, end_i, ranges):
        ranges2=ranges[start_i:end_i+1]**2
        if start_i >= end_i or start_i < 0:
            
            return (start_i+end_i)//2
        #max_in_gap=np.argmax(ranges[start_i:end_i+1])
        max_in_gap=int(round(np.average(np.arange(start_i,end_i+1),weights=ranges2)))
        return max_in_gap

    def pid_control(self, error,target_dist):
 
        global integral
        global prev_error
        global kp
        global ki
        global kd
        global max_v
        global min_v
        

        integral+=error
        angle = kp*error+kd*(error-prev_error)+ki*integral
        prev_error=error
        dist_factor = target_dist / 3.0
        angle_factor = max(0, 1.0 - abs(angle) / 0.5)
    
        velocity = max_v * angle_factor * dist_factor
        
        # Making sure velocity stays in bounds
        velocity = np.clip(velocity, min_v, max_v)
      
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)

    """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
    def lidar_callback(self, data):
        
        alpha=0.5#Smoothing factor
        global car_width
        angle_min=data.angle_min
        angle_increment=data.angle_increment

        #PROCESSING ranges
        ranges = data.ranges
        ranges=np.array(ranges)
        # DEBUG       print("header=",data.header,"angle min et max ",data.angle_min,data.angle_max,"angle_increment=",data.angle_increment,"time increment and scan time",data.time_increment,data.scan_time)
        # DEBUG       print("length=",len(ranges),"mean=",np.mean(ranges),np.max(ranges),np.min(ranges))
        proc_ranges = self.preprocess_lidar(ranges)
        #Replacing bubble with edge detection
        #proc_ranges=self.extend_disparities(ranges)
        
        

        #Artificially reducing Field Of View
        view_start = 300
        view_end = 780
        proc_ranges[0:view_start] = 0
        proc_ranges[view_end:] = 0
        
        
        #Creating a safety bubble 
        closest_idx=np.argmin(ranges[300:780])+300
        min_distance=ranges[closest_idx]
        
        bubble_radius =int( math.atan(car_width/min_distance)//angle_increment )# Dynamic based on distance to obstacle
        start_b = max(0, closest_idx - bubble_radius)
        end_b = min(len(proc_ranges), closest_idx + bubble_radius)
        
        
        proc_ranges[start_b:end_b] = 0.0
        

         
        #Find max length gap 
        max_gap_start,max_gap_end=self.find_max_gap(proc_ranges)

    

        #Find the best point in the gap 
        best_point=self.find_best_point(max_gap_start,max_gap_end,ranges)
        mid_point=(max_gap_start+max_gap_end)//2
        #best_point=(best_point+mid_point)//2
        
        
        #Smoothing angle
        target_angle = best_point * angle_increment + angle_min
        smoothed_angle = alpha * target_angle + (1 - alpha) * self.last_angle
        self.last_angle = smoothed_angle
        
        #Publish Drive message
        self.pid_control(smoothed_angle,ranges[best_point])
        '''
        print(list(proc_ranges))#DEBUG
        print(min_distance)#DEBUG
        print(closest_idx)#DEBUG
        print(start_b,end_b)#DEBUG
        print('max gap ',max_gap_start,max_gap_end)#DEBUG
        print('best point: ',best_point)#DEBUG
        time.sleep(15)#DEBUG'''

def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    rfgs = reactive_follow_gap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv) 