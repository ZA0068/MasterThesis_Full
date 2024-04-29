#!/bin/bash

ROS2
ros2setup

for i in {1..4} # Number of instances
do
  port=$((19997+i))
  /opt/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu22_04/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_$port_FALSE_TRUE -G"port=$port" &
done
111