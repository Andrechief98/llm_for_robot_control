
NUM_SIMULATIONS=1

# Loop through simulations
for ((i=0; i<NUM_SIMULATIONS; i++)); do

  cd /home/andrea/ros_packages_aggiuntivi/src/llm_generation_trajectory/random_obstacles_generation_scripts/dynamics

  python3 6obs.py

  sleep 5
  
  echo "Running simulation $((i+1)) of $NUM_SIMULATIONS"
  
  # Run the simulation
  gnome-terminal -- roslaunch llm_generation_trajectory dynamic_6obs.launch

  sleep 10

  cd /home/andrea/ros_packages_aggiuntivi/src/llm_generation_trajectory/bag_files/dynamic_obstacles/6_obs

  gnome-terminal -- rosbag record -O dynamic_obstacles_10.bag --duration=200 /gazebo/model_states /gazebo/bounding_boxes /gptGeneratedPath /callDuration /actor1/cmd_vel /actor2/cmd_vel /actor3/cmd_vel /actor4/cmd_vel /actor5/cmd_vel /actor6/cmd_vel

  sleep 1

  gnome-terminal -- rosservice call /gptCall "prompt_type: 'obstacles'
model: 'o3-mini'
few_shot: false" 

  
  # sleep 50

  # rosnode kill --all
  # sleep 5
  # killall -9 rosmaster & killall -9 gazebo & killall -9 gzserver & killall -9 gzclient
  
  
  echo "Simulation $((i+1)) completed"
  echo "------------------------------"
done