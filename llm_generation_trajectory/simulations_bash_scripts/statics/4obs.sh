
NUM_SIMULATIONS=10

# Loop through simulations
for ((i=0; i<NUM_SIMULATIONS; i++)); do

  cd /home/andrea/ros_packages_aggiuntivi/src/llm_generation_trajectory/random_obstacles_generation_scripts/statics

  python3 4obs.py

  sleep 5
  
  echo "Running simulation $((i+1)) of $NUM_SIMULATIONS"
  
  # Run the simulation
  gnome-terminal -- roslaunch llm_generation_trajectory static_4obs.launch

  sleep 10
  
  cd /home/andrea/ros_packages_aggiuntivi/src/llm_generation_trajectory/bag_files/static_obstacles/4_obs

  # gnome-terminal -- rosbag record -O static_obstacles_"$((i+1))".bag --duration=120 /gazebo/model_states /gazebo/bounding_boxes /gptGeneratedPath /callDuration
  gnome-terminal -- rosbag record -O static_obstacles_$((i+1)).bag --duration=50 /gazebo/model_states /gazebo/bounding_boxes /gptGeneratedPath /callDuration

  sleep 1

  gnome-terminal -- rosservice call /gptCall "prompt_type: 'obstacles'
model: 'DeepSeek V3'
few_shot: false" 
  
  sleep 50

  rosnode kill --all
  sleep 5
  killall -9 rosmaster & killall -9 gazebo & killall -9 gzserver & killall -9 gzclient
  
  
  
  echo "Simulation $((i+1)) completed"
  echo "------------------------------"
done