import rosbag
import rospy
import matplotlib.pyplot as plt
import numpy as np
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Pose, PoseArray
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion
import matplotlib.patches as patches

def extract_actor_velocities(bag, num_actors):
    """Extract actor velocities from /actor{i}/cmd_vel topics"""
    velocities = {}
    
    for i in range(1, num_actors + 1):
        topic = f"/actor{i}/cmd_vel"
        for _, msg, _ in bag.read_messages(topics=[topic]):
            velocities[i] = (msg.linear.x, msg.angular.z)  # Store both linear and angular velocity
            break
    
    return velocities

def get_model_info(model_states, model_name):
    """Get position, orientation and bounding box info for a model"""
    for i, name in enumerate(model_states.name):
        if name == model_name:
            pose = model_states.pose[i]
            # Convert quaternion to euler angles
            orientation = euler_from_quaternion([pose.orientation.x, 
                                                pose.orientation.y,
                                                pose.orientation.z,
                                                pose.orientation.w])[2]  # yaw
            
            # Default bounding box dimensions (adjust according to your simulation)
            bbox_length = 0.5  # length along the orientation axis
            bbox_width = 0.3   # width perpendicular to orientation
            
            return {
                'position': (pose.position.x, pose.position.y),
                'orientation': orientation,
                'bbox': (bbox_length, bbox_width)
            }
    return None

def compute_future_positions(initial_pos, initial_ori, velocity, angular_velocity, timesteps, dt=0.5):
    """Predict future positions of actors with constant velocity and angular velocity"""
    future_positions = []
    current_pos = list(initial_pos)
    current_ori = initial_ori
    
    for t in range(timesteps):
        # Update orientation first (if there's angular velocity)
        current_ori = initial_ori
        
        # Calculate velocity components based on current orientation
        v_x = velocity * np.cos(current_ori)
        v_y = velocity * np.sin(current_ori)
        
        # Update position
        current_pos[0] += v_x * dt
        current_pos[1] += v_y * dt
        
        future_positions.append((current_pos[0], current_pos[1], current_ori, t*dt))
    
    return future_positions

def read_rosbag(bag_file, num_actors=2):
    """Read a rosbag file and plot the simulation over time"""
    try:
        bag = rosbag.Bag(bag_file, 'r')
        model_states = None
        gpt_path = None
        
        # Find the last model states message to get final positions
        for topic, msg, _ in bag.read_messages():

            if topic == "/gazebo/model_states":
                model_states = msg
            elif topic == "/gptGeneratedPath":
                gpt_path = msg
        
        if not model_states:
            rospy.logwarn("/gazebo/model_states data not found in rosbag.")
            return
        
        # Extract actor velocities
        velocities = extract_actor_velocities(bag, num_actors)
        
        # Get initial positions, orientations and bounding boxes
        actors_info = {}
        robot_info = None
        
        for i in range(1, num_actors + 1):
            actor_name = f"actor{i}"
            info = get_model_info(model_states, actor_name)
            if info:
                actors_info[i] = info
        
        robot_info = get_model_info(model_states, "locobot")
        
        # Calculate future positions for actors
        actor_trajectories = {}
        max_timesteps = 20  # Default if no GPT path
        
        if gpt_path:
            # Handle both PoseArray and Path message types
            if hasattr(gpt_path, 'poses'):  # PoseArray
                max_timesteps = len(gpt_path.poses)
        
        for actor_id, info in actors_info.items():
            vel, ang_vel = velocities.get(actor_id, (0, 0))
            actor_trajectories[actor_id] = compute_future_positions(
                info['position'],
                info['orientation'],
                vel,
                ang_vel,
                max_timesteps
            )
        
        # Extract GPT path points with timestamps
        gpt_path_points = []
        if gpt_path:
            if hasattr(gpt_path, 'poses'):  # PoseArray
                for i, pose in enumerate(gpt_path.poses):
                    gpt_path_points.append((pose.position.x, 
                                          pose.position.y,
                                          i*0.5))  # 0.5s timestep
            elif hasattr(gpt_path, 'poses'):  # Path (with PoseStamped)
                for i, pose_stamped in enumerate(gpt_path.poses):
                    gpt_path_points.append((pose_stamped.pose.position.x, 
                                          pose_stamped.pose.position.y,
                                          i*0.5))
        
        # Plot everything
        plot_trajectories(actor_trajectories, gpt_path_points, robot_info, actors_info)
        
    except Exception as e:
        rospy.logerr(f"Error reading rosbag: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bag.close()

def plot_trajectories(actor_trajectories, gpt_path_points, robot_info, actors_info):
    """Plot predicted actor trajectories, GPT path, and bounding boxes"""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot actor trajectories
    for actor_id, trajectory in actor_trajectories.items():
        if not trajectory:
            continue
            
        x_vals = [p[0] for p in trajectory]
        y_vals = [p[1] for p in trajectory]
        times = [p[3] for p in trajectory]
        
        # Plot trajectory line
        line, = plt.plot(x_vals, y_vals, '--', alpha=0.7, label=f"Actor {actor_id} Trajectory")
        color = line.get_color()
        
        # Plot points with time annotations
        for i, (x, y, _, t) in enumerate(trajectory):
            if i % 2 == 0:  # Annotate every few points to avoid clutter
                plt.scatter(x, y, color=color, s=30)
                plt.annotate(f"{t:.1f}s", (x, y), textcoords="offset points", xytext=(5,5), ha='center')
        
        # Plot bounding boxes at start and end
        if actor_id in actors_info:
            bbox_length, bbox_width = actors_info[actor_id]['bbox']
            # Start position bbox
            start_x, start_y = trajectory[0][0], trajectory[0][1]
            start_ori = trajectory[0][2]
            rect = patches.Rectangle((start_x - bbox_length/2, start_y - bbox_width/2), 
                                   bbox_length, bbox_width,
                                   angle=np.degrees(start_ori), rotation_point='center',
                                   linewidth=1, edgecolor=color, facecolor='none', linestyle='--')
            ax.add_patch(rect)
            
            # End position bbox
            end_x, end_y = trajectory[-1][0], trajectory[-1][1]
            end_ori = trajectory[-1][2]
            rect = patches.Rectangle((end_x - bbox_length/2, end_y - bbox_width/2), 
                                   bbox_length, bbox_width,
                                   angle=np.degrees(end_ori), rotation_point='center',
                                   linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
    
    # Plot GPT path
    if gpt_path_points:
        gpt_x = [p[0] for p in gpt_path_points]
        gpt_y = [p[1] for p in gpt_path_points]
        gpt_times = [p[2] for p in gpt_path_points]
        
        plt.plot(gpt_x, gpt_y, 'b-', linewidth=2, label="GPT Generated Path")
        
        # Annotate points with time
        for i, (x, y, t) in enumerate(gpt_path_points):
            if i % 2 == 0:  # Annotate every few points
                plt.scatter(x, y, color='blue', s=50)
                plt.annotate(f"{t:.1f}s", (x, y), textcoords="offset points", xytext=(5,5), ha='center')
    
    # Plot robot initial position and bounding box
    if robot_info:
        robot_x, robot_y = robot_info['position']
        robot_ori = robot_info['orientation']
        bbox_length, bbox_width = robot_info['bbox']
        
        plt.scatter(robot_x, robot_y, color='green', s=100, label="Locobot Start")
        
        # Add bounding box
        rect = patches.Rectangle((robot_x - bbox_length/2, robot_y - bbox_width/2), 
                               bbox_length, bbox_width,
                               angle=np.degrees(robot_ori), rotation_point='center',
                               linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Robot and Actor Trajectories with Bounding Boxes")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rospy.init_node('rosbag_reader', anonymous=True)
    bag_path = "/home/andrea/ros_packages_aggiuntivi/src/llm_generation_trajectory/bag_files/dynamic_obstacles/2_obs/dynamic_obstacles_1.bag"
    read_rosbag(bag_path)

