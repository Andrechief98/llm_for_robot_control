import rosbag
import rospy
import matplotlib.pyplot as plt
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseArray
import time

def read_rosbag(bag_file):
    """Legge un file rosbag e plotta i bounding boxes e il percorso generato."""
    try:
        bag = rosbag.Bag(bag_file, 'r')
        
        model_states = None
        bounding_boxes = None
        gpt_path = None
        call_duration = None
        
        for topic, msg, t in bag.read_messages():
            if topic == "/gazebo/model_states" and model_states is None:
                model_states = msg
            elif topic == "/gazebo/bounding_boxes" and bounding_boxes is None:
                bounding_boxes = msg
            elif topic == "/gptGeneratedPath" and gpt_path is None:
                gpt_path = msg
            elif topic == "/callDuration" and call_duration is None:
                call_duration = msg.data  # È una stringa, quindi la memorizziamo direttamente
            
            if model_states and bounding_boxes and gpt_path and call_duration:
                break
        
        bag.close()
        
        if model_states and bounding_boxes:
            plot_data(model_states, bounding_boxes, gpt_path, call_duration)
        else:
            rospy.logwarn("Dati di /gazebo/model_states o /gazebo/bounding_boxes non trovati.")
        
        if gpt_path is None:
            rospy.logwarn("Dati di /gptGeneratedPath non trovati nel rosbag.")
        else:
            rospy.loginfo("Dati di /gptGeneratedPath trovati e verranno plottati.")
        
        if call_duration is None:
            rospy.logwarn("Dati di /callDuration non trovati nel rosbag.")
        else:
            rospy.loginfo(f"Call Duration trovata: {call_duration}")
        
    except Exception as e:
        rospy.logerr(f"Errore nella lettura del rosbag: {e}")

def plot_data(model_states, bounding_boxes, gpt_path, call_duration):
    """Plotta i bounding boxes e il percorso generato."""
    plt.figure()
    
    # Plotta i bounding boxes
    for i in range(len(bounding_boxes.name)):
        x_min, y_min = bounding_boxes.min[i].x, bounding_boxes.min[i].y
        x_max, y_max = bounding_boxes.max[i].x, bounding_boxes.max[i].y
        
        rect_x = [x_min, x_max, x_max, x_min, x_min]
        rect_y = [y_min, y_min, y_max, y_max, y_min]
        plt.fill(rect_x, rect_y, 'r', alpha=0.3, edgecolor='r', linewidth=2, label="Bounding Box" if i == 0 else "")
    
    # Plotta il percorso generato
    if gpt_path:
        x_vals = [pose.position.x for pose in gpt_path.poses]
        y_vals = [pose.position.y for pose in gpt_path.poses]
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label="Percorso Generato")
    
    # Aggiunge il valore di callDuration nel plot
    if call_duration is not None:
        plt.text(0.95, 0.05, f"Call Duration: {call_duration}s", transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right', 
                 bbox=dict(facecolor='white', alpha=0.5))
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bounding Boxes e Percorso Generato")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    rospy.init_node('rosbag_reader', anonymous=True)

    for i in range(10):
        print(f"Visualizing {i+1}")
        bag_path = f"/home/andrea/ros_packages_aggiuntivi/src/llm_generation_trajectory/bag_files/static_obstacles/6_obs/static_obstacles_{i+1}.bag"  # Modifica con il percorso corretto
        read_rosbag(bag_path)

