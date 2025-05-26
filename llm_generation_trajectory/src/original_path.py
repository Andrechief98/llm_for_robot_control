import rospy
import matplotlib.pyplot as plt
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped

def create_pose(x, y, frame="map"):
    pose = PoseStamped()
    pose.header.frame_id = frame
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.w = 1.0  # Nessuna rotazione
    return pose

def call_make_plan(start, goal, tolerance=0.5):
    rospy.wait_for_service('/locobot/move_base/make_plan')
    try:
        make_plan = rospy.ServiceProxy('/locobot/move_base/make_plan', GetPlan)
        response = make_plan(start, goal, tolerance)
        return response.plan.poses
    except rospy.ServiceException as e:
        rospy.logerr(f"Errore nella chiamata al servizio make_plan: {e}")
        return []

def plot_path(poses):
    if not poses:
        rospy.logwarn("Nessun percorso ricevuto!")
        return
    
    x_vals = [pose.pose.position.x for pose in poses]
    y_vals = [pose.pose.position.y for pose in poses]
    
    plt.figure()
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Percorso generato da move_base")
    plt.grid()
    plt.show()

def main():
    rospy.init_node('make_plan_client')
    
    start = create_pose(0, 0)
    goal = create_pose(5, 5)
    
    rospy.loginfo("Richiesta di generazione percorso...")
    path = call_make_plan(start, goal)
    
    plot_path(path)

if __name__ == '__main__':
    main()
