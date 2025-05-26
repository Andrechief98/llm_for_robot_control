#!/usr/bin/env python3

from llm_generation_trajectory.srv import gptCall, gptCallResponse
import rospy
from openai import OpenAI 
from gazebo_msgs.msg import ModelStates
from gazebo_plugins.msg import ModelBoundingBoxes  
from gazebo_plugins.msg import ActorVelocities
from geometry_msgs.msg import Point, Pose, PoseArray, Twist
from std_srvs.srv import Empty
from std_msgs.msg import String
import json
import random
import math
import time


class Server():

    def __init__(self):

        # Initialize the OpenAI client
        self.client = OpenAI()

        # Subscriber to receive gazebo models positions
        self.pos_subscriber = rospy.Subscriber("/gazebo/model_states", ModelStates, self.storePosDataCallback, queue_size=1)

        # Subscriber to receive gazebo bounding boxes 
        self.pos_subscriber = rospy.Subscriber("/gazebo/bounding_boxes", ModelBoundingBoxes, self.storeBoundingBoxesDataCallback, queue_size=1)

        # Creation of the service to call GPT and provide the generate trajectory
        self.service = rospy.Service('gptCall', gptCall, self.handle_request)

        # Obtain the parameter to understand if the simulation is static or dynamic
        self.simulation_type = rospy.get_param("simulation_type")
        obs_number = rospy.get_param("obs_number")

        # To publish the generated trajectory and allow the rosbag record
        self.path_publisher = rospy.Publisher("/gptGeneratedPath", PoseArray, queue_size=10)

        # To publish the time spent by the LLM to generate the trajectory and allow the rosbag record
        self.time_publisher = rospy.Publisher("/callDuration", String, queue_size=10)

        self.actors_pubs = []


        # If the simulation is dynamic, we set random but costant velocities for each actor and then we publish them after the LLM call
        if self.simulation_type == "dynamic":

            for i in range(1, obs_number + 1):
                pub = rospy.Publisher(f"/actor{i}/cmd_vel", Twist, queue_size=10)
                self.actors_pubs.append(pub)
                

        # Clients to pause and unpause Gazebo during the call to LLM
        # self.pause_gazebo_client = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # self.unpause_gazebo_client = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)


        rospy.init_node('gptCall_server')


    def storePosDataCallback(self, gazeboModelStates_msg):
        self.objects_positions = gazeboModelStates_msg

        self.objects_positions.name.pop(0)
        self.objects_positions.pose.pop(0)

    def storeBoundingBoxesDataCallback(self, boundingBoxes_msg):
        self.objects_boundingBoxes = boundingBoxes_msg



    def callGPT(self, prompt, system_prompt, model):
        
        if model == "gpt-4o":

            response = self.client.chat.completions.create(
                model = model,
                temperature = 0.0,
                messages=[
                    {
                        "role": "assistant", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "robot_path",
                        "description": "Sequence of generated trajectory points",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "points": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "A string containing the tuple (x,y) coordinates of the single point of the generated trajectory"
                                    }
                                },
                            },
                            "required": ["points"]
                        }
                    }
                }

            )
        elif model == "o3-mini":
            response = self.client.chat.completions.create(
                model = model,
                messages=[
                    {
                        "role": "assistant", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
            )

        elif model == "DeepSeek V3":
            self.client.base_url="https://openrouter.ai/api/v1"
            self.client.api_key="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"

            response = self.client.chat.completions.create(
                model = "deepseek/deepseek-chat-v3-0324:free",
                messages=[
                    {
                        "role": "assistant", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
            )

        else:
            rospy.logerr("Model not valid")
        try:
            print("Calling LLM ......")
            return response.choices[0].message.content
        except Exception as error:
            rospy.logerr(f"Error in the LLM call {error}")
            
    
    def handle_request(self, req):
        prompt_type = req.prompt_type
        model = req.model
        few_shot = req.few_shot

        if prompt_type == "lemniscata":
            system_prompt = "You are an expert of path planning. Your task is to generate and return a sequence of points representing the trajectory requested by the user"
            
            prompt = """
                ** TASK: **
            Generate a sequence of 30 points that form a complete horizontal Gerono lemniscate trajectory. Then generate a vertical Bernoulli Lemniscata the crosses the Gerono one

            ** CONSTRAINTS: **
            - the **center** of the both lemniscatas must be in the point (x,y) = (0,0). 
            - The points should be evenly distributed along the trajectory to ensure smooth motion.
            - Don't provide any explanation



                ** OUTPUT FORMAT: **
                You must return a **valid JSON object** that conforms to json.loads() function in Python. The object should contain::
                    "points" :  A list of 30 coordinate pairs [x, y], where each pair represents a point on the trajectory.

                ** EXAMPLE OF OUTPUT FORMAT: **
                {
                    "points" :  [
                        [8.5355, 5.0000],
                        [8.3150, 5.6880],
                        [7.7720, 6.1260],
                        [7.1250, 6.2480],
                        [6.5250, 6.1330],
                        [6.0116, 5.8754],
                        [5.5730, 5.5450],
                        [5.1855, 5.1845],
                        [4.8145, 4.8155],
                        [4.4271, 4.4550],
                        [3.9884, 4.1246],
                        [3.4750, 3.8670],
                        [2.8750, 3.7520],
                        [2.2280, 3.8740],
                        [1.6850, 4.3120],
                        [1.4645, 5.0000],
                        [1.6850, 5.6880],
                        [2.2280, 6.1260],
                        [2.8750, 6.2480],
                        [3.4750, 6.1330],
                        [3.9884, 5.8754],
                        [4.4271, 5.5450],
                        [4.8145, 5.1845],
                        [5.1855, 4.8155],
                        [5.5730, 4.4550],
                        [6.0116, 4.1246],
                        [6.5250, 3.8670],
                        [7.1250, 3.7520],
                        [7.7720, 3.8740],
                        [8.3150, 4.3120]
                    ]
                }
                
                Ensure that the points are computed accurately and scaled correctly."""
            
            if few_shot:
                task_example_lemniscata = """
                ** EXAMPLE 1**
                USER: Generate a list of (x, y) points sampled along a complete Bernoulli lemniscate, using 21 points.
                        ** CONSTRAINTS: **
                                    - The lemniscate must be centered in the origin (0,0);
                                    - The points should be evenly distributed along the trajectory to ensure smooth motion.
                                    - Don't provide any explanation
                {
                    "points" :  [
                        [1.414, 0.0], 
                        [1.227, 0.379], 
                        [0.851, 0.5], 
                        [0.502, 0.406], 
                        [0.229, 0.218],
                        [0.0, 0.0], 
                        [-0.229, -0.218], 
                        [-0.502, -0.407], 
                        [-0.851, -0.5], 
                        [-1.227, -0.378],
                        [-1.414, 0.0], 
                        [-1.227, 0.378], 
                        [-0.851, 0.5], 
                        [-0.502, 0.406], 
                        [-0.229, 0.218],
                        [0.0, 0.0], 
                        [0.229, -0.218], 
                        [0.502, -0.406], 
                        [0.851, -0.5], 
                        [1.227, -0.379], 
                        [1.414, 0.0]
                    ]
                }

                ** EXAMPLE 2**
                USER: Generate a list of (x, y) points sampled along a complete Bernoulli lemniscate, using 21 points.
                        CONSTRAINTS:
                            - The lemniscate must be centered at (1, 1);
                            - The points should be evenly distributed along the trajectory to ensure smooth motion;
                            - Apply an offset of +1 to all x and y coordinates.
                            - Don't provide any explanation.
                {
                    "points" :  [
                        [2.414, 1.0], 
                        [2.227, 1.379], 
                        [1.851, 1.5], 
                        [1.502, 1.406], 
                        [1.229, 1.218],
                        [1.0, 1.0], 
                        [0.771, 0.782], 
                        [0.498, 0.593], 
                        [0.149, 0.5], 
                        [-0.227, 0.622],
                        [-0.414, 1.0], 
                        [-0.227, 1.378], 
                        [0.149, 1.5], 
                        [0.498, 1.406], 
                        [0.771, 1.218],
                        [1.0, 1.0], 
                        [1.229, 0.782], 
                        [1.502, 0.594], 
                        [1.851, 0.5], 
                        [2.227, 0.621], 
                        [2.414, 1.0]
                    ]
                }
                """
                prompt = prompt + task_example_lemniscata
    
        elif prompt_type == "obstacles":

            # Definition of the final posion to reach
            prompt = """{"final_position" : {"x": 10.0, "y": 10.0}}\n"""

            if self.simulation_type == "static":

                # Definition of the system prompt
                system_prompt = """
                You are an expert in robotic path planning. 
                The user will provide you the positions, velocity and corresponding boundix box of different obstacles in the environment. 

                ** TASK **    
                You have to generate a free obstacle path trajectory to reach a user-specified position taking in consideration:
                - the final position to reach;
                - the initial position of the robot and its bounding box
                - the positions of all the obstacles and their corresponding bounding boxes

                    
                ** ADDITIONAL INSTRUCTIONS **

                - Robot name:
                    the robot is indicated as "locobot"

                - Clearance Consideration:
                    The generated trajectory must account for the robot’s own bounding box (0.5 × 0.5) by adding an appropriate clearance to the obstacles. For example, if using a center-based approach, expand each obstacle’s bounding box by half the robot’s width and height.

                - Dynamic Obstacles (if any):
                    If obstacles have non-zero velocities, consider predicting their positions over time and generating a time-parameterized trajectory that avoids collisions at any time along the path.

                - Trajectory Requirements:
                    Provide a high-resolution trajectory with at least 30 waypoints
                    Ensure the trajectory is smooth, obstacle-free, and safe considering both static and dynamic constraints. 
                    

                ** OUTPUT FORMAT: **
                    You must return a **valid JSON object** that conforms to json.loads() function in Python. The object should contain:
                        "points" :  A list of coordinate pairs [x, y], where each pair represents a point on the trajectory.

                    ** EXAMPLE OF OUTPUT FORMAT: **
                    {
                        "points": [
                            [0.0000, 0.0000],
                            [0.3448, 0.6112],
                            [0.6897, 1.2034],
                            [1.0345, 1.7753],
                            [1.3793, 2.3306],
                            [1.7241, 2.8655],
                            [2.06897, 3.3814],
                            [2.4138, 3.8786],
                            [2.7586, 4.3566],
                            [3.1034, 4.8156],
                            [3.4483, 5.2553],
                            [3.7931, 5.6769],
                            [4.1379, 6.0741],
                            [4.4828, 6.4603],
                            [4.8276, 6.8249],
                            [5.1724, 7.1706],
                            [5.5172, 7.4970],
                            [5.8621, 7.8025],
                            [6.2069, 8.0897],
                            [6.5517, 8.3603],
                            [6.8966, 8.6187],
                            [7.2414, 8.8356],
                            [7.5862, 9.0415],
                            [7.9310, 9.2345],
                            [8.2759, 9.4161],
                            [8.6207, 9.5783],
                            [8.9655, 9.7090],
                            [9.3103, 9.8325],
                            [9.6552, 9.9205],
                            [10.0000, 10.0000]
                        ]
                    }"""
                
                if few_shot:
                    # we provide also an example of a correct output 

                    task_example = """
                    ** TASK EXAMPLE: **

                        "prompt" : {
                            {"final_position" : {"x": 10.0, "y": 10.0}}
                            {"Type": "obstacle_0", "Position": {"x": 9.18, "y": 2.47}, "Velocity": {"Vx": 0.0, "Vy": 0.0}, "Bounding_box": {"x_min": 8.19, "y_min": 1.68, "x_max": 10.17, "y_max": 3.27}}
                            {"Type": "obstacle_1", "Position": {"x": 0.06, "y": 8.66}, "Velocity": {"Vx": 0.0, "Vy": 0.0}, "Bounding_box": {"x_min": -0.64, "y_min": 7.95, "x_max": 0.76, "y_max": 9.37}}
                            {"Type": "obstacle_2", "Position": {"x"the0.42, "y": 5.41}, "Velocity": {"Vx": 0.0, "Vy": 0.0}, "Bounding_box": {"x_min": -0.15, "y_min": 4.65, "x_max": 0.99, "y_max": 6.17}}
                            {"Type": "obstacle_3", "Position": {"x": 2.01, "y": 7.86}, "Velocity": {"Vx": 0.0, "Vy": 0.0}, "Bounding_box": {"x_min": 1.48, "y_min": 7.32, "x_max": 2.54, "y_max": 8.4}}
                            {"Type": "obstacle_4", "Position": {"x": 6.21, "y": 9.46}, "Velocity": {"Vx": 0.0, "Vy": 0.0}, "Bounding_box": {"x_min": 5.72, "y_min": 8.57, "x_max": 6.7, "y_max": 10.35}}
                            {"Type": "obstacle_5", "Position": {"x": 4.02, "y": 3.71}, "Velocity": {"Vx": 0.0, "Vy": 0.0}, "Bounding_box": {"x_min": 3.03, "y_min": 2.92, "x_max": 5.0, "y_max": 4.5}}
                            {"Type": "locobot", "Position": {"x": 0.01, "y": 0.03}, "Velocity": {"Vx": 0.0, "Vy": 0.0}, "Bounding_box": {"x_min": -0.18, "y_min": -0.18, "x_max": 0.23, "y_max": 0.23}}
                        }

                        "output" : {
                            "points": [
                                [0.0100, 0.0300],
                                [1.0000, 0.1000],
                                [2.0000, 0.2000],
                                [3.0000, 0.4000],
                                [4.0000, 0.7000],
                                [5.0000, 1.0000],
                                [6.0000, 1.2000],
                                [6.8000, 1.4000],
                                [7.4000, 1.5500],
                                [7.8000, 1.7000],
                                [7.8000, 2.0000],
                                [7.8000, 2.3000],
                                [7.8000, 2.6000],
                                [7.6000, 2.9000],
                                [7.2000, 3.2000],
                                [6.5000, 3.8000],
                                [5.8000, 4.4000],
                                [5.0000, 5.0000],
                                [4.2000, 5.6000],
                                [3.8000, 6.2000],
                                [3.5000, 6.8000],
                                [3.8000, 7.4000],
                                [4.5000, 7.7000],
                                [5.5000, 7.7000],
                                [6.5000, 7.7000],
                                [7.5000, 8.2000],
                                [8.3000, 8.6000],
                                [9.2000, 9.1000],
                                [9.7000, 9.5500],
                                [10.0000, 10.0000]
                            ]
                        }
                    """
                    system_prompt = system_prompt + task_example


                # Extraction of obstacles positions and bounding boxes
                for idx in range(len(self.objects_positions.name)):

                    dictionary = {
                        "Type" : self.objects_positions.name[idx],
                        "Position" : {
                            "x" : round(self.objects_positions.pose[idx].position.x,2),
                            "y" : round(self.objects_positions.pose[idx].position.y,2)
                        },
                        "Velocity" : {
                            "Vx" : round(self.objects_positions.twist[idx].linear.x,2),
                            "Vy" : round(self.objects_positions.twist[idx].linear.y,2)
                        },
                        "Bounding_box" : {
                            "x_min" : round(self.objects_boundingBoxes.min[idx].x,2),
                            "y_min" : round(self.objects_boundingBoxes.min[idx].y,2),
                            "x_max" : round(self.objects_boundingBoxes.max[idx].x,2),
                            "y_max" : round(self.objects_boundingBoxes.max[idx].y,2)
                        }
                    }

                    prompt = prompt + json.dumps(dictionary) + "\n"

            elif self.simulation_type == "dynamic":
                
                # Definition of the system prompt
                system_prompt = """
                You are an expert in robotic path planning.
                The user will provide you the positions, velocity, and corresponding bounding box of different obstacles in the environment.

                ** TASK **
                You have to generate a free obstacle path trajectory to reach a user-specified position, taking into consideration:
                - The final position to reach.
                - The initial position of the robot, its maximum velocity, and its bounding box.
                - The positions of all the obstacles, their corresponding velocities, and bounding boxes.

                ** ADDITIONAL INSTRUCTIONS **
                *Robot name:*
                - The robot is indicated as "locobot".

                *Robot maximum velocity:*
                - The robot is characterized by a maximum velocity of 0.5 m/s (maximum norm between Vx and Vy).

                *Clearance Consideration:*
                - The generated trajectory must account for the robot’s own bounding box (0.5 × 0.5) by adding an appropriate clearance to the obstacles.
                - For example, if using a center-based approach, expand each obstacle’s bounding box by half the robot’s width and height.

                *Dynamic Obstacles:*
                - Consider predicting their next positions over time and generating a time-parameterized trajectory that avoids collisions at any time along the path.
                - You can consider the provided obstacles' velocities as constant during the entire path.

                *Trajectory Requirements:*
                - Provide a high-resolution trajectory considering a timestep of 0.5 s.
                - Ensure the trajectory is smooth, obstacle-free, and safe considering both static and dynamic constraints.

                * OUTPUT FORMAT:*
                -You must return a valid JSON object that conforms to json.loads() function in Python. The object should contain:
                    "points": A list of coordinate pairs [x, y], where each pair represents a point on the trajectory corresponding to a given and ordered timestep.

                    ** EXAMPLE OF OUTPUT FORMAT: **
                    {
                        "points": [
                                [0.5, 0.0],
                                [1.0000, 0.0],
                                [1.5000, 0.0],
                                [2.0000, 0.0],
                                [2.5000, 0.0],
                                [3.0000, 0.0],
                                [2.5000, 0.0],
                                [3.0000, 0.0],
                                [3.5000, 0.0],
                                [4.0000, 0.0],
                                [4.5000, 0.0],
                                [5.0000, 0.0],
                                [5.5000, 0.0],
                                [6.0,0.0],
                                [6.19,0.48],
                                [6.38,0.95],
                                [6.57,1.43],
                                [6.76,1.90],
                                [6.95,2.38],
                                [7.14,2.86],
                                [7.33,3.33],
                                [7.52,3.81],
                                [7.71,4.29],
                                [7.90,4.76],
                                [8.10,5.24],
                                [8.29,5.71],
                                [8.48,6.19],
                                [8.67,6.67],
                                [8.86,7.14],
                                [9.05,7.62],
                                [9.24,8.10],
                                [9.43,8.57],
                                [9.62,9.05],
                                [9.81,9.52],
                                [10.0,10.0]
                            ]
                    }"""
                
                if few_shot:
                    # we provide also an example of a correct output 

                    task_example = """
                    ** TASK EXAMPLE: **

                        "prompt" : {
                            {"final_position" : {"x": 10.0, "y": 10.0}}
                            {"Type": "actor1", "Position": {"x": 4.91, "y": 8.92}, "Velocity": {"Vx": -0.16, "Vy": -0.19}, "Bounding_box": {"x_min": 4.07, "y_min": 8.05, "x_max": 5.75, "y_max": 9.79}}
                            {"Type": "actor2", "Position": {"x": 7.86, "y": 8.46}, "Velocity": {"Vx": -0.34, "Vy": -0.44}, "Bounding_box": {"x_min": 7.12, "y_min": 7.64, "x_max": 8.6, "y_max": 9.28}}
                            {"Type": "locobot", "Position": {"x": 0.0, "y": 0.0}, "Velocity": {"Vx": 0.0, "Vy": 0.0}, "Bounding_box": {"x_min": -0.16, "y_min": -0.17, "x_max": 0.19, "y_max": 0.18}}
                        }

                        "output" : {
                            "points": [
                                [0.5, 0.0],
                                [1.0000, 0.0],
                                [1.5000, 0.0],
                                [2.0000, 0.0],
                                [2.5000, 0.0],
                                [3.0000, 0.0],
                                [2.5000, 0.0],
                                [3.0000, 0.0],
                                [3.5000, 0.0],
                                [4.0000, 0.0],
                                [4.5000, 0.0],
                                [5.0000, 0.0],
                                [5.5000, 0.0],
                                [6.0,0.0],
                                [6.19,0.48],
                                [6.38,0.95],
                                [6.57,1.43],
                                [6.76,1.90],
                                [6.95,2.38],
                                [7.14,2.86],
                                [7.33,3.33],
                                [7.52,3.81],
                                [7.71,4.29],
                                [7.90,4.76],
                                [8.10,5.24],
                                [8.29,5.71],
                                [8.48,6.19],
                                [8.67,6.67],
                                [8.86,7.14],
                                [9.05,7.62],
                                [9.24,8.10],
                                [9.43,8.57],
                                [9.62,9.05],
                                [9.81,9.52],
                                [10.0,10.0]
                            ]
                        }
                    """
                    system_prompt = system_prompt + task_example
                
                # Extraction of obstacles positions, obstacles velocities and bounding boxes
                velocities_list = []

                for idx in range(len(self.objects_positions.name)):

                    if self.objects_positions.name[idx] == "locobot":
                        dictionary = {
                            "Type" : self.objects_positions.name[idx],
                            "Position" : {
                                "x" : round(self.objects_positions.pose[idx].position.x,2),
                                "y" : round(self.objects_positions.pose[idx].position.y,2)
                            },
                            "Velocity" : {
                                "Vx" : round(self.objects_positions.twist[idx].linear.x,2),
                                "Vy" : round(self.objects_positions.twist[idx].linear.y,2)
                            },
                            "Bounding_box" : {
                                "x_min" : round(self.objects_boundingBoxes.min[idx].x,2),
                                "y_min" : round(self.objects_boundingBoxes.min[idx].y,2),
                                "x_max" : round(self.objects_boundingBoxes.max[idx].x,2),
                                "y_max" : round(self.objects_boundingBoxes.max[idx].y,2)
                            }
                        }

                    else:

                        orientation = round(self.objects_positions.pose[idx].orientation.z,2)-1.57

                        velocity = round(random.uniform(0.2, 1),2)

                        vel_dict = {
                            "Vx" : round(velocity*math.cos(orientation),2), 
                            "Vy" : round(velocity*math.sin(orientation),2),
                        }

                        velocities_list.append(velocity)

                        dictionary = {
                            "Type" : self.objects_positions.name[idx],
                            "Position" : {
                                "x" : round(self.objects_positions.pose[idx].position.x,2),
                                "y" : round(self.objects_positions.pose[idx].position.y,2)
                            },
                            "Velocity" : vel_dict,
                            "Bounding_box" : {
                                "x_min" : round(self.objects_boundingBoxes.min[idx].x,2),
                                "y_min" : round(self.objects_boundingBoxes.min[idx].y,2),
                                "x_max" : round(self.objects_boundingBoxes.max[idx].x,2),
                                "y_max" : round(self.objects_boundingBoxes.max[idx].y,2)
                            }
                        }

                    prompt = prompt + json.dumps(dictionary) + "\n"
                
            else:
                return rospy.logerr("Parameter 'simulation_type' not valid")
            
        else:
            return rospy.logerr("'prompt_type' not valid") 
        
        
        # self.pause_gazebo_client()
        print(system_prompt)
        print(prompt)
        start_time = time.time()
        try:
            response = self.callGPT(prompt, system_prompt, model)
            end_time = time.time()
        except Exception as error:
            print(error)

        elapsed_time = end_time-start_time
        elapsed_time_msg = String(str(elapsed_time))

        # Example of response from Lemniscata
        # response = """
        #     {
        #         "points" : [
        #             [4.5, 3.0], 
        #             [4.4672, 3.305], 
        #             [4.3703, 3.5571], 
        #             [4.2135, 3.7133], 
        #             [4.0036, 3.7457], 
        #             [3.75, 3.6495], 
        #             [3.4635, 3.4409], 
        #             [3.1568, 3.1559], 
        #             [2.8432, 2.8441], 
        #             [2.5365, 2.5591], 
        #             [2.25, 2.3505], 
        #             [1.9964, 2.2543], 
        #             [1.7865, 2.2867], 
        #             [1.6297, 2.4429], 
        #             [1.5328, 2.695], 
        #             [1.5, 3.0], 
        #             [1.5328, 3.305], 
        #             [1.6297, 3.5571], 
        #             [1.7865, 3.7133], 
        #             [1.9964, 3.7457], 
        #             [2.25, 3.6495], 
        #             [2.5365, 3.4409], 
        #             [2.8432, 3.1559], 
        #             [3.1568, 2.8441], 
        #             [3.4635, 2.5591], 
        #             [3.75, 2.3505], 
        #             [4.0036, 2.2543], 
        #             [4.2135, 2.2867], 
        #             [4.3703, 2.4429], 
        #             [4.4672, 2.695]
        #         ]
        #     }
        # """

        # Example of response from "obstacles"
        # response = """
        # {
        #     "points": [
        #         [0.01, -0.06],
        #         [0.808, -0.06],
        #         [1.606, -0.06],
        #         [2.404, -0.06],
        #         [3.202, -0.06],
        #         [4.0, -0.06],
        #         [4.4286, 0.2343],
        #         [4.8572, 0.5286],
        #         [5.2858, 0.8229],
        #         [5.7144, 1.1172],
        #         [6.1430, 1.4115],
        #         [6.5716, 1.7058],
        #         [7.0, 2.0],
        #         [7.1429, 2.4286],
        #         [7.2857, 2.8571],
        #         [7.4286, 3.2857],
        #         [7.5714, 3.7143],
        #         [7.7143, 4.1429],
        #         [7.8571, 4.5714],
        #         [8.0, 5.0],
        #         [8.05, 5.4],
        #         [8.1, 5.8],
        #         [8.15, 6.2],
        #         [8.2, 6.6],
        #         [8.25, 7.0],
        #         [8.3, 7.4],
        #         [8.35, 7.8],
        #         [8.4, 8.2],
        #         [8.45, 8.6],
        #         [8.5, 9.0]
        #     ]
        # }
        # """
        # print(f"{response}")

        response = response[response.find("{"):response.rfind("}")+1]
        print(response)
        print(type(response))

        try:
            dictionary = json.loads(response)
            # print(dictionary["points"])
            trajectory = dictionary["points"]
            poseArray = PoseArray()
            poseArray.header.stamp = rospy.Time.now()
            poseArray.header.frame_id = "world"


            
            for single_point in trajectory:
                pose = Pose()
                pose.position = Point(x=single_point[0], y=single_point[1], z=0)  
                # pose.orientation =   
                poseArray.poses.append(pose)


            # self.unpause_gazebo_client()

            
            if prompt_type == "obstacles":
                # Publish msgs to fill the topic and enable the rosbag record
                while True:
                    if self.simulation_type == "dynamic":
                        # Publish of the velocities messages of all the actors:
                        for actor_pub, vel in zip(self.actors_pubs, velocities_list):
                            # print(vel)
                            # print(actor_pub)
                            twist_msg = Twist()

                            twist_msg.linear.x = vel

                            actor_pub.publish(twist_msg)
                    
                    self.path_publisher.publish(poseArray)
                    self.time_publisher.publish(elapsed_time_msg)
                    
                    rospy.Rate(5).sleep()
                    print("Publishing")

            elif prompt_type == "lemniscata":
                rospy.logerr(f"INFERENCE TIME {elapsed_time}")
                return gptCallResponse(poseArray)
        except Exception as error:
            rospy.logerr(f"Error in the json transformation: {error}")

        
            
        

    

def main():
    serverNode = Server()
    print("Ready")
    rospy.spin()


if __name__ == "__main__":
    main()