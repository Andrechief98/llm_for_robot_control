#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
import os
import time
from geometry_msgs.msg import Twist
from turtlesim.srv import TeleportAbsolute
from std_srvs.srv import Empty
import statistics

class Publisher(Node):
    def __init__(self):
        super().__init__('publisher')
        
        # Publisher per i comandi di velocità
        self.cmd_vel_pub = self.create_publisher(Twist, 'turtle1/cmd_vel', 10)
        
        # Client per il servizio teleport_absolute
        self.teleport_client = self.create_client(TeleportAbsolute, 'turtle1/teleport_absolute')
        self.clearScreen = self.create_client(Empty, 'clear')
        # while not teleport_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info("In attesa del servizio teleport_absolute...")
        

def main(args=None):
    rclpy.init(args=args)
    node = Publisher()

    # Chiamata al servizio teleport_absolute
    teleport_req = TeleportAbsolute.Request()
    teleport_req.x = 2.0
    teleport_req.y = 2.0
    teleport_req.theta = 0.0
    future = node.teleport_client.call_async(teleport_req)
    time.sleep(2)
    clear_req = Empty.Request()
    future = node.clearScreen.call_async(clear_req)
    time.sleep(2)

    script_dir = os.path.dirname(__file__)

    with open(os.path.join(script_dir, "LowLevelCommands/1_Interaction_results.json"), "r") as f:
        data = json.load(f)

    list_of_models = ["gpt-4o", "DeepSeek V3", "gpt-4o-fewShot", "DeepSeek V3-fewShot","o3-mini",]
    list_of_models = ["gpt-4o-fewShot", "DeepSeek V3-fewShot"]
    list_of_trials = ["trial_1"]

    
    durations = {
        "gpt-4o" : [],
        "DeepSeek V3" : [],
        "gpt-4o-fewShot" : [],
        "DeepSeek V3-fewShot" : [],
        "o3-mini" : [],
        "tests" : []
    }

    
    for model in list_of_models:
        for trial in list_of_trials:
            for objective_dict in data[model][trial]:
                durations[model].append(round(objective_dict["response_time"],2))
    
    print("Durations:")
    print(durations)

    deepSeek_fewshot_mean = statistics.mean(durations["DeepSeek V3-fewShot"])
    gpt4_fewshot_mean = statistics.mean(durations["gpt-4o-fewShot"])
    print(f"DeepSeek V3-fewShot mean: {deepSeek_fewshot_mean}")
    print(f"gpt-4o-fewShot mean: {gpt4_fewshot_mean}")
    
    deepSeek_fewshot_std = statistics.stdev(durations["DeepSeek V3-fewShot"])
    gpt4_fewshot_std = statistics.stdev(durations["gpt-4o-fewShot"])
    print(f"DeepSeek V3-fewShot stdev: {deepSeek_fewshot_std}")
    print(f"gpt-4o-fewShot stdev: {gpt4_fewshot_std}")



    for model in list_of_models:
        print(f"Tested model: {model}")
        for trial in list_of_trials:
            for objective_dict in data[model][trial]:
                durations[model].append(objective_dict["response_time"])

                if objective_dict["response"] == "error":
                    continue

                commands_list = objective_dict["response"]["commands_list"]
                print(f"""Objective: 
                      {objective_dict["objective"]}""")

                for command in commands_list:
                    command_start_time = node.get_clock().now()
                    now = node.get_clock().now()
                    elapsed_time = (now - command_start_time).nanoseconds / 1e9

                    rate = node.create_rate(10)  # 10 Hz
                    try:
                        timeDuration = float(command["time_duration"])

                        while elapsed_time < timeDuration:
                            
                            # Creazione del messaggio di velocità
                            msg = Twist()
                            msg.linear.x = float(command["linear_velocity"])
                            msg.angular.z = float(command["angular_velocity"])
                            

                            node.cmd_vel_pub.publish(msg)
                            # node.get_logger().info(f"Published: {msg.linear.x}, {msg.angular.z}")

                            now = node.get_clock().now()
                            elapsed_time = (now - command_start_time).nanoseconds / 1e9
                        

                    except Exception as e:
                        print(e)
                        continue


                # Chiamata al servizio teleport_absolute
                msg = Twist()
                msg.linear.x = 0.0
                msg.angular.z = 0.0

                node.cmd_vel_pub.publish(msg)
                time.sleep(2)

                teleport_req = TeleportAbsolute.Request()
                teleport_req.x = 2.0
                teleport_req.y = 2.0
                teleport_req.theta = 0.0
                
                future = node.teleport_client.call_async(teleport_req)
                time.sleep(2)
                clear_req = Empty.Request()
                future = node.clearScreen.call_async(clear_req)




    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
