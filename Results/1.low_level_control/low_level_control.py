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
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'turtle1/cmd_vel', 10)
        
        # Client for the "teleport_absolute" service (to position the turtle)
        self.teleport_client = self.create_client(TeleportAbsolute, 'turtle1/teleport_absolute')

        # Client for the "clear" service (to clear the screen)
        self.clearScreen = self.create_client(Empty, 'clear')


def main(args=None):
    rclpy.init(args=args)
    node = Publisher()

    # 1) Initial teleport to (2, 2) with theta = 0
    teleport_req = TeleportAbsolute.Request()
    teleport_req.x = 2.0
    teleport_req.y = 2.0
    teleport_req.theta = 0.0
    node.teleport_client.call_async(teleport_req)
    time.sleep(2)

    # 2) Clear the screen
    clear_req = Empty.Request()
    node.clearScreen.call_async(clear_req)
    time.sleep(2)

    # 3) Load the JSON file with generated commands
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, "results.json"), "r") as f:
        data = json.load(f)

    # List of models and trials to iterate over
    list_of_models = ["gpt-4o", "DeepSeek V3", "gpt-4o-fewShot", "DeepSeek V3-fewShot", "o3-mini"]
    list_of_models = ["gpt-4o", "o3-mini"]
    list_of_trials = ["trial_1"]

    # DICTIONARY to collect inference times (in seconds) for each model
    durations = {
        "gpt-4o"             : [],
        "DeepSeek V3"        : [],
        "gpt-4o-fewShot"     : [],
        "DeepSeek V3-fewShot": [],
        "o3-mini"            : [],
    }

    # DICTIONARY to collect success flags (1 or 0) for each test for every model
    successes = {
        "gpt-4o"             : [],
        "DeepSeek V3"        : [],
        "gpt-4o-fewShot"     : [],
        "DeepSeek V3-fewShot": [],
        "o3-mini"            : [],
    }


    # Main loop: for each model, for each trial, for each objective_dict inside
    for model in list_of_models:
        print(f"==============================")
        print(f"  Testing model: {model}")
        print(f"==============================")
        for trial in list_of_trials:

            for objective_dict in data[model][trial]:
                durations[model].append(objective_dict["inference_time"])

                print(f" Objective: {objective_dict.get('objective')}")

                # If the response is "error", skip
                if objective_dict.get("response") == "error":
                    print("\tError in response, skipping this test.")
                    # Consider this test failed (0)
                    successes[model].append(0)
                    continue

                # Extract the list of commands
                commands_list = objective_dict["response"].get("commands_list", [])
                if len(commands_list) == 0:
                    print("\tCommand list is empty, skipping.")
                    successes[model].append(0)
                    continue

                for command in commands_list:
                    command_start_time = node.get_clock().now()
                    now = node.get_clock().now()
                    elapsed_time = (now - command_start_time).nanoseconds / 1e9

                    rate = node.create_rate(10)  # 10 Hz
                    try:
                        timeDuration = float(command["time_duration"])

                        while elapsed_time < timeDuration:
                            
                            # Velocity command
                            msg = Twist()
                            msg.linear.x = float(command["linear_velocity"])
                            msg.angular.z = float(command["angular_velocity"])
                            

                            node.cmd_vel_pub.publish(msg)

                            now = node.get_clock().now()
                            elapsed_time = (now - command_start_time).nanoseconds / 1e9
                        

                    except Exception as e:
                        print(e)
                        continue


                # Stop the robot
                msg = Twist()
                msg.linear.x = 0.0
                msg.angular.z = 0.0

                node.cmd_vel_pub.publish(msg)
                time.sleep(2)

                # *** Here we ask the user to indicate success (1) or failure (0) ***
                while True:
                    user_input = input(f"success? (1=success, 0=failure): ")
                    if user_input in ["0", "1"]:
                        successes[model].append(int(user_input))
                        break
                    else:
                        print("\tInvalid input. Type only '1' (success) or '0' (failure).")

                # Teleport the turtle to the initial position for the next objective
                teleport_req = TeleportAbsolute.Request()
                teleport_req.x = 2.0
                teleport_req.y = 2.0
                teleport_req.theta = 0.0
                node.teleport_client.call_async(teleport_req)
                time.sleep(2)

                # Clear the screen
                clear_req = Empty.Request()
                node.clearScreen.call_async(clear_req)
                time.sleep(1)

    # END OF LOOPS: we have collected durations and successes for each model

    print("\n\n=== AGGREGATED RESULTS ===\n")

    for model in list_of_models:
        # Calculate success rate
        total_tests = len(successes[model])
        if total_tests > 0:
            sum_success = sum(successes[model])
            success_rate = sum_success / total_tests
        else:
            success_rate = 0.0

        # Calculate mean and standard deviation of inference times
        if len(durations[model]) >= 1:
            mean_duration = statistics.mean(durations[model])
            # If there's only one sample, stdev will raise an exception: handle it
            try:
                std_duration = statistics.stdev(durations[model])
            except statistics.StatisticsError:
                std_duration = 0.0
        else:
            mean_duration = 0.0
            std_duration = 0.0

        print(f"Model: {model}")
        print(f"\tTotal tests: {total_tests}")
        print(f"\tSuccesses (1): {sum_success}  |  Failures (0): {total_tests - sum_success}")
        print(f"\tSuccess rate: {success_rate*100:.2f}%")
        print(f"\tMean inference time: {mean_duration:.4f} s")
        print(f"\tInference time std. dev.: {std_duration:.4f} s\n")


    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# === AGGREGATED RESULTS ===

# Model: gpt-4o
#         Total tests: 10
#         Successes (1): 0  |  Failures (0): 10
#         Success rate: 0.00%
#         Mean inference time: 3.1886 s
#         Inference time std. dev.: 2.3412 s

# Model: DeepSeek V3
#         Total tests: 10
#         Successes (1): 5  |  Failures (0): 5
#         Success rate: 50.00%
#         Mean inference time: 22.8939 s
#         Inference time std. dev.: 8.3380 s

# Model: gpt-4o-fewShot
#         Total tests: 10
#         Successes (1): 3  |  Failures (0): 7
#         Success rate: 30.00%
#         Mean inference time: 4.2013 s
#         Inference time std. dev.: 1.2522 s

# Model: DeepSeek V3-fewShot
#         Total tests: 10
#         Successes (1): 3  |  Failures (0): 7
#         Success rate: 30.00%
#         Mean inference time: 14.8463 s
#         Inference time std. dev.: 6.9433 s

# Model: o3-mini
#         Total tests: 10
#         Successes (1): 10  |  Failures (0): 0
#         Success rate: 100.00%
#         Mean inference time: 28.9520 s
#         Inference time std. dev.: 19.3789 s