from openai import OpenAI
from pydantic import BaseModel
import json
#from OpenAI_utilities import *
    
from tqdm import tqdm
from time import time
#from LLM_agent import *
import re
import os



task_description = """

**Initial Conditions:**
- Initial position: (x, y) = (2, 2)
- Initial orientation: theta = 0 (along the x-axis)

**Velocity Constraints:**
- Linear velocity: [v_min, v_max] = [0 m/s, 0.5 m/s]
- Angular velocity: [w_min,  w_max] = [-1.5 rad/s, 1.5 rad/s]

**Space Constraints:**
- Coordinates: [x_min, x_max] = [0, 11] and [y_min, y_max] = [0, 11]

**Required Output Format:**
You must provide your answer in JSON format using the following schema:
- **commands_list**: a list of objects, each with the following fields:
  - **position_to_reach**: the final position (x, y) in meters, based on the considered movement
  - **orientation_to_reach**: the final orientation theta in radians with respect the x axis and in the interval [-pi_greco, pi_greco]
  - **linear_velocity**: the robot's linear velocity in m/s along the x-axis of the robot
  - **angular_velocity**: the robot's angular velocity in rad/s around the z-axis of the robot (signed value)
  - **time_duration**: the duration of the movement in seconds
- The returned json must be compliant with the function json.loads() in python. Therefore, only the dictionary must be returned without any additional characters.



**Notes:**
- Ensure that all commands respect the velocity and space constraints.
- Calculate the final position and orientation based on the initial state and the movement command.
- If the task requires multiple movements, the list must contain all commands in the order of execution.

**Example Output:**
{
  "commands_list": [
    {
      "position_to_reach": [2.6, 2],
      "orientation_to_reach": 0.4
      "linear_velocity": 0.3,
      "angular_velocity": 0.2,
      "time_duration": 2
    },
    {
      "position_to_reach": [3.8, 2],
      "orientation_to_reach": 0.1
      "linear_velocity": 0.4,
      "angular_velocity": -0.1,
      "time_duration": 3
    }
  ]
}
"""

few_shot = True

example_fewShot = """
** EXAMPLE OF TASK **

        USER: "Reach point (4.0, 4.0) passing through (3.5, 2.0)",
        "response": {
          "commands_list": [
            {
              "position_to_reach": [3.5, 2.0],
              "orientation_to_reach": 0.0,
              "linear_velocity": 0.5,
              "angular_velocity": 0.0,
              "time_duration": 3.0
            },
            {
              "position_to_reach": [4.0, 4.0],
              "orientation_to_reach": 1.326,
              "linear_velocity": 0.0,
              "angular_velocity": 1,
              "time_duration": 1.326
            },
            {
              "position_to_reach": [4.0, 4.0],
              "orientation_to_reach": 1.326,
              "linear_velocity": 0.5,
              "angular_velocity": 0.0,
              "time_duration": 4.12
            }

          ]
        },
        "response_time": 4.788285970687866
      }
      
      USER: "perform a rectangle trajectory of dimension X = 1 m and Y = 3 m",
        "response": {
          "commands_list": [
            {
              "position_to_reach": [
                5,
                2
              ],
              "orientation_to_reach": 0,
              "linear_velocity": 0.5,
              "angular_velocity": 0,
              "time_duration": 2
            },
            {
              "position_to_reach": [
                5,
                2
              ],
              "orientation_to_reach": 1.5708,
              "linear_velocity": 0,
              "angular_velocity": 1.5,
              "time_duration": 1.0472
            },
            {
              "position_to_reach": [
                5,
                3
              ],
              "orientation_to_reach": 1.5708,
              "linear_velocity": 0.5,
              "angular_velocity": 0,
              "time_duration": 6
            },
            {
              "position_to_reach": [
                5,
                3
              ],
              "orientation_to_reach": 3.1416,
              "linear_velocity": 0,
              "angular_velocity": 1.5,
              "time_duration": 1.0472
            },
            {
              "position_to_reach": [
                2,
                3
              ],
              "orientation_to_reach": 3.1416,
              "linear_velocity": 0.5,
              "angular_velocity": 0,
              "time_duration":2
            },
            {
              "position_to_reach": [
                2,
                3
              ],
              "orientation_to_reach": -1.5708,
              "linear_velocity": 0,
              "angular_velocity": 1.5,
              "time_duration": 1.0472
            },
            {
              "position_to_reach": [
                2,
                2
              ],
              "orientation_to_reach": -1.5708,
              "linear_velocity": 0.5,
              "angular_velocity": 0,
              "time_duration": 6
            },
            {
              "position_to_reach": [
                2,
                2
              ],
              "orientation_to_reach": 0,
              "linear_velocity": 0,
              "angular_velocity": 1.5,
              "time_duration": 1.0472
            }
          ]
        },
        "response_time": 17.102046012878418
      }
    
 """

if few_shot:
    task_description = task_description + example_fewShot 

objectives = [
    "perform a rectangle trajectory of dimension X = 3 m and Y = 1 m",
	"perform a square trajectory of length 2 meters",
	"perform an equilateral triangle trajectory of side 2 meters with base parallel to x axis",
    "perform a circle trajectory of radius 1 meter",
	"reach the position (x,y) = (2.5, 9.0)",
	"go to point (x,y)=(6,7) passing through (x,y)=(3,3)",
	"perform a square trajectory and than a circle trajectory inscribed in the square",
	"perform an horizontal U trajectory",
	"perform a square trajectory with an arch as side",
    "Create 3/4 of a circle and go to point (x,y)= (6,9)"
]



def create_chat_completion(client, model, complete_prompt):
    if model == "gpt-4o":
        response = client.chat.completions.create(
                            model=model,
                            temperature=0.0,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": complete_prompt
                                        },
                                    ],
                                }
                            ],
                            response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "robot_speed_command",
                                "description": "Velocity commands for a differential drive robot",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                    "commands_list": {
                                        "type": "array",
                                        "items": {
                                        "type": "object",
                                        "properties": {
                                            "position_to_reach": {
                                                "type": "array",
                                                "items": { "type": "number" },
                                                "description": "Final position (x,y) to reach in meters"
                                                },
                                            "orientation_to_reach": {
                                                "type": "number",
                                                "description": "Final orientation with respect x axis to reach in radians"
                                                },
                                            "linear_velocity": {
                                                "type": "number",
                                                "description": "Linear velocity in m/s"
                                                },
                                            "angular_velocity": {
                                                "type": "number",
                                                "description": "Angular velocity in rad/s"
                                                },
                                            "time_duration": {
                                                "type": "number",
                                                "description": "Time duration of the movement in seconds"
                                                }
                                        },
                                        "required": [
                                            "position_to_reach",
                                            "orientation_to_reach",
                                            "linear_velocity",
                                            "angular_velocity",
                                            "time_duration"
                                        ]
                                        }
                                    }
                                    },
                                    "required": ["commands_list"]
                                }
                                }
                            }
                        )

    elif model == "gpt-4o-mini":
        response = client.chat.completions.create(
                            model=model,
                            temperature=0.0,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": complete_prompt
                                        },
                                    ],
                                }
                            ],
                            response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "robot_speed_command",
                                "description": "Velocity commands for a differential drive robot",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                    "commands_list": {
                                        "type": "array",
                                        "items": {
                                        "type": "object",
                                        "properties": {
                                            "position_to_reach": {
                                                "type": "array",
                                                "items": { "type": "number" },
                                                "description": "Final position (x,y) to reach in meters"
                                                },
                                            "orientation_to_reach": {
                                                "type": "number",
                                                "description": "Final orientation with respect x axis to reach in radians"
                                                },
                                            "linear_velocity": {
                                                "type": "number",
                                                "description": "Linear velocity in m/s"
                                                },
                                            "angular_velocity": {
                                                "type": "number",
                                                "description": "Angular velocity in rad/s"
                                                },
                                            "time_duration": {
                                                "type": "number",
                                                "description": "Time duration of the movement in seconds"
                                                }
                                        },
                                        "required": [
                                            "position_to_reach",
                                            "orientation_to_reach",
                                            "linear_velocity",
                                            "angular_velocity",
                                            "time_duration"
                                        ]
                                        }
                                    }
                                    },
                                    "required": ["commands_list"]
                                }
                                }
                            }
                        )

    elif model == "DeepSeek V3":
        client.base_url="https://openrouter.ai/api/v1"
        client.api_key="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"
            
        response = client.chat.completions.create(
                            model="deepseek/deepseek-chat-v3-0324:free",
                            temperature=0.0,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": complete_prompt
                                        },
                                    ],
                                }
                            ],
                            response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "robot_speed_command",
                                "description": "Velocity commands for a differential drive robot",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                    "commands_list": {
                                        "type": "array",
                                        "items": {
                                        "type": "object",
                                        "properties": {
                                            "position_to_reach": {
                                                "type": "array",
                                                "items": { "type": "number" },
                                                "description": "Final position (x,y) to reach in meters"
                                                },
                                            "orientation_to_reach": {
                                                "type": "number",
                                                "description": "Final orientation with respect x axis to reach in radians"
                                                },
                                            "linear_velocity": {
                                                "type": "number",
                                                "description": "Linear velocity in m/s"
                                                },
                                            "angular_velocity": {
                                                "type": "number",
                                                "description": "Angular velocity in rad/s"
                                                },
                                            "time_duration": {
                                                "type": "number",
                                                "description": "Time duration of the movement in seconds"
                                                }
                                        },
                                        "required": [
                                            "position_to_reach",
                                            "orientation_to_reach",
                                            "linear_velocity",
                                            "angular_velocity",
                                            "time_duration"
                                        ]
                                        }
                                    }
                                    },
                                    "required": ["commands_list"]
                                }
                                }
                            }
                        )

    elif model == "o1-preview":
        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": complete_prompt
                                        },
                                    ],
                                }
                            ],
                        )

    elif model == "o1-mini":
        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": complete_prompt
                                        },
                                    ],
                                }
                            ],
                        )
    elif model == "o3-mini":
        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": complete_prompt
                                        },
                                    ],
                                }
                            ],
                        )
    else:
        raise ValueError("Model not recognized")
    
    return response


if __name__ == "__main__":

    n_trials = 1

    script_dir = os.path.dirname(__file__)

    #models_to_test = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "o3-mini", DeepSeek V3]
    models_to_test = ["gpt-4o","DeepSeek V3"]

    assistant_instructions = """You are a differential drive robot tasked with translating natural language instructions into low-level movement commands. You will receive various path planning tasks, and you must convert these tasks into a sequence of commands for the robot."""



    initial_prompt = assistant_instructions + "\n" + task_description


    client = OpenAI()

    with open(os.path.join(script_dir, "1_Interaction_results.json"), 'r') as f:
        results = json.load(f)

    #print(results)

    

    tot_time = 0
    for model in models_to_test:
        if few_shot:
            model_name = model + "-fewShot"
        else:
            model_name = model 
        print(model_name)

        results.update(
            {
                model_name : {
                    "trial_1" : [],
                    # "trial_2" : [],
                    # "trial_3" : []
                }
                }
            )

        for trial in range(n_trials):
            trial_number = "trial_" + str(trial+1)
            print(trial_number)
            trial_results = [] 

            for objective in tqdm(objectives): 

                complete_prompt = initial_prompt + "\n" + objective

                init_time = time()

                try:
                    response = create_chat_completion(client, model, complete_prompt)

                    final_response = response.choices[0].message.content
                    


                    end_time = time()
                    tot_time += end_time - init_time
                    
                    final_response.find("{")
                    final_response.rfind("}")

                    final_response = final_response[final_response.find("{"):final_response.rfind("}")+1]

                    
                    # print(f"""\033[1m>> Assistant:\033[0;0m 
                    #         {final_response}""")
                    
                    trial_results.append(
                        {
                            "objective" : objective, 
                            "response" : json.loads(final_response), 
                            "response_time" : end_time - init_time
                            }
                        )
                    
                except Exception as e:
                    print(f"Exception: \n {e}")
                    end_time = time()
                    tot_time += end_time - init_time
                    
                    trial_results.append(
                        {
                            "objective" : objective, 
                            "response" : "error", 
                            "response_time" : end_time - init_time
                            }
                        )
                    
                    continue
                    
            results[model_name][trial_number]=trial_results


    
    # print(results)


    script_dir = os.path.dirname(__file__)

    with open(os.path.join(script_dir, "1_Interaction_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
        
  