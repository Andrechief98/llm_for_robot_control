import matplotlib.pyplot as plt
import json
import statistics
import os

script_dir = os.path.dirname(__file__)

with open(os.path.join(script_dir,"lemniscata_results.json"),"r") as f:
    lemniscata_dict = json.load(f)

models_to_test = ["o3-mini", "gpt-4o", "gpt-4o few-shot", "DeepSeek V3", "DeepSeek V3 few-shot"]

inference_times = {
    "o3-mini":[],
    "gpt-4o":[],
    "gpt-4o few-shot":[],
    "DeepSeek V3":[],
    "DeepSeek V3 few-shot":[],
}



for model in models_to_test:
    print(f"Model: {model}")
    for test in lemniscata_dict[model]["tests"]:
        print(f"\tTest: {test['lemniscata']}")

        # Extracting generated path and corresponding inference time
        points = test["response"]
        inference_time = test["inference_time"]

        # Appending the inference time of the single path in a list (necessary to compute final mean and std_dev)
        inference_times[model].append(inference_time)

        # Separation of x and y coordinates
        x_values, y_values = zip(*points)
        print(f"\tNumber of points: {len(points)}")

        # Plot creation
        plt.figure(figsize=(6, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')

        # Adding labels to the axes
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Lemniscata plot")

        # Setting equal aspect ratio for the axes
        plt.axis("equal")

        # Display the plot
        plt.show()



print("\nInference Time Statistics:")
for model, times in inference_times.items():
    if times:
        mean_time = statistics.mean(times)
        # statistics.stdev richiede almeno due valori; se ce n'è solo uno, fissiamo stdev a 0.0
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  {model}: mean = {mean_time:.4f}s, stdev = {stdev_time:.4f}s")
    else:
        print(f"  {model}: no inference times recorded")

#SUCCESS RATE:
# 'o3-mini': [1,1,1,1,1,1,1,1,1,1] (100%)
# 'gpt-4o': [0,0,0,0,0,0,0,0,0,0] (0%)
# 'gpt-4o few-shot': [1,1,1,0,0,0,0,0,0,0] (30%)
# 'DeepSeek V3': [0,1,0,0,0,0,0,0,0,0] (10%)
# 'DeepSeek V3 few-shot': [1,1,1,0,0,0,0,0,0,0] (30%)

# INFERENCE TIME:
# 'o3-mini': [123, 134, 167, 142, 156, 136, 143, 150, 74, 177], 
# 'gpt-4o': [4.14, 11.7, 3.55, 4.83, 5.03, 8.99, 7.59, 8.24, 5.63, 7.87], 
# 'gpt-4o few-shot': [6.68, 3.475, 5.59, 4.317, 4.78, 5.31, 5.84, 6.49, 4.61, 4.84], 
# 'DeepSeek V3': [9.98, 52.49, 10.14, 11.19, 14.68, 19.34, 12.55, 20.14, 12.63, 9.44], 
# 'DeepSeek V3 few-shot': [11.58, 10.54, 10.13, 9.86, 9.74, 11.48, 12.43, 15.24, 12.0, 16.84]