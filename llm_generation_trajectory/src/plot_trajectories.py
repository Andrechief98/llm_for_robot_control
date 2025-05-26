import matplotlib.pyplot as plt
import numpy as np

def plot_environment(obstacles, robot_trajectory=None, robot_size=(0.5, 0.5), 
                     time_step=1.0, annotate_every=2.0):
    """
    Plotta l'ambiente con ostacoli e traiettoria del robot.
    
    annotate_every: intervallo di tempo in secondi per mostrare solo alcuni punti.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    
    if robot_trajectory is not None:
        robot_trajectory = np.array(robot_trajectory)
        total_steps = len(robot_trajectory)
        times = np.arange(total_steps) * time_step
        
        # Selezioniamo solo i punti con un timestep multiplo di annotate_every
        selected_indices = np.where(times % annotate_every == 0)[0]
        
        # Plotto la traiettoria come linea con punti selezionati
        ax.plot(robot_trajectory[selected_indices, 0], 
                robot_trajectory[selected_indices, 1], '-o', color='blue', label='Robot')
        
        # Plotto i bounding box solo per i punti selezionati
        for i in selected_indices:
            rx, ry = robot_trajectory[i]
            robot_rect = plt.Rectangle((rx - robot_size[0] / 2, ry - robot_size[1] / 2), 
                                       robot_size[0], robot_size[1], color='green', alpha=0.2)
            ax.add_patch(robot_rect)
            ax.text(rx, ry, f"t={times[i]:.1f}", fontsize=8, color='black')
    
    total_time = times[-1] if robot_trajectory is not None else 10
    num_steps = int(total_time / time_step) + 1
    t_array = np.linspace(0, total_time, num_steps)
    
    for idx, obs in enumerate(obstacles):
        if len(obs) == 2:
            init_pos, size = obs
            velocity = (0.0, 0.0)
        elif len(obs) == 3:
            init_pos, size, velocity = obs
        else:
            raise ValueError("Formato ostacolo non valido")
        
        traj = np.array([(init_pos[0] + velocity[0] * t, init_pos[1] + velocity[1] * t) for t in t_array])
        selected_indices = np.where(t_array % annotate_every == 0)[0]
        
        ax.plot(traj[selected_indices, 0], traj[selected_indices, 1], '-x', linestyle='--', color='red',
                label='Ostacolo' if idx == 0 else "")
        
        for i in selected_indices:
            ox, oy = traj[i]
            obs_rect = plt.Rectangle((ox - size[0] / 2, oy - size[1] / 2), size[0], size[1], 
                                     color='red', alpha=0.15)
            ax.add_patch(obs_rect)
            ax.text(ox, oy, f"t={t_array[i]:.1f}", fontsize=8, color='black')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    plt.show()

# Esempio di ostacoli
test_obstacles = [
    ((4.91, 8.92), (1.68, 1.74), (-0.16, -0.19)),
    ((7.86, 8.46), (1.48, 1.64), (-0.34, -0.44))
]

# Esempio di traiettoria
test_trajectory = trajectory = [

    [0.0000, 0.0000],
        [0.1768, 0.1768],
        [0.3536, 0.3536],
        [0.5303, 0.5303],
        [0.7071, 0.7071],
        [0.8839, 0.8839],
        [1.0607, 1.0607],
        [1.2375, 1.2375],
        [1.4142, 1.4142],
        [1.5918, 1.5918],
        [1.7684, 1.7684],
        [1.9451, 1.9451],
        [2.1217, 2.1217],
        [2.2983, 2.2983],
        [2.4749, 2.4749],
        [2.6516, 2.6516],
        [2.8284, 2.8284],
        [3.0050, 3.0050],
        [3.1818, 3.1818],
        [3.3584, 3.3584],
        [3.5355, 3.5355],
        [3.7121, 3.7121],
        [3.8887, 3.8887],
        [4.0653, 4.0653],
        [4.2421, 4.2421],
        [4.4188, 4.4188],
        [4.5954, 4.5954],
        [4.7720, 4.7720],
        [4.9488, 4.9488],
        [5.1254, 5.1254],
        [5.3021, 5.3021],
        [5.4787, 5.4787],
        [5.6553, 5.6553],
        [5.8320, 5.8320],
        [6.0086, 6.0086],
        [6.1852, 6.1852],
        [6.3620, 6.3620],
        [6.5386, 6.5386],
        [6.7153, 6.7153],
        [6.8919, 6.8919],
        [7.0686, 7.0686],
        [7.2452, 7.2452],
        [7.4218, 7.4218],
        [7.5985, 7.5985],
        [7.7751, 7.7751],
        [7.9517, 7.9517],
        [8.1284, 8.1284],
        [8.3050, 8.3050],
        [8.4816, 8.4816],
        [8.6582, 8.6582],
        [8.8349, 8.8349],
        [9.0115, 9.0115],
        [9.1881, 9.1881],
        [9.3648, 9.3648],
        [9.5414, 9.5414],
        [9.7180, 9.7180],
        [9.8946, 9.8946],
        [10.0000, 10.0000]
  
]

# Plottiamo con un intervallo di 2 secondi
plot_environment(test_obstacles, robot_trajectory=test_trajectory, time_step=0.5, annotate_every=2.0)
