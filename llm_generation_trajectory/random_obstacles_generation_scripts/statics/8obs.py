import random

def generate_obstacle_positions(num_obstacles, world_size, min_size, max_size):
    obstacles = []
    
    def is_overlapping(new_obstacle):
        nx, ny, nw, nl, _ = new_obstacle  # Ignoriamo l'altezza per il controllo delle collisioni
        for ox, oy, ow, ol, _ in obstacles:
            if (abs(nx - ox) < (nw/2 + ow/2)) and (abs(ny - oy) < (nl/2 + ol/2)):
                return True
        return False
    
    for _ in range(num_obstacles):
        while True:
            w = random.uniform(min_size, max_size)
            l = random.uniform(min_size, max_size)
            h = random.uniform(min_size, max_size)
            x = random.uniform(0, world_size)
            y = random.uniform(1.5, world_size)
            new_obstacle = (x, y, w, l, h)
            
            if not is_overlapping(new_obstacle):
                obstacles.append(new_obstacle)
                break
    
    return obstacles

def create_gazebo_world(obstacles, world_filename="/home/andrea/ros_packages_aggiuntivi/src/llm_generation_trajectory/world/static_world_8obs.world"):
    with open(world_filename, "w") as xmlFile:
        xmlFile.write(
            f"""<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
"""
        )

        for i, (x, y, w, l, h) in enumerate(obstacles):
            # Applichiamo il round a due cifre decimali
            xmlFile.write(
                f"""
    <model name="obstacle_{i}">
      <pose>{round(x,2)} {round(y,2)} {round(h/2,2)} 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{round(w,2)} {round(l,2)} {round(h,2)}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{round(w,2)} {round(l,2)} {round(h,2)}</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>
"""
            )

        xmlFile.write("""
    <plugin 
      name="bounding_box_plugin" 
      filename="/home/andrea/ros_packages_aggiuntivi/devel/lib/libbounding_box_plugin.so" 
    />
                      
    </world>
</sdf>
""")
    print(f"Gazebo world saved to {world_filename}")

if __name__ == "__main__":
    num_obstacles = 8
    world_size = 10
    min_size, max_size = 0.5, 2.0
    obstacles = generate_obstacle_positions(num_obstacles, world_size, min_size, max_size)
    create_gazebo_world(obstacles)
