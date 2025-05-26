import random

def generate_actor_positions(num_actors, world_size):
    """
    Genera per ciascun actor una posizione di spawn e un obiettivo (goal).
    I valori sono randomizzati all'interno di un mondo di dimensione world_size.
    """
    actors = []
    for _ in range(num_actors):
        spawn_x = random.uniform(1.5, world_size)
        spawn_y = random.uniform(1.5, world_size)
        orient_z = random.uniform(0, 2*3.14)
        actors.append((spawn_x, spawn_y, orient_z))
    return actors

def create_gazebo_world_with_actors(actors, world_filename="/home/andrea/ros_packages_aggiuntivi/src/llm_generation_trajectory/world/dynamic_world_4obs.world"):
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
        
        # Per ogni actor, generiamo il blocco XML.
        # Usiamo valori arrotondati a 2 cifre decimali.
        for i, (spawn_x, spawn_y, orient_z) in enumerate(actors):
            # Impostiamo una leggera differenza per il primo actor (nessun delay) e il secondo (con delay)
            delay = "0" if i == 0 else "15"
            xmlFile.write(
                f"""
    <actor name="actor{i+1}">
      <pose>{round(spawn_x,2)} {round(spawn_y,2)} 1 0 0 {1.57 + round(orient_z,2)}</pose>
      <skin>
        <filename>walk.dae</filename>
        <scale>1.0</scale>
      </skin>
      <animation name="walking">
        <filename>walk.dae</filename>
        <scale>1.00</scale>
        <interpolate_x>true</interpolate_x>
      </animation>
      <delay_start>{delay}</delay_start>
      <plugin name="actor{i+1}_plugin" filename="libgazebo_ros_actor_command.so">
        <!-- <follow_mode>path</follow_mode> -->
        <follow_mode>velocity</follow_mode>
        <vel_topic>/actor{i+1}/cmd_vel</vel_topic>
        <path_topic>/actor{i+1}/cmd_path</path_topic>
        <animation_factor>4.0</animation_factor>
        <linear_tolerance>0.1</linear_tolerance>
        <linear_velocity>1</linear_velocity>
        <angular_tolerance>0.0872</angular_tolerance>
        <angular_velocity>2.5</angular_velocity>
        <default_rotation>1.57</default_rotation>
      </plugin>
    </actor>
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
    num_actors = 4
    world_size = 10  # Dimensione del mondo in cui vengono randomizzate le posizioni
    actors = generate_actor_positions(num_actors, world_size)
    create_gazebo_world_with_actors(actors)
