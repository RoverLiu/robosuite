import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, xml_path_completion
from robosuite.models.objects import CylinderObject
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements

class PushArena(Arena):
    """
    Workspace that contains an empty table.


    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        table_full_size=(1.2, 2.5, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        line_width=0.2,
        has_legs=True,
        xml="arenas/table_arena.xml",
    ):
        super().__init__(xml_path_completion(xml))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = self.bottom_pos + np.array([0, 0, -self.table_half_size[2]]) + self.table_offset

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.has_legs = has_legs
        self.table_legs_visual = [
            self.table_body.find("./geom[@name='table_leg1_visual']"),
            self.table_body.find("./geom[@name='table_leg2_visual']"),
            self.table_body.find("./geom[@name='table_leg3_visual']"),
            self.table_body.find("./geom[@name='table_leg4_visual']"),
        ]

        self.line_width = line_width
        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))

        self.table_top.set("pos", array_to_string(np.array([0, 0, self.table_half_size[2]])))

        # If we're not using legs, set their size to 0
        if not self.has_legs:
            for leg in self.table_legs_visual:
                leg.set("rgba", array_to_string([1, 0, 0, 0]))
                leg.set("size", array_to_string([0.0001, 0.0001]))
        else:
            # Otherwise, set leg locations appropriately
            delta_x = [0.1, -0.1, -0.1, 0.1]
            delta_y = [0.1, 0.1, -0.1, -0.1]
            for leg, dx, dy in zip(self.table_legs_visual, delta_x, delta_y):
                # If x-length of table is less than a certain length, place leg in the middle between ends
                # Otherwise we place it near the edge
                x = 0
                if self.table_half_size[0] > abs(dx * 2.0):
                    x += np.sign(dx) * self.table_half_size[0] - dx
                # Repeat the same process for y
                y = 0
                if self.table_half_size[1] > abs(dy * 2.0):
                    y += np.sign(dy) * self.table_half_size[1] - dy
                # Get z value
                z = (self.table_offset[2] - self.table_half_size[2]) / 2.0
                # Set leg position
                leg.set("pos", array_to_string([x, y, -z]))
                # Set leg size
                leg.set("size", array_to_string([0.025, z]))

        # define the goal position
        self.goal_pose = self.sample_start_pos()
        

    @property
    def table_top_abs(self):
        """
        Grabs the absolute position of table top

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset

    def sample_start_pos(self):
        """
        Helper function to return sampled end position of the cube

        Returns:
            np.array: the (x,y) value of the newly sampled dirt starting location
        """
        # First define the pose 
        # make it to a circle
        radius = 0.5   # half a meter
        pi = 3.1416
        angle = np.random.uniform(-pi/3*2, -pi/3)
        # angle = np.random.choice([np.random.uniform(pi/3, pi/3*2), np.random.uniform(-pi/3*2, -pi/3)])


        return np.array(
            (
                radius*np.cos(angle),
                radius*np.sin(angle),
            )
        ) 
        # return np.array(
        #     (
        #         np.random.uniform(
        #             -self.table_half_size[0] * 0.8,
        #             self.table_half_size[0] * 0.8,
        #         ),
        #         np.random.uniform(
        #             -self.table_half_size[1] * 0.8,
        #             self.table_half_size[1] * 0.8,
        #         ),
        #     )
        # )

    def reset_arena(self, sim, goal_marker):
        """
        Reset the visual marker locations in the environment. Requires @sim (MjSim) reference to be passed in so that
        the Mujoco sim can be directly modified

        Args:
            sim (MjSim): Simulation instance containing this arena and visual markers
        """
        # Sample new initial position and direction for generated marker paths
        pos = self.sample_start_pos()

        # check visual markers
        # Get IDs to the body, geom, and site of each marker
        body_id = sim.model.body_name2id(goal_marker.root_body)
        geom_id = sim.model.geom_name2id(goal_marker.visual_geoms[0])
        site_id = sim.model.site_name2id(goal_marker.sites[0])
        # Determine new position for this marker
        position = np.array([pos[0], pos[1], self.table_half_size[2]])
        # Set the current marker (body) to this new position
        sim.model.body_pos[body_id] = position
        # Reset the marker visualization -- setting geom rgba alpha value to 1
        sim.model.geom_rgba[geom_id][3] = 1
        # Hide the default visualization site
        sim.model.site_rgba[site_id][3] = 0