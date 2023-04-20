from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import PushArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
import time
from robosuite.models.objects import CylinderObject
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements
from robosuite.environments.obstacle_estimator.svm_estimator import SVMModel
from robosuite.environments.obstacle_estimator.relative_fc_nn_estimator import RelativeFCNN
from robosuite.environments.obstacle_estimator.relative_fc_nn_estimator_complex import RelativeComplexFCNN
from robosuite.environments.obstacle_estimator.absolute_fc_nn_estimator_complex import AbsoluteComplexFCNN

class PushAway(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="WipingGripper",
        initialization_noise="default",
        table_full_size=(1.2, 2.5, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),

        use_camera_obs=True,

        use_object_obs=True,        # use cube info 

        close_to_goal_threshold = 0.02,

        is_contact_logging = True,
        contact_force_limit = 20,

        is_using_estimator = True,

        model_name = 'fc_nn_complex_direct',
        model_prefix = 'fc_nn/large_complex_direct/',

        # model_name = 'fc_nn_complex',
        # # model_prefix = 'fc_nn/small_complex/',
        # model_prefix = 'fc_nn/big_complex/',

        # model_name = 'fc_nn_position',
        # model_prefix = 'fc_nn/relative/big_pos/',

        # model_name = 'SVM',
        # model_prefix = 'svm/small/',
        is_estimator_logging = True,


        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # close to the goal position
        self.close_to_goal_threshold = close_to_goal_threshold

        # log the contact information
        self.is_contact_logging = is_contact_logging
        self.is_estimator_logging = is_estimator_logging

        if self.is_contact_logging:
            self.contact_log_name = "contact_log/"+time.strftime("%Y%m%d-%H%M%S")+".csv"
            print("#########################################\nContacf detail is logged to: {}".format(self.contact_log_name))

            # first line
            with open(self.contact_log_name, 'a') as f:
                f.write( 'contact_x,contact_y,contact_z,moment_x,moment_y,moment_z,force_x,force_y,force_z,cube_x,cube_y,cube_z,qx,qy,qz,qw\n')
            

        if self.is_estimator_logging:
            self.estimator_log_name = "estimator_log/"+time.strftime("%Y%m%d-%H%M%S")+".csv"
            print("#########################################\nContacf detail is logged to: {}".format(self.contact_log_name))

            with open(self.estimator_log_name, 'a') as f:
                f.write( 'cube_x,cube_y,cube_z,qx,qy,qz,qw\n')

        
        # contact force limit
        self.contact_force_limit = contact_force_limit

        # using estimator
        self.is_using_estimator = is_using_estimator

        # set up estimator
        if self.is_using_estimator:
            print("\nUsing estimator for cube position\n")
            if model_name == 'SVM':
                # let the reset to set up the initial position
                self.estimator = SVMModel(prefix=model_prefix)
                # self.estimator = SVMModel(prefix=model_prefix, start_pose=[0.0,0.0,0.0,0.0,0.0,0.0,1.0])

            elif model_name == 'fc_nn_position':
                self.estimator = RelativeFCNN(prefix=model_prefix)

            elif model_name == 'fc_nn_complex':
                self.estimator = RelativeComplexFCNN(prefix=model_prefix)
            elif model_name == 'fc_nn_complex_direct':
                self.estimator = AbsoluteComplexFCNN(prefix=model_prefix)
                



        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is pushed away

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Contact: in {0, 1}, non-zero if arm contacts the block, penalize the 
                the large contact force
            - Moving: reward the block position

        The sparse reward only consists of the pushing component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            if self.is_using_estimator:
                cube_pos = self.estimator.get_pose()
                cube_pos = cube_pos[0:3]
            else:
                cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)

            # mapping to better range
            reaching_reward = 1 - np.tanh(2/0.34*(dist-0.06))
            # reward += 0.25*reaching_reward

            # block position reward
            goal_pose = self.model.mujoco_arena.goal_pose
            dist = np.linalg.norm(goal_pose - cube_pos[:2])
            position_reward = 1 - np.tanh(2*dist)
            reward += 1.5*position_reward

            # contact reward
            logged = False
            for i in range(self.sim.data.ncon):
                # Note that the contact array has more than `ncon` entries,
                # so be careful to only read the valid entries.
                contact = self.sim.data.contact[i]

                geom1_body = self.sim.model.geom_bodyid[contact.geom1]
                geom2_body = self.sim.model.geom_bodyid[contact.geom2]

                # anything contact with cube would be logged
                # note: we use the cube to get the contact information
                # this could be done with robot arm. However, there is too much element to consider for the robot arm
                # as a result, direct consider the contact with the cube would work.
                # this is nothing different than the reverse of the arm contact
                # in reality, this would be generated by other algorithms/prediction
                if geom1_body == self.cube_body_id or geom2_body == self.cube_body_id:
                    # if it is the bottom surface, skip
                    contact_pos = contact.pos
                    contact_force = self.sim.data.cfrc_ext[self.cube_body_id]
                    abs_force = np.linalg.norm(contact_force[3:]) 

                    # contact the table
                    if abs( contact_pos[2] - self.model.mujoco_arena.table_top_abs[2]) < self.close_to_goal_threshold:
                        continue

                    force_reward = 1 - np.tanh(0.2*abs(abs_force - self.contact_force_limit))
                    reward += 0.5*force_reward + 0.25

                    # log data here
                    if self.is_contact_logging:
                        data = np.concatenate( [contact_pos, contact_force, np.array(self.sim.data.body_xpos[self.cube_body_id]), convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")])
                        self.log_data(data)

                        logged = True

                    break
            
            if not logged and self.is_contact_logging:
                # log data here
                data = np.concatenate( [np.zeros(9), np.array(self.sim.data.body_xpos[self.cube_body_id]), convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")])
                self.log_data(data)

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = PushArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        # cube
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.12, 0.12, 0.12],  # [0.015, 0.015, 0.015],
            size_max=[0.15, 0.15, 0.15],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            density=30,
            material=redwood,
        )
        
        # goal position marker
        # add marker to goal position
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.0",
            "shininess": "0.0",
        }

        dirt = CustomMaterial(
            texture="Dirt",
            tex_name="dirt",
            mat_name="dirt_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=True,
        )

        self.goal_marker  = CylinderObject(
            name="GoalPosition",
            size=[ 0.1, 0.001],
            rgba=[1, 1, 1, 1],
            material=dirt,
            obj_type="visual",
            joints=None,
        )

        # Manually add this object to the arena xml
        mujoco_arena.merge_assets(self.goal_marker)
        table = find_elements(root=mujoco_arena.worldbody, tags="body", attribs={"name": "table"}, return_first=True)
        table.append(self.goal_marker.get_obj())


        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=self.cube,
                # x_range=[-self.table_full_size[0]*0.2, self.table_full_size[0]*0.2],
                # y_range=[-self.table_full_size[1]*0.2, self.table_full_size[1]*0.2],
                x_range=[-0.02, 0.02],
                y_range=[-0.02, 0.02],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
             
            # todo: estimator

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

        # todo: fake it now - need to change later
        # self.estimator.set_start_pose([0,0,0,0,0,0,1])

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        sensors = []

        # low-level object information
        # Get robot prefix and define observables modality
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        # if there is cube info
        if self.use_object_obs:
            
            if not self.is_using_estimator:
                # cube-related observables
                @sensor(modality=modality)
                def cube_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.cube_body_id])

                sensors.append(cube_pos)

                @sensor(modality=modality)
                def cube_quat(obs_cache):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

                sensors.append(cube_quat)

                # @sensor(modality=modality)
                # def gripper_to_cube_pos(obs_cache):
                #     return (
                #         obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"]
                #         if f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache
                #         else np.zeros(3)
                #     )
                # sensors.append(gripper_to_cube_pos)
            
                # add goal position
                @sensor(modality=modality)
                def goal_position(obs_cache):
                    return self.model.mujoco_arena.goal_pose
            else:
                # using estimator
                # cube-related observables
                @sensor(modality=modality)
                def cube_pos(obs_cache):
                    for i in range(self.sim.data.ncon):
                        # Note that the contact array has more than `ncon` entries,
                        # so be careful to only read the valid entries.
                        contact = self.sim.data.contact[i]

                        geom1_body = self.sim.model.geom_bodyid[contact.geom1]
                        geom2_body = self.sim.model.geom_bodyid[contact.geom2]

                        # anything contact with cube would be logged
                        if geom1_body == self.cube_body_id or geom2_body == self.cube_body_id:
                            contact_pos = contact.pos
                            contact_force = self.sim.data.cfrc_ext[self.cube_body_id]
                            
                            # if it is the bottom surface, skip
                            if abs( contact_pos[2] - self.model.mujoco_arena.table_top_abs[2]) < self.close_to_goal_threshold:
                                continue
                            
                            self.estimator.combined_predict(np.concatenate([contact_pos, contact_force]))

                            return np.array( self.estimator.get_pose()[:3])
                    return np.array(self.estimator.get_pose()[:3])
                
                @sensor(modality=modality)
                def cube_quat(obs_cache):
                    for i in range(self.sim.data.ncon):
                        # Note that the contact array has more than `ncon` entries,
                        # so be careful to only read the valid entries.
                        contact = self.sim.data.contact[i]

                        geom1_body = self.sim.model.geom_bodyid[contact.geom1]
                        geom2_body = self.sim.model.geom_bodyid[contact.geom2]

                        # anything contact with cube would be logged
                        if geom1_body == self.cube_body_id or geom2_body == self.cube_body_id:
                            contact_pos = contact.pos
                            contact_force = self.sim.data.cfrc_ext[self.cube_body_id]
                            
                            # if it is the bottom surface, skip
                            if abs( contact_pos[2] - self.model.mujoco_arena.table_top_abs[2]) < self.close_to_goal_threshold:
                                continue
                            
                            self.estimator.combined_predict(np.concatenate([contact_pos, contact_force]))

                            return np.array( self.estimator.get_pose()[3:])
                    return np.array(self.estimator.get_pose()[3:])

                sensors.append(cube_quat)

                sensors.append(cube_pos)

                # add goal position
                @sensor(modality=modality)
                def goal_position(obs_cache):
                    return self.model.mujoco_arena.goal_pose
        else:
            # add goal position
            @sensor(modality=modality)
            def goal_position(obs_cache):
                return self.model.mujoco_arena.goal_pose
        
        sensors.append(goal_position)

        # add contact info
        @sensor(modality=modality)
        def contact_info(obs_cache):
            for i in range(self.sim.data.ncon):
                # Note that the contact array has more than `ncon` entries,
                # so be careful to only read the valid entries.
                contact = self.sim.data.contact[i]

                geom1_body = self.sim.model.geom_bodyid[contact.geom1]
                geom2_body = self.sim.model.geom_bodyid[contact.geom2]

                # anything contact with cube would be logged
                if geom1_body == self.cube_body_id or geom2_body == self.cube_body_id:
                    contact_pos = contact.pos
                    contact_force = self.sim.data.cfrc_ext[self.cube_body_id]
                    
                    # if it is the bottom surface, skip
                    if abs( contact_pos[2] - self.model.mujoco_arena.table_top_abs[2]) < self.close_to_goal_threshold:
                        continue
                    
                    return np.concatenate( [contact_pos, contact_force])

            return np.zeros(9)

        sensors.append(contact_info)

        # add arm info
        @sensor(modality=modality)
        def arm_info(obs_cache):
            joint_pos = self.robots[0]._joint_positions
            joint_energy = self.robots[0].js_energy
            joint_vel = self.robots[0]._joint_velocities
            return np.array(joint_pos+joint_energy+joint_vel)

        sensors.append(arm_info)
        

        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            self.model.mujoco_arena.reset_arena(self.sim, self.goal_marker)

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # print("data for rest - pos: {}, quat: {}, obj: {}".format(obj_pos, obj_quat, obj))
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

                if obj == self.cube:
                     # reset estimator
                    if self.is_using_estimator:
                        pose = list(obj_pos) + list(convert_quat(np.array(obj_quat), to="xyzw"))
                        print("pose generated: {}".format(pose))
                        self.estimator.set_start_pose(pose)


           

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been moved.

        Returns:
            bool: True if cube has been moved
        """
        if self.is_using_estimator:
            cube_position = self.estimator.get_pose()
        else:
            cube_position = self.sim.data.body_xpos[self.cube_body_id]
        targe_position = self.model.mujoco_arena.goal_pose

        # cube is higher than the table top above a margin
        return abs(cube_position[0]-targe_position[0]) < self.close_to_goal_threshold and abs(cube_position[1]-targe_position[1]) < self.close_to_goal_threshold

    def log_data(self, data):
        """
        log the data to the default file
        """

        with open(self.contact_log_name, 'a') as f:
            np.savetxt(f, np.reshape(data, (1,-1)), delimiter=",")

        if self.is_estimator_logging:
            # log position and rotation
            with open(self.estimator_log_name, 'a') as f:
                np.savetxt(f, np.reshape(self.estimator.get_pose(), (1,-1)), delimiter=",")

        