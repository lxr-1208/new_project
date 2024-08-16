import cv2
import numpy as np
import isaacgym
import mediapipe as mp
import csv
from isaacgym import gymapi, gymtorch
import time
import math
import torch
import pytorch3d
from pytorch3d.transforms import axis_angle_to_quaternion
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
gym.add_ground(sim, plane_params)
asset_options = gymapi.AssetOptions()
asset_options.flip_visual_attachments = False
asset_options.fix_base_link = True
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True
asset_options.thickness = 0.001
asset_options.angular_damping = 0.01
asset_options.use_physx_armature = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

asset_root = "../../assets"
shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
shadow_hand_another_asset_file = "mjcf/open_ai_assets/hand/shadow_hand1.xml"
object_asset_file = "dataset_apien/12838/mobility.urdf"
shadow_hand_asset = gym.load_asset(sim, asset_root, shadow_hand_asset_file, asset_options)
shadow_hand_another_asset = gym.load_asset(sim, asset_root, shadow_hand_another_asset_file, asset_options)
object_asset = gym.load_asset(sim, asset_root, object_asset_file, asset_options)

spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 8)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

total_data = np.load('data.npy', allow_pickle=True).item()
right_hand_rot = total_data['entities']['right']['hand_poses'][:, 0:3]
right_hand_trans = total_data['entities']['right']['hand_trans']
left_hand_rot = total_data['entities']['left']['hand_poses'][:, 0:3]
left_hand_trans = total_data['entities']['left']['hand_trans']
object_rot = total_data['entities']['object']['object_poses'][:, 0:3]
object_trans = total_data['entities']['object']['object_poses'][:, 3:6]
object_scale = total_data['entities']['object']['obj_scale']

shadow_hand_start_pose = gymapi.Transform()
shadow_another_hand_start_pose = gymapi.Transform()
object_start_pose = gymapi.Transform()
shadow_hand_actor = gym.create_actor(env, shadow_hand_asset, shadow_hand_start_pose, 'shadow_hand', 0, 1)
shadow_another_hand_actor = gym.create_actor(env, shadow_hand_another_asset, shadow_another_hand_start_pose, 'shadow_a_hand', 0, 1)
object_actor = gym.create_actor(env, object_asset, object_start_pose, 'object', 0, 1)
gym.set_actor_scale(env, object_actor, object_scale)

root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
root_state_tensor_torch = gymtorch.wrap_tensor(root_state_tensor)

shadow_hand_actor_idx = gym.find_actor_index(env, 'shadow_hand', gymapi.DOMAIN_SIM)
shadow_another_hand_actor_idx = gym.find_actor_index(env, 'shadow_a_hand', gymapi.DOMAIN_SIM)
object_actor_idx = gym.find_actor_index(env, 'object', gymapi.DOMAIN_SIM)

for index in range(right_hand_rot.shape[0]):
    # 右手状态
    quat = axis_angle_to_quaternion(torch.tensor(right_hand_rot[index], dtype=torch.float32))
    right_hand_pos = torch.tensor([right_hand_trans[index][0]+14, right_hand_trans[index][1], right_hand_trans[index][2] + 2], dtype=torch.float32)
    root_state_tensor_torch[shadow_another_hand_actor_idx, :3] = right_hand_pos
    root_state_tensor_torch[shadow_another_hand_actor_idx, 3] = quat[1]
    root_state_tensor_torch[shadow_another_hand_actor_idx, 4] = quat[2]
    root_state_tensor_torch[shadow_another_hand_actor_idx, 5] = quat[3]
    root_state_tensor_torch[shadow_another_hand_actor_idx, 6] = quat[0]
    # 左手状态
    quat = axis_angle_to_quaternion(torch.tensor(left_hand_rot[index], dtype=torch.float32))
    left_hand_pos = torch.tensor([left_hand_trans[index][0]+14, left_hand_trans[index][1], left_hand_trans[index][2] + 2], dtype=torch.float32)
    root_state_tensor_torch[shadow_hand_actor_idx, :3] = left_hand_pos
    root_state_tensor_torch[shadow_hand_actor_idx, 3] = quat[1]
    root_state_tensor_torch[shadow_hand_actor_idx, 4] = quat[2]
    root_state_tensor_torch[shadow_hand_actor_idx, 5] = quat[3]
    root_state_tensor_torch[shadow_hand_actor_idx, 6] = quat[0]
    # 物体状态
    quat = axis_angle_to_quaternion(torch.tensor(object_rot[index], dtype=torch.float32))
    object_pos = torch.tensor([object_trans[index][0] * object_scale + 14, object_trans[index][1] * object_scale, object_trans[index][2] * object_scale + 1], dtype=torch.float32)
    root_state_tensor_torch[object_actor_idx, :3] = object_pos
    root_state_tensor_torch[object_actor_idx, 3] = quat[1]
    root_state_tensor_torch[object_actor_idx, 4] = quat[2]
    root_state_tensor_torch[object_actor_idx, 5] = quat[3]
    root_state_tensor_torch[object_actor_idx, 6] = quat[0]
    #print(root_state_tensor_torch)
    root_state_tensor = gymtorch.unwrap_tensor(root_state_tensor_torch)
    gym.set_actor_root_state_tensor(sim, root_state_tensor)

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    time.sleep(1/60)

# 销毁查看器和模拟
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)