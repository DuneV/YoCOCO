import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config

import collections
from absl import logging
from dm_control import mujoco as dm_mujoco
from dm_control.mujoco.wrapper import mjbindings
import numpy as np
from pathlib import Path

locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

mjlib = mjbindings.mjlib

# from dm_control
_INVALID_JOINT_NAMES_TYPE = (
    '`joint_names` must be either None, a list, a tuple, or a numpy array; '
    'got {}.')
_REQUIRE_TARGET_POS_OR_QUAT = (
    'At least one of `target_pos` or `target_quat` must be specified.')

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])
#/ from dm_control 

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_SITES = [
   "right_shoulder_mark",
   "right_elbow_mark",
   "right_wrist_mark",
   "right_palm_mark"
]
LEFT_ARM_SITES = [
   "left_shoulder_mark",
   "left_elbow_mark",
   "left_wrist_mark",
   "left_palm_mark"
]

_INITIAL_QPOS = mj_data.qpos.copy()
_INITIAL_QVEL = mj_data.qvel.copy()
ARM_JOINT_NAMES = RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
ARM_JOINT_IDS = [mj_model.joint(name).id for name in ARM_JOINT_NAMES]
ARM_DOF_IDS = []

for name in ARM_JOINT_NAMES:
    j_id = mj_model.joint(name).id
    dof_adr = mj_model.jnt_dofadr[j_id]
    j_type = mj_model.jnt_type[j_id]
    if j_type == mujoco.mjtJoint.mjJNT_FREE:
        dof_num = 6
    elif j_type == mujoco.mjtJoint.mjJNT_BALL:
        dof_num = 3
    else:  # hinge or slide
        dof_num = 1
    ARM_DOF_IDS.extend(range(dof_adr, dof_adr + dof_num))

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_
dm_physics = dm_mujoco.Physics.from_xml_path(config.ROBOT_SCENE)

_latest_frame = None                   # np.array shape (6,3)
_first_msg_event = threading.Event()   # let sim wait until first data arrives


# ------- helpers -------

def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n

def human_to_robot(bod_list, physics):
    """
    converts the body vectors from human limb lengths to robot limb lengths
    bod_list: list of 6 np arrays, each with shape (3,)
              [R_sh, R_el, R_wr, L_sh, L_el, L_wr]
    returns: list of 4 np arrays, each with shape (3,) (doesnt return shoulder as it wont change its position)
    """
    upper_arm_h = 0.1854387771  # meters
    forearm_h = 0.1845164872    # meters

    R_sh_pos = physics.named.data.site_xpos["right_shoulder_mark"]
    L_sh_pos = physics.named.data.site_xpos["left_shoulder_mark"]

    R_el_target = R_sh_pos + unit(bod_list[1] - bod_list[0]) * upper_arm_h
    R_wr_target = R_el_target + unit(bod_list[2] - bod_list[1]) * forearm_h

    L_el_target = L_sh_pos + unit(bod_list[4] - bod_list[3]) * upper_arm_h
    L_wr_target = L_el_target + unit(bod_list[5] - bod_list[4]) * forearm_h

    return [R_el_target, R_wr_target, L_el_target, L_wr_target]

def multisite_qpos_from_site_pose(
    physics,
    site_names,
    target_positions,
    joint_names=None,
    tol=1e-3,
    regularization_strength=3e-2,
    max_update_norm=2.0,
    progress_thresh=20.0,
    max_steps=100,
    inplace=False
):
    """
    Multi-site IK: Finds joint positions that satisfy target positions for multiple sites.
    Args:
        physics: dm_control.mujoco.Physics instance.
        site_names: list of site names (e.g., ["right_elbow_mark", "right_palm_mark"])
        target_positions: list of (3,) np.arrays, one per site.
        joint_names: list of joint names to use for IK.
        ... (other args as in qpos_from_site_pose)
    Returns:
        IKResult namedtuple.
    """
    dtype = physics.data.qpos.dtype
    n_sites = len(site_names)
    jac = np.empty((3 * n_sites, physics.model.nv), dtype=dtype)
    err = np.empty(3 * n_sites, dtype=dtype)
    weights = [0.6,0.4]
    if not inplace:
        physics = physics.copy(share_model=True)
    mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

    for i, (site_name, target_pos) in enumerate(zip(site_names, target_positions)):
        site_id = physics.model.name2id(site_name, 'site')
        site_xpos = physics.named.data.site_xpos[site_name]
        err[3*i:3*i+3] = target_pos - site_xpos
        err[3*i:3*i+3] *= weights[i]  # Apply weight to the error
        jac_pos = jac[3*i:3*i+3]
        jac[3*i:3*i+3] *= weights[i]  # Apply weight to the Jacobian
        mjlib.mj_jacSite(physics.model.ptr, physics.data.ptr, jac_pos, None, site_id)

    if joint_names is None:
        dof_indices = slice(None)
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        indexer = physics.named.model.dof_jntid.axes.row
        dof_indices = indexer.convert_key_item(joint_names)
    else:
        raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

    update_nv = np.zeros(physics.model.nv, dtype=dtype)
    steps = 0
    success = False

    for steps in range(max_steps):
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            success = True
            break
        jac_joints = jac[:, dof_indices]
        reg_strength = regularization_strength if err_norm > tol else 0.0
        update_joints = nullspace_method(jac_joints, err, regularization_strength=reg_strength)
        update_norm = np.linalg.norm(update_joints)
        progress_criterion = err_norm / (update_norm + 1e-8)
        if progress_criterion > progress_thresh:
            break
        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm
        update_nv[dof_indices] = update_joints
        mjlib.mj_integratePos(physics.model.ptr, physics.data.qpos, update_nv, 1)
        mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)
        # Recompute error and jacobian for next iteration
        for i, (site_name, target_pos) in enumerate(zip(site_names, target_positions)):
            site_xpos = physics.named.data.site_xpos[site_name]
            err[3*i:3*i+3] = target_pos - site_xpos
            jac_pos = jac[3*i:3*i+3]
            site_id = physics.model.name2id(site_name, 'site')
            mjlib.mj_jacSite(physics.model.ptr, physics.data.ptr, jac_pos, None, site_id)

    if not inplace:
        qpos = physics.data.qpos.copy()
    else:
        qpos = physics.data.qpos

    return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)

def nullspace_method(jac_joints, delta, regularization_strength=0.0):
  """Calculates the joint velocities to achieve a specified end effector delta.

  Args:
    jac_joints: The Jacobian of the end effector with respect to the joints. A
      numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
      and `nv` is the number of degrees of freedom.
    delta: The desired end-effector delta. A numpy array of shape `(3,)` or
      `(6,)` containing either position deltas, rotation deltas, or both.
    regularization_strength: (optional) Coefficient of the quadratic penalty
      on joint movements. Default is zero, i.e. no regularization.

  Returns:
    An `(nv,)` numpy array of joint velocities.

  Reference:
    Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
    transpose, pseudoinverse and damped least squares methods.
    https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
  """
  hess_approx = jac_joints.T.dot(jac_joints)
  joint_delta = jac_joints.T.dot(delta)
  if regularization_strength > 0:
    # L2 regularization
    hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
    return np.linalg.solve(hess_approx, joint_delta)
  else:
    return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
  

def Simulation_loop():
    global mj_data, mj_model, _latest_frame

    if config.ENABLE_ELASTIC_BAND:
        elastic_band = ElasticBand()
        band_body = mj_model.body("torso_link").id if (config.ROBOT in ("h1","g1")) else mj_model.body("base_link").id
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback)
    else:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    mj_model.opt.timestep = config.SIMULATE_DT

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    time.sleep(0.2)

    # Wait for first message so targets are defined
    print("Waiting for first /human/arm_poses message…")
    _first_msg_event.wait()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_body, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        # -- IK block --
        if _latest_frame is not None:
            dm_physics.data.qpos[:] = mj_data.qpos
            mjbindings.mjlib.mj_fwdPosition(dm_physics.model.ptr, dm_physics.data.ptr)
            
            Rel_trgt, Rwr_trgt, Lel_trgt, Lwr_trgt = human_to_robot(_latest_frame, dm_physics)
            
            ik_result1 = multisite_qpos_from_site_pose(
                dm_physics,
                site_names=["right_wrist_mark","right_elbow_mark"],
                target_positions=[Rwr_trgt, Rel_trgt],
                joint_names=RIGHT_ARM_JOINTS,
                tol=1e-3,
                max_steps=10,
                inplace=True
            )
            ik_result2 = multisite_qpos_from_site_pose(
                dm_physics,
                site_names=["left_wrist_mark", "left_elbow_mark"],
                target_positions = [Lwr_trgt, Lel_trgt],
                joint_names=LEFT_ARM_JOINTS,
                tol=1e-3,
                max_steps=10,
                inplace=True
            )

            mj_data.qpos[:] = ik_result1.qpos
            mj_data.qpos[:] = ik_result2.qpos
            mujoco.mj_forward(mj_model, mj_data)

        # -- end IK block --

        # possibly add something to freeze base here
        # Overwrite all non-arm joints with their initial values
        for i in range(mj_model.nv):
            if i not in ARM_DOF_IDS:
                mj_data.qpos[i] = _INITIAL_QPOS[i]
                mj_data.qvel[i] = 0.0  # possible _INITIAL_QVEL[i] if I initial motion 
        
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        locker.release()

        # Keep real-time
        dt = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if dt > 0:
            time.sleep(dt)
  
def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)

class RobotSim(Node):
    def __init__(self):
        super().__init__('robot_sim_node')
        self.sub = self.create_subscription(PoseArray, '/human/arm_poses', self.get_coords, 10)
        self.get_logger().info("Subscribed to /human/arm_poses")

    def get_coords(self, msg):
        global _latest_frame
        if len(msg.poses) < 6:
            print("Received incomplete pose array")
            return
        bod_list = []
        #print(msg)
        for pose in msg.poses[:6]:
            arr = np.array([pose.position.x, pose.position.y, pose.position.z])
            if not np.isfinite(arr).all():
                print("Received invalid pose data")
                return
            bod_list.append(arr)
        #print(bod_list)
        _latest_frame = bod_list
        _first_msg_event.set()


def main(args=None):
    # rclpy.init(args=args)
    # robot_sim_node = RobotSim()
    # # viewer is in simulation loop so dont need to start physics viewer thread
    # # viewer_thread = Thread(target=PhysicsViewerThread, daemon=True)
    # sim_thread = Thread(target=Simulation_loop, daemon=True)

    # #viewer_thread.start()
    # sim_thread.start()

    # rclpy.spin(robot_sim_node)
    # robot_sim_node.destroy_node()
    # rclpy.shutdown()    
    return None

if __name__ == '__main__':
    main()
