"""Forward-walking task for the self-built dog (mirrors Go2 joystick, sagittal only)."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.mydog import base as dog_base
from mujoco_playground._src.locomotion.mydog import mydog_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=500,
      Kp=40.0,
      Kd=0.5,
      action_scale=0.1,
      torque_limit=12.0,
      command_vx=[0.3, 1.0],
      gait_period_s=0.5,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05, joint_vel=2.0, gyro=0.3, gravity=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.5,
              orientation=1.0,
              feet_air_time=0.5,
              feet_clearance=-2.0,
              gait=2.0,
              foot_lift=1.0,
              dof_pos_limits=-0.5,
              torques=-0.0005,
              action_rate=-0.01,
              termination=-1.0,
          ),
          tracking_sigma=0.25,
          fall_upv=0.4,
          max_foot_height=0.06,
      ),
      impl="jax",
      nconmax=1024,
      njmax=80,
  )


class Walk(dog_base.MyDogEnv):
  """Walk forward at a commanded speed (sagittal), stay upright."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
    self._imu_site_id = self._mj_model.site("imu").id
    self._up_vec = jp.array([0.0, 0.0, -1.0])

    act_joints = [
        self._mj_model.actuator_trnid[a, 0] for a in range(self._mj_model.nu)
    ]
    self._act_qadr = jp.array(
        [self._mj_model.jnt_qposadr[j] for j in act_joints]
    )
    self._act_vadr = jp.array(
        [self._mj_model.jnt_dofadr[j] for j in act_joints]
    )
    self._default_act = self._default_pose[self._act_qadr - 7]

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    self._foot_sensor_adr = jp.array([
        self._mj_model.sensor_adr[self._mj_model.sensor(f"foot{i}_floor").id]
        for i in range(1, 5)
    ])
    self._foot_geom_id = jp.array([
        self._mj_model.geom(f"foot_col{i}").id for i in range(1, 5)
    ])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    rng, key = jax.random.split(rng)
    cmd = jp.array([
        jax.random.uniform(key, minval=self._config.command_vx[0],
                           maxval=self._config.command_vx[1]),
        0.0, 0.0,
    ])

    data = mjx_env.make_data(
        self.mj_model, qpos=qpos, qvel=qvel, ctrl=self._default_act,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax, njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    info = {
        "rng": rng, "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "swing_peak": jp.zeros(4),
        "last_swing_peak": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "gait_phase": jp.zeros(()),
    }
    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    q_act = state.data.qpos[self._act_qadr]
    qd_act = state.data.qvel[self._act_vadr]
    target = self._default_act + action * self._config.action_scale
    torque = self._config.Kp * (target - q_act) - self._config.Kd * qd_act
    ctrl = jp.clip(torque, -self._config.torque_limit, self._config.torque_limit)

    data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

    contact = jp.array([
        data.sensordata[a] > 0 for a in self._foot_sensor_adr
    ])
    first_contact = (state.info["feet_air_time"] > 0.0) & (
        contact | state.info["last_contact"]
    )
    liftoff = (~contact) & state.info["last_contact"]
    state.info["feet_air_time"] += self.dt
    p_fz = data.geom_xpos[self._foot_geom_id][:, 2]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
    state.info["last_swing_peak"] = jp.where(
        first_contact, state.info["swing_peak"],
        jp.where(liftoff, jp.zeros_like(state.info["swing_peak"]),
                 state.info["last_swing_peak"]),
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, first_contact, contact, done
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    reward = jp.nan_to_num(reward)

    state.info["last_act"] = action
    state.info["feet_air_time"] *= ~contact
    state.info["swing_peak"] *= ~contact
    state.info["last_contact"] = contact
    state.info["gait_phase"] = jp.mod(
        state.info["gait_phase"] + 2.0 * jp.pi * self.dt / self._config.gait_period_s,
        2.0 * jp.pi,
    )
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.float32(done)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    return jp.logical_or(
        self.get_upvector(data)[-1] < self._config.reward_config.fall_upv,
        data.qpos[2] < 0.08,
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    gravity = self.get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]
    info["rng"], k1, k2 = jax.random.split(info["rng"], 3)
    noisy_gyro = gyro + (2 * jax.random.uniform(k1, gyro.shape) - 1) * \
        self._config.noise_config.level * self._config.noise_config.scales.gyro
    noisy_gravity = gravity + (2 * jax.random.uniform(k2, gravity.shape) - 1) * \
        self._config.noise_config.level * self._config.noise_config.scales.gravity
    state = jp.concatenate([
        noisy_gyro,                       # 3
        noisy_gravity,                    # 3
        info["command"],                  # 3
        jp.sin(info["gait_phase"])[None],  # 1  (步态时钟)
        jp.cos(info["gait_phase"])[None],  # 1
        joint_angles - self._default_pose,  # 8
        joint_vel,                        # 8
        info["last_act"],                 # 8
    ])
    return {"state": jp.clip(jp.nan_to_num(state), -100.0, 100.0)}

  def _get_reward(
      self, data: mjx.Data, action: jax.Array, info: dict[str, Any],
      first_contact: jax.Array, contact: jax.Array, done: jax.Array,
  ) -> dict[str, jax.Array]:
    forward = self.get_forwardvector(data)
    v_fwd = jp.dot(forward, data.qvel[0:3])
    gravity = self.get_gravity(data)
    up_error = jp.sum(jp.square(self._up_vec - gravity))

    cmd_vx = info["command"][0]
    track = jp.exp(-jp.square(v_fwd - cmd_vx) / self._config.reward_config.tracking_sigma)
    air = jp.sum(first_contact * info["feet_air_time"])

    # 步态时钟奖励（trot：对角腿同相 L1↔L4、L2↔L3）
    phase = info["gait_phase"]
    trot_offsets = jp.array([0.0, jp.pi, jp.pi, 0.0])
    expected_stance = jp.sin(phase + trot_offsets) < 0.0
    gait = jp.mean((contact == expected_stance).astype(jp.float32))

    # 抬脚高度：触地时惩罚"摆动峰值偏离目标抬脚高度"
    clearance = jp.sum(first_contact * jp.square(
        info["swing_peak"] / self._config.reward_config.max_foot_height - 1.0
    ))

    # 每只脚最近一次摆动的抬脚峰值达到目标高度才给分（不抬脚的脚得 0 分，
    # 无法靠"都不抬"刷分）→ 强制四足都迈步，治拖腿/不对称。
    lift = jp.sum(jp.clip(
        info["last_swing_peak"] / self._config.reward_config.max_foot_height,
        0.0, 1.0,
    ))

    return {
        "tracking_lin_vel": track,
        "orientation": jp.exp(-2.0 * up_error),
        "feet_air_time": air,
        "feet_clearance": clearance,
        "gait": gait,
        "foot_lift": lift,
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "torques": jp.sum(jp.square(data.actuator_force)),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "termination": done,
    }

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out)
