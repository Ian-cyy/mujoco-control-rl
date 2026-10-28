"""Standing / balance task for the self-built dog."""

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
      episode_length=300,
      Kp=40.0,
      Kd=0.5,
      action_scale=0.1,
      torque_limit=12.0,
      target_height=0.30,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              joint_vel=2.0,
              gyro=0.3,
              gravity=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              upright=1.0,
              height=1.0,
              stand_still=0.5,
              action_rate=-0.01,
              torques=-0.001,
          ),
          fall_grace=0.3,
      ),
      impl="jax",
      nconmax=512,
      njmax=60,
  )


class Stand(dog_base.MyDogEnv):
  """Hold the base upright at a target height and stay calm."""

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
    self._act_vadr = jp.array([self._mj_model.jnt_dofadr[j] for j in act_joints])
    self._default_act = self._default_pose[self._act_qadr - 7]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=self._default_act,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
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

    data = mjx_env.step(
        self.mjx_model, state.data, ctrl, self.n_substeps
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    reward = jp.nan_to_num(reward)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.float32(done)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    height = data.site_xpos[self._imu_site_id][2]
    fell = self.get_upvector(data)[-1] < self._config.reward_config.fall_grace
    low = height < 0.04
    return jp.logical_or(jp.logical_or(fell, low), jp.logical_not(jp.isfinite(height)))

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    gravity = self.get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]

    info["rng"], k1, k2 = jax.random.split(info["rng"], 3)
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(k1, gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(k2, gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    state = jp.concatenate([
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        joint_angles - self._default_pose,  # 16
        joint_vel,  # 16
        info["last_act"],  # 8
    ])
    return {"state": jp.clip(jp.nan_to_num(state), -100.0, 100.0)}

  def _get_reward(
      self, data: mjx.Data, action: jax.Array, info: dict[str, Any]
  ) -> dict[str, jax.Array]:
    gravity = self.get_gravity(data)
    base_z = data.site_xpos[self._imu_site_id][2]
    qd = data.qvel[6:]

    up_error = jp.sum(jp.square(self._up_vec - gravity))
    height_error = base_z - self._config.target_height

    return {
        "upright": jp.exp(-2.0 * up_error),
        "height": jp.clip(base_z / self._config.target_height, 0.0, 1.0),
        "stand_still": jp.exp(-0.5 * jp.sum(jp.square(action))),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "torques": jp.sum(jp.square(data.actuator_force)),
    }
