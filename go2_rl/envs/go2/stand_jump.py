# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Standing long-jump over a thin wall for the Go2.

From rest, leap forward over a thin vertical wall and land upright.

Observation (45):
    - Linear velocity (3), Gyroscope (3), Gravity (3),
      Joint angles rel. nominal (12), Joint velocities (12), Last action (12)

Action: joint position targets = nominal pose + action * action_scale.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import base as go2_base
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=120,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      soft_joint_pos_limit_factor=0.95,
      wall_x=0.6,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
              forwardvector=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              forward=1.0,
              air=0.5,
              air_forward=2.0,
              air_height=2.0,
              air_level=0.5,
              cross=5.0,
              stay=3.0,
              orientation=0.6,
              height=0.4,
              ang_vel=-0.05,
              partial_air=-0.5,
              action_rate=-0.01,
              torques=-0.0002,
          ),
          max_forward_speed=3.0,
          target_height=0.30,
          stand_height=0.22,
          upright_gz=-0.6,
          facing_thresh=0.5,
          cross_bonus=15.0,
          fall_penalty=8.0,
      ),
      impl="jax",
      nconmax=4 * 8192,
      njmax=40,
  )


class StandJump(go2_base.Go2Env):
  """Standing long jump over a thin wall."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    xml_path = (
        consts.ROOT_PATH / "xmls" / "scene_mjx_feetonly_wall_flat_terrain.xml"
    ).as_posix()
    super().__init__(
        xml_path=xml_path,
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor
    self._imu_site_id = self._mj_model.site("imu").id
    self._up_vec = jp.array([0.0, 0.0, -1.0])
    self._start_x = self._init_q[0]
    self._wall_x = self._config.wall_x

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "has_crossed": jp.zeros((), dtype=bool),
    }
    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward/cross_bonus"] = jp.zeros(())
    metrics["reward/fall_penalty"] = jp.zeros(())
    metrics["forward_displacement"] = jp.zeros(())
    metrics["crossed"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    airborne = jp.all(~contact)

    crossed_now = data.qpos[0] > self._wall_x
    first_cross = (crossed_now & (~state.info["has_crossed"])).astype(jp.float32)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, contact)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    reward = jp.nan_to_num(reward) + first_cross * self._config.reward_config.cross_bonus
    reward = reward - done.astype(jp.float32) * self._config.reward_config.fall_penalty

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["has_crossed"] = state.info["has_crossed"] | crossed_now
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["reward/cross_bonus"] = first_cross * self._config.reward_config.cross_bonus
    state.metrics["reward/fall_penalty"] = -done.astype(jp.float32) * self._config.reward_config.fall_penalty
    state.metrics["forward_displacement"] = data.qpos[0] - self._start_x
    state.metrics["crossed"] = crossed_now.astype(jp.float32)

    done = jp.float32(done)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    return self.get_upvector(data)[-1] < 0.0

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    forwardvector = mjx_env.get_sensor_data(
        self.mj_model, data, "forwardvector"
    )
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_forwardvector = (
        forwardvector
        + (2 * jax.random.uniform(noise_rng, shape=forwardvector.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.forwardvector
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    state = jp.concatenate([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_forwardvector,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
    ])
    return {"state": jp.clip(jp.nan_to_num(state), -100.0, 100.0)}

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    num_contact = jp.sum(contact)
    airborne = jp.all(~contact)
    partial = (num_contact > 0) & (num_contact < 4)

    forwardvector = mjx_env.get_sensor_data(
        self.mj_model, data, "forwardvector"
    )
    v_forward = jp.dot(forwardvector, data.qvel[0:3])
    gravity = self.get_gravity(data)
    gyro = self.get_gyro(data)
    torso_height = data.site_xpos[self._imu_site_id][2]
    crossed = (data.qpos[0] > self._wall_x).astype(jp.float32)

    up_error = jp.sum(jp.square(self._up_vec - gravity))
    ground = jp.logical_not(airborne)
    upright = gravity[2] < self._config.reward_config.upright_gz
    at_height = torso_height > self._config.reward_config.stand_height
    facing = forwardvector[0] > self._config.reward_config.facing_thresh
    stay = crossed * ground.astype(jp.float32) * upright.astype(
        jp.float32
    ) * at_height.astype(jp.float32) * facing.astype(jp.float32)
    height_error = torso_height - self._config.reward_config.target_height

    return {
        "forward": jp.clip(
            v_forward, 0.0, self._config.reward_config.max_forward_speed
        ),
        "air": airborne.astype(jp.float32),
        "air_forward": airborne.astype(jp.float32) * jp.clip(
            v_forward, 0.0, self._config.reward_config.max_forward_speed
        ),
        "air_height": airborne.astype(jp.float32) * jp.clip(
            torso_height - self._config.reward_config.target_height, 0.0, 0.5
        ),
        "air_level": airborne.astype(jp.float32) * jp.exp(-2.0 * up_error),
        "cross": crossed,
        "stay": stay,
        "orientation": jp.exp(-2.0 * up_error),
        "height": (
            ground * jp.exp(-10.0 * jp.square(height_error))
        ).astype(jp.float32),
        "ang_vel": jp.sum(jp.square(gyro)),
        "partial_air": partial.astype(jp.float32),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "torques": jp.sum(jp.square(data.actuator_force)),
    }
