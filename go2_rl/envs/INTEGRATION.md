# 将 go2 / mydog 环境接入 mujoco_playground

本仓库的 `envs/` 是**自写环境**，依赖 `mujoco_playground v0.1.0` 的框架。使用步骤：

1. 安装 `mujoco_playground`（源码 / pip 均可，版本 v0.1.0）。
2. 拷贝环境目录：
   ```
   cp -r envs/go2    <mujoco_playground>/_src/locomotion/go2
   cp -r envs/mydog  <mujoco_playground>/_src/locomotion/mydog
   ```
   （Go2 的 `xmls/*.xml` 里引用的模型来自 `mujoco_menagerie/unitree_go2`，按 playground 约定放到 `external_deps/mujoco_menagerie`。）
3. 注册环境（`_src/locomotion/__init__.py`）：
   ```python
   from mujoco_playground._src.locomotion.go2 import joystick as go2_joystick
   from mujoco_playground._src.locomotion.go2 import getup as go2_getup
   from mujoco_playground._src.locomotion.go2 import jump as go2_jump
   from mujoco_playground._src.locomotion.go2 import stand_jump as go2_stand_jump
   from mujoco_playground._src.locomotion.mydog import stand as mydog_stand
   from mujoco_playground._src.locomotion.mydog import walk as mydog_walk
   # 在 _envs / _config 字典里加对应名字 -> 类 / default_config
   #   "Go2Getup": go2_getup.Getup,  ... "MyDogWalkFlatTerrain": mydog_walk.Walk, ...
   ```
4. 域随机化：把 `go2_randomize.domain_randomize` / `mydog_randomize.domain_randomize` 注册为对应 env 的 randomizer。
5. 训练入口：`learning/train_jax_ppo.py --env_name=<名字> ...`

> 说明：具体注册字典的键名以你安装的 `mujoco_playground` 版本为准；本目录代码与 v0.1.0 对齐。
