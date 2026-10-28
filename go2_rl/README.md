# go2_rl — Go2 / 自建狗 强化学习（MJX + Brax PPO）

用 `mujoco_playground`（MJX）定义环境、Brax 的 PPO 训练，再导出 ONNX 部署。
本目录只保留**源码 + 策略**，不含训练日志/数据。

## 内容
```
scripts/
  export_go2_onnx.py        checkpoint → ONNX（手搓计算图，无需 TF）
  onnx_go2_viewer.py        walk
  onnx_go2_getup_viewer.py  起身
  onnx_go2_run_viewer.py    跑/蹦
  onnx_go2_standjump_viewer.py  立定跳越墙
  onnx_go2_switch_viewer.py walk↔getup 切换
  onnx_mydog_stand_viewer.py / onnx_mydog_walk_viewer.py  自建狗
  mirror_mydog_legs.py / mirror_mydog_meshes.py / mydog_symmetry_check.py  自建狗镜像对称
envs/
  go2/    自写环境：joystick(走) / getup / jump(run) / stand_jump + xmls + randomize
  mydog/  自建狗：stand / walk + mydog_simple.xml + randomize
policies/ 导出的 ONNX（walk/getup/getup_dr/run/stand_jump + mydog stand/walk 等）
```

## 前置
1. 安装 `mujoco_playground v0.1.0`（支持 Python 3.10）与 `brax`、`jax`、`mujoco-mjx`（版本配套）。
2. 把 `envs/go2/`、`envs/mydog/` 放进 `mujoco_playground/_src/locomotion/` 并注册（见 `envs/INTEGRATION.md`）。
3. Go2 模型来自 `mujoco_menagerie`（`unitree_go2`），放到 playground 期望的 `external_deps` 路径。

## 训练（示意）
```bash
cd <mujoco_playground 所在目录>
python3 -u learning/train_jax_ppo.py --env_name=Go2JoystickFlatTerrain \
  --num_timesteps=100000000 --num_envs=2048 --num_evals=10 \
  --domain_randomization=True --use_tb=True --num_videos=0 --suffix=your_run
```
环境名：`Go2JoystickFlatTerrain` / `Go2Getup` / `Go2JumpFlatTerrain` / `Go2StandJumpFlatTerrain` / `MyDogStandFlatTerrain` / `MyDogWalkFlatTerrain`。

## 导出 / 查看 ONNX
```bash
sed 's/Go2JoystickFlatTerrain/Go2JoystickFlatTerrain/' scripts/export_go2_onnx.py > /tmp/export.py
python3 /tmp/export.py --ckpt <run>/checkpoints --out policies/your.onnx
python3 scripts/onnx_go2_viewer.py --vx 0.6
```

## 观测/动作（概要）
- Go2 walk：obs 48（本体感知）；getup：obs 42（相对当前动作）；standjump：obs 48。
- 自建狗 stand：obs 30；walk：obs 35（含步态时钟）；动作均 8 维关节增量，env 内 PD。

## 已知
- **MJX↔MuJoCo/真机 gap** 明显；训练环境里的策略不一定直接迁移。
- 自建狗 “四条腿均匀 trot” 未完全解决（见笔记）。网格资产未包含。
