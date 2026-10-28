# mujoco-control-rl

在 MuJoCo / MJX 上做 **经典控制**（PID / LQR / MPC / VMC）与 **强化学习**（Brax PPO、Go2 四足、自建机械狗）的实战代码。

> 📖 配套学习笔记：https://ian-cyy.github.io/learning_notes/mujoco-notes/
> 笔记讲原理与结论，本仓库是可运行的源码。

## 目录
```
cartpole/    倒立摆：PID / 采样 MPC / 线性 QP-MPC(OSQP) + 抗扰对比
furuta/      旋转倒立摆：能量摆起 + LQR（可自恢复）
go2/         Go2 站立/抗推：PD / LQR / 采样MPC + 增益扫描 + 结论
go2_mpc/     Go2 单刚体 SRBD-MPC(CasADi+IPOPT) + VMC trot
go2_rl/      MJX + Brax PPO：Go2(walk/getup/run/standjump) + 自建狗(stand/walk)
  ├─ scripts/    导出 ONNX / ONNX viewer / 自建狗镜像对称工具
  ├─ envs/       自写的 go2 / mydog 环境（需 mujoco_playground）
  └─ policies/   训练导出的 ONNX 策略
```

## 环境（实测）
```
Ubuntu 22.04 · RTX 3060 Laptop 6G (CUDA 12.4) · Python 3.10
MuJoCo 3.14/3.15 · mujoco-mjx 3.15 · jax 0.6.2 · brax
mujoco_playground v0.1.0 · onnx/onnxruntime 1.23.2 · casadi 3.8.1 · osqp
```
基础依赖：`pip install mujoco scipy osqp casadi onnx onnxruntime`
官方模型库：`mujoco_menagerie`（Go2 等），放到与 `mujoco_ws` 同级或按各脚本路径调整。

## 快速开始
每个目录都可独立运行，例如：
```bash
# 倒立摆
python3 cartpole/mpc_cartpole_qp.py --compare
# 旋转倒立摆
python3 furuta/furuta_swingup.py
# Go2 站立/抗推
python3 go2/go2_stand.py                # PD(300/150)
python3 go2/go2_push_test.py --controller pd --headless --force 100
# Go2 SRBD-MPC
python3 go2_mpc/mpc_go2_srbd.py --gait stand --time 5
```
各目录细节见对应文件头注释与 `go2/go2_结论.md`。

## go2_rl（强化学习）
训练用 MJX + `mujoco_playground v0.1.0` + Brax PPO；`envs/` 里是自写环境，需按 `envs/INTEGRATION.md` 放入 `mujoco_playground` 并注册。
`scripts/` 含 ONNX 导出与轻量 viewer（onnxruntime + MuJoCo，秒开）。详见 `go2_rl/README.md`。

## 说明 / 限制
- 安装：`pip install -r requirements.txt`（经典控制 + ONNX viewer 可跑；RL 训练另装 jax/brax/mujoco_playground，见注释）。
- **自建狗**：`go2_rl/envs/mydog/xmls/` 已内置模型与 `.stl` 资产，配套 viewer 可直接跑。
- **Go2**：`go2_rl/envs/go2/xmls/assets/` 未内置 Unitree 网格（许可/体积），需从 `mujoco_menagerie/unitree_go2/assets` 复制（见该目录 README）。
- `go2_rl/envs/` 为自写环境，需按 `envs/INTEGRATION.md` 接入 `mujoco_playground` 才能训练。
- 自建狗实验中 **“四条腿稳定 trot”未完全成功**（RL 与手写均受限），结论如实记录，见笔记「自建狗」篇。
- 训练产物 `logs/`、`archive/` 不包含。

## License
Apache-2.0（见 LICENSE；Go2 网格资产来自 mujoco_menagerie，遵循其许可）
