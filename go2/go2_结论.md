# Go2 站立 / 抗推 实验结论

> 环境：Ubuntu + mujoco(mujoco.rollout) + scipy；模型 `mujoco_menagerie/unitree_go2`。
> 相关脚本：`go2_stand.py`（PD/MPC/LQR 站立）、`go2_push_test.py`（推扰+恢复）、
> `go2_pd_sweep.py`/`go2_pd_fine.py`、`go2_lqr_sweep.py`/`go2_lqr_fine.py`、`go2_weight_sweep.py`。

## 指标定义
- **恢复步数**：推后回到稳态(z、姿态)的步数；有下界 `HOLD_STEPS=30` → 最快 29（≈立即）。
- **生存分 survival/16**：方向 {+x,-x,+y,-y} × 力 {300,600N} × 时长 {0.1,0.3s} 中"能恢复不倒"的个数。
- ⚠️ **survive 对力非单调**（同配置 150/200/250 倒、300 反而恢复）→ **"临界推力"二分法不可用**。

## 主要结论
1. **采样 MPC 无益**（对该任务）：随机采样、CEM、LQR 底 + 对齐代价，都没比其基底好；300N 时甚至更差（204 步/FALL）。
   - 根因：MPC 在视界内是**开环 + 力矩恒定 + 视界太短(H=15, 0.03s)**，且代价(早期含 `TARGET_Z=0.27` 偏差)与"抗推"不对齐。
   - 早期 bug：候选里没有"纯 base"，已加 `delta[0]=0`（仍不够，见上）。
2. **LQR**：全状态反馈、**局部（小信号）最优**；但
   - 闭环 `max|eig|≈0.999`（几乎贴单位圆，轻阻尼）；
   - 原因是**不可控的中性模态**（浮动基 + 脚踩地 → 基底水平平移关节电机改不了，特征值钉在 ≈1），
     **任何 Q/R / 速度权重都压不下来** → **细调 Q/R 对鲁棒性 flat（最好 4/16）**；
   - 大冲击下出线性区即失效。**不是公式/矩阵写错。**
3. **PD**（关节位置伺服）：抗推**主要由增益决定**
   - `KD` 越大越抗推（过大不稳，如 KD=64/300 反而倒）；`KP` 高 + `KD` 高（约 2:1）最优；
   - 甜区 **`KP≈300, KD≈150` → 13/16**；机制：高 KP 使力矩常饱和(近似 bang-bang)、高 KD 强阻尼 → 机身像刚性块，靠宽支撑面抗冲击。
   - **强于 LQR**（4/16）和旧默认（60/3 → 2/16）。
4. **方向/时长强依赖**：`+x`(前)最强、`-x`(后)与 `±y`(侧)最弱；**短冲量(0.1s) ≫ 持续力(0.3s)** 好扛。
5. **重力前馈无效**：浮动基用 `τ += qfrc_bias[关节]` 补偿站姿不成立（机身重力由接触承担），反而更差。

## 关键数字
| 控制器 | 100N 恢复 | 多方向 survival/16 |
|---|---|---|
| PD 旧默认 60/3 | 238 | 2 |
| PD 140/60 | 29 | 9 |
| **PD 300/150（现行默认）** | 29 | **13** |
| LQR（最好配置） | 29 | 4 |
| MPC(LQR底) 300N | 204/FALL | — |

## 现行设置
- `go2_stand.py`：**`KP=300.0, KD=150.0`**；`PD_FF=False`（重力前馈开关，默认关）。
- `--controller pd|mpc|lqr`、`--mpc-base pd|lqr`。

## 局限 / 待办
- 结论仅限**仿真**：单一作用点（base）、xfrc 推力、未测斜向/组合、随机/共振现象明显。
- 想进一步提升 LQR 类：需处理不可控模态 → **主动迈步/接触切换（步态/全身控制 MPC/WBC/RL）**，非"站着硬扛"。
- 采样 MPC 若要用，应：候选含纯 base、视界拉长、控制带时间结构(非恒定)、代价对齐恢复目标。

## 怎么运行 / 验证
模型路径已改成不依赖当前目录，从哪运行都行。

### 站稳（看窗口）
```bash
cd ~/mujoco_ws/go2
python3 go2_stand.py                                   # 默认 PD(KP300/KD150)
python3 go2_stand.py --controller lqr                  # 纯 LQR
python3 go2_stand.py --controller mpc --mpc-base lqr   # LQR 底的 MPC
python3 go2_stand.py --headless 1500                   # 无窗口看数值
```
预期：`upright_err=0.000`、`base_z` 稳定不塌。
窗口：左键转视角 / 滚轮缩放；双击身体后 **Ctrl+右键拖动**施力。

### 抗推（自动打分）
```bash
python3 go2_push_test.py --controller pd  --headless --force 100   # 打印恢复步数
python3 go2_push_test.py --controller pd  --force 100              # 窗口里每秒自动推
python3 go2_push_test.py --controller lqr --headless --force 300
```
预期（100N）：当前默认 PD ≈ `29 steps`（旧默认 60/3 = 238）。

### 复现扫描
```bash
python3 go2_pd_sweep.py --force 100   # KP×KD 网格（恢复步数）
python3 go2_pd_fine.py                # KP×KD 临界推力（注意非单调）
python3 go2_lqr_fine.py               # LQR Q/R，多方向 survival/16
python3 go2_weight_sweep.py           # MPC 的 W_Z×W_UP
```
结果 CSV 在 `/tmp/opencode/*.csv`，终端末尾有表格。

### 判定标准
- 站立：`base_z` 稳、`upright_err≈0`；
- 抗推：**恢复步数**（29 为下界=最好）或 **survival 分数**（越高越抗推）；
- 基准：`PD300/150 → 13/16`、`LQR → 4/16`、`MPC → 更差`。

依赖：`mujoco`（`python3 -m pip install --user mujoco`）、`scipy`、`mujoco_menagerie`。
