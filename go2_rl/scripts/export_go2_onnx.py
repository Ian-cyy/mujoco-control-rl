#!/usr/bin/env python3
# 把 Go2 的 Brax PPO checkpoint 里的策略(MLP)导出为 ONNX (不需要 TensorFlow)
# 网络: (obs-mean)/std -> Dense512-swish -> Dense256-swish -> Dense128-swish -> Dense24 -> 取前12 -> tanh
import argparse, functools, os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as rt
import jax, jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training.checkpoint import load as ckpt_load
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from etils import epath

ENV = "Go2JoystickFlatTerrain"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def resample(path):
    p = epath.Path(path).resolve()          # orbax 需要绝对路径
    if p.is_dir():
        subs = [d for d in p.glob("*") if d.is_dir() and d.name.isdigit()]
        if subs:
            return str(sorted(subs, key=lambda x: int(x.name))[-1])
        return str(p)                        # 传入的就是某个步数目录 -> 用它本身
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoints 目录或具体步数子目录")
    ap.add_argument("--out", default=os.path.join(ROOT, "policies", "go2_policy.onnx"))
    args = ap.parse_args()

    env = registry.load(ENV)
    obs_size = int(env.observation_size["state"][0])
    act_size = int(env.action_size)
    ppo_params = locomotion_params.brax_ppo_config(ENV)

    ck = resample(args.ckpt); print("checkpoint:", ck)
    raw = ckpt_load(ck)
    normalizer, params = raw[0], raw[1]
    mean = np.asarray(normalizer.mean["state"], dtype=np.float32)
    std = np.asarray(normalizer.std["state"], dtype=np.float32)
    mlp = params["params"]
    hidden = list(ppo_params.network_factory.policy_hidden_layer_sizes)

    Ws, Bs = [], []
    for i in range(len(hidden) + 1):
        layer = mlp[f"hidden_{i}"]
        Ws.append(np.asarray(layer["kernel"], dtype=np.float32))
        Bs.append(np.asarray(layer["bias"], dtype=np.float32))
        print(f"hidden_{i}: W{Ws[-1].shape} b{Bs[-1].shape}")

    nodes, inits = [], []
    inits.append(numpy_helper.from_array(mean, "mean"))
    inits.append(numpy_helper.from_array(std, "std"))
    nodes += [helper.make_node("Sub", ["obs", "mean"], ["xn0"]),
              helper.make_node("Div", ["xn0", "std"], ["xn"])]
    prev = "xn"
    for i in range(len(Ws)):
        inits += [numpy_helper.from_array(Ws[i], f"W{i}"), numpy_helper.from_array(Bs[i], f"b{i}")]
        nodes += [helper.make_node("MatMul", [prev, f"W{i}"], [f"mm{i}"]),
                  helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"z{i}"])]
        if i < len(Ws) - 1:
            nodes += [helper.make_node("Sigmoid", [f"z{i}"], [f"sg{i}"]),
                      helper.make_node("Mul", [f"z{i}", f"sg{i}"], [f"sw{i}"])]
            prev = f"sw{i}"
        else:
            prev = f"z{i}"
    nodes += [helper.make_node("Split", [prev], ["loc", "scale"], axis=1, split=[act_size, act_size]),
              helper.make_node("Tanh", ["loc"], ["continuous_actions"])]

    graph = helper.make_graph(nodes, "go2_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_size])],
        [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, act_size])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, args.out); print("saved:", args.out)

    # 数值比对
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory,
        preprocess_observations_fn=running_statistics.normalize)
    ppo_network = network_factory(env.observation_size, act_size)
    infer = jax.jit(ppo_networks.make_inference_fn(ppo_network)((normalizer, params), deterministic=True))
    obs = np.random.randn(1, obs_size).astype(np.float32)
    jx = np.asarray(infer({"state": jp.array(obs)}, jax.random.PRNGKey(0))[0])[0]
    on = rt.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(
        ["continuous_actions"], {"obs": obs})[0][0]
    print("max abs diff:", float(np.max(np.abs(jx - on))))


if __name__ == "__main__":
    main()
