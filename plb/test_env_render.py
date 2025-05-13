import argparse
import random
import numpy as np
import torch

from plb.envs import make
from plb.algorithms.logger import Logger

from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn
import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN

from plb.optimizer.optim import Optimizer, Adam, Momentum
from plb.engine.taichi_env import TaichiEnv
from plb.config.utils import make_cls_config
RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default=DIFF_ALGOS + RL_ALGOS)
    parser.add_argument("--env_name", type=str, default="AgF1-v1")
    parser.add_argument("--path", type=str, default='./tmp')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=50)
    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

    args=parser.parse_args()

    return args

def main():
    args = get_args()
    if args.num_steps is None:
        if args.algo in DIFF_ALGOS:
            args.num_steps = 50 * 200
        else:
            args.num_steps = 500000

    logger = Logger(args.path)
    set_random_seed(args.seed)

    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)

    import cv2
    env.reset()
    tai_env = env.unwrapped.taichi_env
    action_dim = tai_env.primitives.action_dim
    horizon = args.horizon
    action = np.random.uniform(-.1, 1, size=(1, action_dim))
    tai_env.step(action)
    poses = [
        ((0.5, 0.33, -2.0), (0.0, 0.0)),
    ]

    images = []
    for pos, rot in poses:
        env.taichi_env.renderer.set_camera_pose(camera_pos=pos, camera_rot=rot)

    tai_env.render()
    img = tai_env.render(mode='rgb_array')
    print(img.min(), img.max())
    idx = 1
    cv2.imwrite(f"{args.path}/{idx:04d}.png", img[..., ::-1].astype(np.uint8))


if __name__ == '__main__':
    main()
