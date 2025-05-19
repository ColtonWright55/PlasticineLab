import cv2
import taichi as ti
import torch

from plb.engine.taichi_env import TaichiEnv
import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np


from plb.config import load
cfg = load(r"C:\Users\colto\Documents\GitHub\PlasticineLab\plb\envs\env_configs\chopsticks.yml")

cfg.defrost()
cfg.PRIMITIVES[0]['shape'] = 'Gripper'
cfg.PRIMITIVES[0]['action']['dim'] = 7
del cfg.PRIMITIVES[0]['h']
del cfg.PRIMITIVES[0]['r']
cfg.PRIMITIVES[0]['size'] = (0.05, 0.2, 0.05)
#del cfg.PRIMITIVES[0]['init_gap']
cfg.PRIMITIVES[0]['init_gap'] = 0.2
cfg.PRIMITIVES[0]['minimal_gap'] = 0.15
cfg.PRIMITIVES[0]['friction'] = 50.
cfg.freeze()


ti.init(arch=ti.gpu, debug=False, fast_math=False, device_memory_fraction=0.9)
env = TaichiEnv(cfg, nn=False, loss=False)


env.initialize()
state = env.get_state()

p = state['state'][-1]
#p[7] = 0.3
p[:3] = [0.5, 0.2, 0.5]
p[7] = 0.4
state['state'] = (np.random.random((10000, 3)) * 0.2 + np.array([0.4, 0.0, 0.4]), *state['state'][1:5], )#p)
env.set_state(**state)
env.render('plt')

env.set_state(**state)
env.renderer.spp = 3

poses = [
    ((0.5, 0.33, -2.0), (0.0, 0.0)),
]
images = []
for pos, rot in poses:
    env.renderer.set_camera_pose(camera_pos=pos, camera_rot=rot)

env.render('plt')

images = []
for i in range(50):
    if i < 10:
        env.step([0,0,0, 0,0,0, 1])
    elif i >= 10 and i<=20:
        env.step([0,1,0, 0,0,0, 0])
    else:
        env.step([0,0,0, 0,1,0, 0])
    img2 = env.render('rgb_array')

    images.append(img2)
    cv2.imwrite(f"output/{i:04d}.png", img2[..., ::-1].astype(np.uint8))

