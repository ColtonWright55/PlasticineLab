from setuptools import setup

install_requires = ['scipy', 'numpy', 'torch', 'opencv-python', 'tqdm', 'taichi==0.9.0', 'gym==0.21.0', 'tensorboard', 'yacs',
                     'matplotlib', 'descartes', 'shapely', 'natsort', 'torchvision', 'einops', 'alphashape', 'open3d', 'tensorboardX']

setup(name='plb',
      version='0.0.1',
      install_requires=install_requires,
      py_modules=['plb'],
      python_requires='>=3.7,<=3.9',
      )
