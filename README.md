# PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics

## Usage
 - Install `python3 -m pip install -e .`
 - Run `python3 -m plb.algorithms.solve [algo] [env_name] --path [output-dir]`. It will run algorithms `algo` for environment `env-name` and store results in `output-dir`. For example
    `python3 -m plb.algorithms.solve action Move-v1 --path output` will run call an Adam optimizer to optimize an action sequence in environment `Move-v1`




## Installation / Upstream repo edits

As of 5/1/2025, it's working on Windows with Python 3.8.10 in a .venv after running `.\.venv\Scripts\activate`, `pip install -e .`, and finally `python -m plb.algorithms.solve --algo action --env_name Move-v1 --path output`.

### Upstream repo edits

This was a pain to get working. [Taichi](https://github.com/taichi-dev/taichi) has made many API changes since this paper came out, so I had to hunt down the correct Taichi version. There were other dependencies not originally in the setup file, like [Open3D](https://github.com/isl-org/Open3D) & [tensorboardX](https://github.com/lanpa/tensorboardX). Also, we are using gym==0.21.0 and not gym--0.26.2 / gymnasium.

You must set the GPU memory size to match your GPU in `plb\engine\taichi_env.py`, on the line `ti.init(arch=ti.gpu, debug=False, device_memory_GB=4)`, or replace it with `ti.init(arch=ti.gpu, debug=False, device_memory_fraction=0.9)`


`toy\legacy.py` Works with taichi==0.7.13 to taichi==0.7.25, but breaks with taichi==0.7.32 and up. Meanwhile, `python -m plb.algorithms.solve --algo action --env_name Move-v1 --path output` cannot work on version taichi==0.7.13 or taichi==0.7.25, because you get the `AttributeError: module 'taichi' has no attribute 'ad'`. Which goes away if taichi==0.9.0. With taichi==0.9.0 and gym==0.26.2, you get the error:

```
  File "C:\Users\colto\Documents\GitHub\PlasticineLab\plb\algorithms\solve.py", line 76, in <module>
    main()
  File "C:\Users\colto\Documents\GitHub\PlasticineLab\plb\algorithms\solve.py", line 59, in main
    env.seed(args.seed)
  File "C:\Users\colto\Documents\GitHub\PlasticineLab\.venv\lib\site-packages\gym\core.py", line 241, in __getattr__
    return getattr(self.env, name)
  File "C:\Users\colto\Documents\GitHub\PlasticineLab\.venv\lib\site-packages\gym\core.py", line 241, in __getattr__
    return getattr(self.env, name)
  File "C:\Users\colto\Documents\GitHub\PlasticineLab\.venv\lib\site-packages\gym\core.py", line 241, in __getattr__
    return getattr(self.env, name)
AttributeError: 'PlasticineEnv' object has no attribute 'seed'
```

This was a gym error, so I replaced gym==0.26.2 with gym==0.21.0, and it seems fixed. I put all the dependencies in setup.py, so should be fast to install now.

*NOTE*: I cannot run this on my laptop; CUDA driver not found.





## Performance

Maybe we could use this to see what's going on:

Overview

Taichi includes a collection of profiling tools to help with code debugging and optimization. These tools collect hardware and Taichi-related information to measure program performance and identify bottlenecks.

https://docs.taichi-lang.org/docs/profiler