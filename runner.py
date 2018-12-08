import os
import queue
from joblib import Parallel, delayed

N_GPU = 1

q = queue.Queue(maxsize=N_GPU)
for i in range(N_GPU):
    q.put(i)


def runner(x):
    gpu = q.get()
    cmd = "python3 train.py --config_file ./confs/{} --cuda_device {}".format(x, gpu)
    os.system(cmd)
    q.put(gpu)


Parallel(n_jobs=N_GPU, backend="threading")(delayed(runner)(line.strip()) for line in open('runner.cfg'))
