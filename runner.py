from joblib import Parallel, delayed
import queue
import os

N_GPU = 8

q = queue.Queue(maxsize=N_GPU)
for i in range(N_GPU):
    q.put(i)


def runner(x):
    gpu = q.get()
    cmd = "python train.py {} --cuda_device {}".format(x, gpu)
    os.system(cmd)
    q.put(gpu)


Parallel(n_jobs=N_GPU, backend="threading")(delayed(runner)(line.strip()) for line in open('runner.cfg'))
