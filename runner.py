from joblib import Parallel, delayed
import queue
import os
import glob

N_GPU = 2

q = queue.Queue(maxsize=N_GPU)
for i in range(N_GPU):
    q.put(i)


def runner(x):
    gpu = q.get()
    cmd = "python3 train.py --config_file {} --cuda_device {}".format(x, gpu)
    os.system(cmd)
    q.put(gpu)


exps = set(line.strip() for line in open('runner.exclude'))
exps = set(glob.glob('./confs/*.yml')) - exps
Parallel(n_jobs=N_GPU, backend="threading")(delayed(runner)(exp) for exp in exps)
