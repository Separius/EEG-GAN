from joblib import Parallel, delayed
import queue
import os

N_GPU = 1

q = queue.Queue(maxsize=N_GPU)
for i in range(N_GPU):
    q.put(i)


# TODO change the config file template to be a product(var1.options, var2.options, ...)?
def runner(x):
    gpu = q.get()
    cmd = "python3 train.py --config_file ./confs/tuh1_512_wgan_5_otur_base.yml {} --cuda_device {}".format(x, gpu)
    os.system(cmd)
    q.put(gpu)


Parallel(n_jobs=N_GPU, backend="threading")(delayed(runner)(line.strip()) for line in open('runner.cfg'))
