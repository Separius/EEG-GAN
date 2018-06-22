from utils import simple_argparser
import pickle
import glob
import os
import subprocess
import scipy.io

default_params = {
    'generated_path': '',
    'is_dir': False,
    'frequency': 80
}


def call_matlab(data):
    scipy.io.savemat('tmp.mat', data)
    ans = subprocess.check_output(["matlab", "validator.matlab", "tmp.mat"]).strip()
    subprocess.check_output(["rm", "tmp.mat"])
    return ans


if __name__ == '__main__':
    params = simple_argparser(default_params)
    if params['is_dir']:
        best_accuracy = None
        best_model = None
        params['generator_path'] = os.path.join(params['generator_path'], 'network-snapshot-generator-*.pkl')
        for generator in glob.glob(params['generator_path']):
            data = pickle.load(open(generator, 'rb'))
            accuracy = float(call_matlab(data))
            if best_accuracy is None or accuracy < best_accuracy:
                best_accuracy = accuracy
                best_model = generator
        print('best model was {} with accuracy:'.format(best_model), best_accuracy)
    else:
        data = pickle.load(open(params['generator_path'], 'rb'))
        print('accuracy is:', call_matlab(data))
