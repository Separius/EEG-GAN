import pickle
import glob

reports = dict()
for f_name in glob.glob('./data/reports/**/*.txt', recursive=True):
    with open(f_name, encoding='iso-8859-1') as f:
        current_key = ''
        current_value = list()
        current_res = dict()
        for line in f:
            l = line.encode('utf-8').strip().decode('utf-8')
            p = l.find(':')
            if p != -1:  # TODO and it does not end with a digit: at 13:30 ... and len(l[:p].split()) < 4?(heuristic)
                if current_key != '' and len(current_value) != 0:
                    current_res.update({current_key: current_value})
                current_key = l[:p].strip()
                v = l[p + 1:].strip()
                if v != '':
                    current_value = [v]
                else:
                    current_value = []
            elif l != '':
                current_value.append(l)
        if current_key != '' and len(current_value) != 0:
            current_res.update({current_key: current_value})
        reports.update({f_name.split('/')[-1][:-4]: current_res})
