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
            if p!=-1:
                new_key = ' '.join([x.strip().upper() for x in l[:p].strip().split()])
            if p != -1 and new_key != '' and not new_key[-1].isdigit() and len(new_key.split()) < 5:
                if current_key != '' and len(current_value) != 0:
                    current_res.update({current_key: current_value})
                current_key = new_key
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
