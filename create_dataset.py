import os
from sklearn.model_selection import train_test_split


datasets = ['aishell2', 'magicdata']
spk_id_from_utt = {'aishell2': lambda x: x[1:6], 'magicdata': lambda x: x[:7]}

wavscp = dict()
trans = dict()
utt2spk = dict()

for d in datasets:
    with open(os.path.join(d, 'wav_filtered.scp')) as f:
        for line in f:
            tokens = line.split()
            utt = tokens[0]
            wavscp[utt] = os.path.join('..', '..', d, tokens[1])
            spk_id = spk_id_from_utt[d](utt)
            utt2spk[utt] = spk_id

    with open(os.path.join(d, 'trans_filtered.txt')) as f:
        for line in f:
            tokens = line.split()
            trans[tokens[0]] = tokens[1]

utts = list(wavscp.keys())
train_utts, test_utts = train_test_split(utts, test_size=0.1, random_state=144)
train_utts, dev_utts = train_test_split(train_utts, test_size=0.1, random_state=144)

data_utts = [train_utts, test_utts, dev_utts]

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)
os.makedirs('data/dev', exist_ok=True)
for du, name in zip(data_utts, ['train', 'test', 'dev']):
    _wavscp = []
    _trans = []
    _utt2spk = []
    for utt in du:
        if utt in wavscp and utt in trans and utt in utt2spk:
            _wavscp.append('\t'.join([utt, wavscp[utt]]) + '\n')
            _trans.append('\t'.join([utt, trans[utt]]) + '\n')
            _utt2spk.append('\t'.join([utt, utt2spk[utt]]) + '\n')
    open(f'data/{name}/wav.scp', 'w').writelines(_wavscp)
    open(f'data/{name}/trans.txt', 'w').writelines(_trans)
    open(f'data/{name}/utt2spk', 'w').writelines(_utt2spk)
