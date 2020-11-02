speakers = set()

with open('std_spk.txt') as f:
    for line in f:
        spk_id = line.split()[0]
        speakers.add(spk_id)

trans = []
with open('trans.txt') as f:
    for line in f:
        tokens = line.split()
        trans_id = tokens[0]
        spk_id = trans_id[:7]
        if spk_id in speakers:
            trans.append(line)

with open('trans_filtered.txt', 'w') as f:
    f.writelines(trans)
    
wavscp = []
with open('train.scp') as f1, open('test.scp') as f2, open('dev.scp') as f3:
    lines = f1.readlines() + f2.readlines() + f3.readlines()
    for line in lines:
        tokens = line.split()
        trans_id = tokens[0]
        spk_id = trans_id[:7]
        if spk_id in speakers:
            wavscp.append(line)

with open('wav_filtered.scp', 'w') as f:
    f.writelines(wavscp)