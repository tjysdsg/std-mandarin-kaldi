speakers = set()

with open('std_spk.txt') as f:
    for line in f:
        tokens = line.split()
        spk_id = tokens[0]
        speakers.add(spk_id)

trans = []
with open('trans.txt') as f:
    for line in f:
        tokens = line.split()
        trans_id = tokens[0]
        spk_id = trans_id[1:6]
        if spk_id in speakers:
            trans.append(line)

with open('trans_filtered.txt', 'w') as f:
    f.writelines(trans)
    
wavscp = []
with open('wav.scp') as f:
    for line in f:
        tokens = line.split()
        trans_id = tokens[0]
        spk_id = trans_id[1:6]
        if spk_id in speakers:
            wavscp.append(line)

with open('wav_filtered.scp', 'w') as f:
    f.writelines(wavscp)