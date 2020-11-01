ret = []
with open('spk_info.txt') as f:
    for line in f:
        area = line.split()[-1]
        if area == 'North':
            ret.append(line)

with open('north_spk.txt', 'w') as f:
    f.writelines(ret)
