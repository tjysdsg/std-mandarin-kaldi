ret = []
with open('spk_info.txt') as f:
    for line in f:
        dialect = ' '.join(line.split()[-2:])
        if dialect in ['he bei', 'bei jing', 'tian jin']:
            ret.append(line)

with open('std_spk.txt', 'w') as f:
    f.writelines(ret)
