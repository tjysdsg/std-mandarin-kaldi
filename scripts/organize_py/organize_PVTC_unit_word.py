from file_tool import read_file_gen
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import soundfile as sf
import string
import random
import math
import sys
import os

ctm_files = [sys.argv[1]] # "/NASdata/zhangchx/kaldi/egs/thchs30/s5/exp/tri4b_dnn_mpe/decode_test_word_it3/ctm"]
wavscps = [sys.argv[2]] # "/NASdata/zhangchx/kaldi/egs/thchs30/s5/data/test/wav.scp"]
save_dir = sys.argv[3]
beg_context = 0 # 3600
end_context = 0 # 1200

def read_utt2wav():
    utt2wav = {}
    for wavscp in wavscps:
        curr_utt2wav = dict({line.split()[0]:line.split()[1] for line in open(wavscp)})
        # merge dict
        utt2wav = {**utt2wav, **curr_utt2wav}
    print("utt2wav:", len(list(utt2wav)))
    return utt2wav

utt2wav = read_utt2wav()

def read_signal(utt_id):
    utt_file_path = utt2wav[utt_id]
    signal, sr = sf.read(utt_file_path)
    return signal, sr

def save_feat(feat, word, utt_id):
    word_save_dir = save_dir + '/' + word + '_'
    np.save(word_save_dir + utt_id, feat)

def sig_index_to_feat_index(sig_beg):
    fea_beg = max(0, math.floor(((sig_beg - 0.015) / 0.01) + 1))
    return fea_beg

def cut_word_and_save(items):
    utt_id = items[0]
    word = items[1]
    tbegin = items[2]
    tend = items[3]

    word_save_dir = save_dir + '/' + word + '_'
    if os.path.exists(word_save_dir + utt_id + ".wav"):
        return 0
    sig, sr = read_signal(utt_id)
    if len(sig.shape) > 1:
        sig = sig[:, 0]
    nsig = sig
    keyword_sample = sig[int(sr*tbegin):int(sr*tend)]
    sf.write(word_save_dir+ utt_id + ".wav",keyword_sample,sr)
    return 1

def get_words_list(ctm_file):
    content_dict = {}
    word_segments = []
    print("get_words_list")
    
    for index, items in tqdm(read_file_gen(ctm_file)):
        if items[0] not in content_dict.keys():
           content_dict[items[0]] = []

        content_dict[items[0]].append(items)

    for utt_id in content_dict.keys():
        content = content_dict[utt_id]
        keyword = ''.join([i[4] for i in content[1:5]])
        word_segments.append([utt_id, keyword, float(content[1][2]),float(content[4][2])+float(content[4][3])])
        word_segments.append([utt_id, "filler",  float(content[6][2]),float(content[-1][2])+float(content[-1][3])])

    return word_segments

def write_textfile(word_segments):
    f1 = open("/Netdata/AudioData/PVTC/clean_PART2/text","w")
    for item in word_segments:
        f1.writelines(item[0] + " " + item[1] + " " + str(round(item[2],2)) + " " +str(round(item[3],2)) + "\n")
    f1.close()

def extract_words(ctm_file):
    process_num = 40
    word_segments = get_words_list(ctm_file)
    write_textfile(word_segments)
    print("word_segments:", len(word_segments))
    # print(word_segments)
    with mp.Pool(process_num) as p:
        frames = list(tqdm(p.imap(cut_word_and_save, word_segments), total=len(word_segments)))

    print(sum(frames), len(frames))
    print("Done.")

for ctm_file in ctm_files:
    extract_words(ctm_file)


