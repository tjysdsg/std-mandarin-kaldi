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
import python_speech_features as psf

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
    random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    np.save(word_save_dir + utt_id + '_' + random_id, feat)

def save_signal(signal, sr, word, utt_id):
    word_save_dir = save_dir + '/' + word + '_'
    # random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    sf.write(word_save_dir + utt_id + '.wav', signal, sr)

def sig_index_to_feat_index(sig_beg, sig_end):
    fea_beg = max(0, math.floor(((sig_beg - 0.015) / 0.01) + 1))
    fea_end = max(0, math.floor(((sig_end - 0.015) / 0.01) + 1))
    return fea_beg, fea_end

def cut_word_and_save(items):
    utt_id = items[0]
    word = items[1]
    tbegin = items[2]
    tend = items[3]

    sig, sr = read_signal(utt_id)
    feats = psf.logfbank(sig, sr, nfilt=40)
    
    # tbeg = max(round(sr * float(tbegin) - beg_context), 0) 
    # tend = round(sr * (float(tbegin) + float(tdur))) + end_context
    # save_signal(sig[tbeg:tend], sr, word, utt_id)
    # tbeg = min(0, float(tbegin) - 0.05)
    # tend = float(tbegin) + float(tdur) + 0.05
    tbeg = float(tbegin)
    tend =  float(tend)
    fea_beg, fea_end = sig_index_to_feat_index(tbeg, tend)
    # while fea_beg + 40 < fea_end:
    #     cur_feat = feats[fea_beg:fea_beg+40]
    #     fea_beg += 1
    save_feat(feats[fea_beg:fea_end], word, utt_id)
    
    return fea_end - fea_beg

def get_words_list(ctm_file):
    content_dict = {}
    word_segments = []
    print("get_words_list")
    
    for index, items in tqdm(read_file_gen(ctm_file)):
        if items[0] not in content_dict.keys():
            content_dict[items[0]] = {}
        # print(items)
        content_dict[items[0]][items[4]] = items

    for utt_id in content_dict.keys():
        content = content_dict[utt_id]
        #print(content)
        # word_segments.append([utt_id, "HEY", float(content["179"][2]), float(content["164"][2]) + float(content["164"][3]) - float(content["179"][2])])
        #if "63" in content.keys():
        #    word_segments.append([utt_id, "ni",  float(content["114"][2]), float(content["114"][3]) + float(content["63"][3])  ])
        #else:
        #    word_segments.append([utt_id, "ni",  float(content["114"][2]), float(content["114"][3]) + float(content["62"][3])  ])
        word_segments.append([utt_id, "ni_hao", float(content["218"][2]),  float(content["178"][2])  + float(content["178"][3])  ])
        word_segments.append([utt_id, "hao_mi", float(content["349"][2]),  float(content["240"][2])  + float(content["240"][3])  ])
        word_segments.append([utt_id, "mi_ya",  float(content["240"][2]), float(content["384"][2]) + float(content["384"][3])  ])
        #word_segments.append([utt_id, "ya",  float(content["199"][2]), float(content["199"][3]) + float(content["6"][3])   ])

    return word_segments

def extract_words(ctm_file):
    process_num = 40
    word_segments = get_words_list(ctm_file)
    print("word_segments:", len(word_segments))
    print(word_segments)
    with mp.Pool(process_num) as p:
        frames = list(tqdm(p.imap(cut_word_and_save, word_segments), total=len(word_segments)))

    print(sum(frames), len(frames))
    print("Done.")

for ctm_file in ctm_files:
    extract_words(ctm_file)


