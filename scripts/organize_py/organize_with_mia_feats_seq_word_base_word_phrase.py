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

noise_list = {'noise': [i.strip() for i in open('/NASdata/qinxy/ASV/e2e_clean/data/noise/musan_noise_list')],
              'music': [i.strip() for i in open('/NASdata/qinxy/ASV/e2e_clean/data/noise/musan_music_list')],
              'babb': [i.strip() for i in open('/NASdata/qinxy/ASV/e2e_clean/data/noise/musan_babble_list')]}

def augmentation(o_sig):
    speech_len = o_sig.shape[0]
    augment_type = random.choices(population=['music', 'noise', 'speech'], weights=[0.3, 0.3, 0.4])[0]

    if augment_type == 'clean':
        return o_sig
    elif augment_type == 'music':
        n_file = random.choice(noise_list["music"])
        n_sig, sr = sf.read(n_file)
        if o_sig.shape[0] > n_sig.shape[0]:
            rep_time = int(o_sig.shape[0] / n_sig.shape[0]) + 1
            n_sig =  np.concatenate([n_sig for _ in range(rep_time)])
        idx = random.randrange(0, n_sig.shape[0] - speech_len)
        n_sig = n_sig[idx:idx + speech_len]
    elif augment_type == 'noise':
        n_file = random.choice(noise_list["noise"])
        n_sig, sr = sf.read(n_file)
        if o_sig.shape[0] > n_sig.shape[0]:
            rep_time = int(o_sig.shape[0] / n_sig.shape[0]) + 1
            n_sig =  np.concatenate([n_sig for _ in range(rep_time)])
        idx = random.randrange(0, n_sig.shape[0] - speech_len)
        n_sig = n_sig[idx:idx + speech_len]
    elif augment_type == 'speech':
        n_file = random.choice(noise_list["babb"])
        n_sig, sr = sf.read(n_file)
        if o_sig.shape[0] > n_sig.shape[0]:
            rep_time = int(o_sig.shape[0] / n_sig.shape[0]) + 1
            n_sig =  np.concatenate([n_sig for _ in range(rep_time)])
        idx = random.randrange(0, n_sig.shape[0] - speech_len)
        n_sig = n_sig[idx:idx + speech_len]

    alpha = random.uniform(0.5, 1)
    o_sig = alpha * o_sig + (1 - alpha) * n_sig
    return o_sig

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
    tmid = items[2]
    # print(items)

    word_save_dir = save_dir + '/' + word + '_'
    if os.path.exists(word_save_dir + utt_id + ".npy"):
        return 0

    sig, sr = read_signal(utt_id)
    if len(sig.shape) > 1:
        sig = sig[:, 0]
    nsig = sig

    feats = psf.logfbank(nsig, sr, nfilt=40)
    
    tmid = float(tmid)
    fea_mid = sig_index_to_feat_index(tmid)
    # while fea_beg + 40 < fea_end:
    #     cur_feat = feats[fea_beg:fea_beg+40]
    #     fea_beg += 1
    win = 140
    while len(feats[fea_mid - win - 1: fea_mid]) < 140:
        win += 1
    # print(items, fea_mid - win, fea_mid + win)
    feats = feats[fea_mid - win - 1 : fea_mid]
    feats = feats[0:140]
    save_feat(feats, word, utt_id)
    
    return 1

def get_words_list(ctm_file):
    content_dict = {}
    word_segments = []
    print("get_words_list")
    
    for index, items in tqdm(read_file_gen(ctm_file)):
        if items[0] not in content_dict.keys():
           content_dict[items[0]] = {}

        if items[4] in content_dict[items[0]].keys():
            content_dict[items[0]][items[4] + "a"] = items
        else:
            content_dict[items[0]][items[4]] = items

    for utt_id in content_dict.keys():
        content = content_dict[utt_id]
        word_segments.append([utt_id, "nihaomiya",  float(content["雅"][2]) + float(content["雅"][3]) ])

    return word_segments

def extract_words(ctm_file):
    process_num = 40
    word_segments = get_words_list(ctm_file)
    print("word_segments:", len(word_segments))
    # print(word_segments)
    with mp.Pool(process_num) as p:
        frames = list(tqdm(p.imap(cut_word_and_save, word_segments), total=len(word_segments)))

    print(sum(frames), len(frames))
    print("Done.")

for ctm_file in ctm_files:
    extract_words(ctm_file)


