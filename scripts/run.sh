#!/bin/bash

# Copyright 2017 Beijing Shell Shell Tech. Co. Ltd. (Authors: Hui Bu)
#           2017 Jiayu Du
#           2017 Xingyu Na
#           2017 Bengu Wu
#           2017 Hao Zheng
# Apache 2.0

. ./cmd.sh

trn_set=$(pwd)/../data/train
dev_set=$(pwd)/../data/dev
tst_set=$(pwd)/../data/test

nj=20
stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  local/prepare_all.sh ${trn_set} ${dev_set} ${tst_set} || exit 1;
fi

# MFCC plus pitch features.
if [ $stage -le 2 ]; then
  for x in train dev test; do
    steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf --cmd "$train_cmd" --nj $nj \
      data/$x exp/make_mfcc/$x mfcc || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  steps/train_mono.sh --cmd "$train_cmd" --nj 10 data/train data/lang exp/mono || exit 1;
  # Monophone decoding
  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 exp/mono/graph data/dev exp/mono/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 exp/mono/graph data/test exp/mono/decode_test

  # Get alignments from monophone system.
  steps/align_si.sh --cmd "$train_cmd" --nj 10 data/train data/lang exp/mono exp/mono_ali || exit 1;
fi

# tri1
if [ $stage -le 4 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" 2500 20000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

  # decode tri1
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 exp/tri1/graph data/dev exp/tri1/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 exp/tri1/graph data/test exp/tri1/decode_test
  
  # align tri1
  steps/align_si.sh --cmd "$train_cmd" --nj 10 data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
fi

# tri2 [delta+delta-deltas]
if [ $stage -le 5 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" 2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;
  
  # decode tri2
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 exp/tri2/graph data/dev exp/tri2/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 exp/tri2/graph data/test exp/tri2/decode_test

  # train and decode tri2b [LDA+MLLT]
  steps/align_si.sh --cmd "$train_cmd" --nj 10 data/train data/lang exp/tri2 exp/tri2_ali || exit 1;
fi

# tri3a, LDA+MLLT,
if [ $stage -le 6 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" 2500 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;
  
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config exp/tri3a/graph data/dev exp/tri3a/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config exp/tri3a/graph data/test exp/tri3a/decode_test

  # From now, we start building a more serious system (with SAT), and we'll
  # do the alignment with fMLLR.
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;
fi

# tri4a
if [ $stage -le 7 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 20000 data/train data/lang exp/tri3a_ali exp/tri4a || exit 1;
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config exp/tri4a/graph data/dev exp/tri4a/decode_dev
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config exp/tri4a/graph data/test exp/tri4a/decode_test
  
  steps/align_fmllr.sh  --cmd "$train_cmd" --nj 10 data/train data/lang exp/tri4a exp/tri4a_ali
fi

# Building a larger SAT system.

# tri5a
if [ $stage -le 7 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 3500 100000 data/train data/lang exp/tri4a_ali exp/tri5a || exit 1;

  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
     exp/tri5a/graph data/dev exp/tri5a/decode_dev || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
     exp/tri5a/graph data/test exp/tri5a/decode_test || exit 1;
  
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
    data/train data/lang exp/tri5a exp/tri5a_ali || exit 1;
fi

# nnet3
if [ $stage -le 8 ]; then
  local/nnet3/run_tdnn.sh
fi

# chain
# local/chain/run_tdnn.sh

# getting results (see RESULTS file)
for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

grep WER exp/nnet3/tdnn_sp/decode_test/cer_* | utils/best_wer.sh

exit 0;
