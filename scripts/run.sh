#!/bin/bash

trn_set=$(pwd)/../data/train
dev_set=$(pwd)/../data/dev
tst_set=$(pwd)/../data/test

nj=20
stage=3
gmm_stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  local/prepare_all.sh ${trn_set} ${dev_set} ${tst_set} || exit 1;
fi

# GMM
if [ $stage -le 2 ]; then
  local/run_gmm.sh --nj $nj --stage $gmm_stage
fi

# tdnn
if [ $stage -le 3 ]; then
  local/nnet3/run_tdnn.sh --nj $nj
fi

local/show_results.sh

exit 0;
