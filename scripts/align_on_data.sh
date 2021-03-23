. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
################################################################
# Align external GMM models on current data

gmm_dir=/NASdata/pc_backup/jiayan/close_talk_manderin/exp/tri14
train_set=train
lm=/NASdata/pc_backup/jiayan/close_talk_manderin/data/lang

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" data/${train_set}_sp $lm $gmm_dir new_ali || exit 1