. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


nj=40
stage=0
. ./utils/parse_options.sh
x_root=combine_025M
x=combine_025M

if [ $stage -le 0 ]; then

utils/fix_data_dir.sh data/$x || exit 1;

steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf \
      --nj $nj data/$x exp/make_mfcc/ mfcc_hires

steps/compute_cmvn_stats.sh data/$x exp/make_mfcc mfcc
    
utils/data/limit_feature_dim.sh 0:39 data/$x data/${x}_nopitch

steps/compute_cmvn_stats.sh data/${x}_nopitch exp/make_mfcc mfcc

utils/fix_data_dir.sh data/$x || exit 1;


fi

if [ $stage -le 1 ]; then

steps/nnet3/align.sh --nj $nj --cmd "$train_cmd" --use-gpu false \
                  data/${x}_nopitch data/lang_aishell exp/nnet3/tdnn_sp/ exp/nnet3_$x

fi


if [ $stage -le 2 ]; then

./steps/get_train_ctm.sh data/$x data/lang_aishell exp/nnet3_$x

fi

exit 0

if [ $stage -le 3 ]; then

python organize_with_ctm_context.py exp/tri2b_snips_raw_$x_root/ctm data/$x/wav.scp audio_heysnips_context/$x_root/

fi

exit 0

if [ $stage -le 1 ]; then

ivectors_dir=exp/ivectors/ivectors_${x}
steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
             data/$x exp/chain_cleaned/tdnn_1d_sp_online/ivector_extractor $ivectors_dir


fi

if [ $stage -le 2 ]; then

decode_nj=16
iter_opts=
graph_dir=exp/chain_cleaned/tdnn_1d_sp/graph_tgsmall/
steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
    --online-ivector-dir exp/ivectors/ivectors_${x} \
    $graph_dir data/${x} exp/decode/decode_mia || exit 1

fi

exit 0

if [ $stage -le 3 ]; then

for x in kw_ch kw_en code_switch; do
ivectors_dir=exp/ivectors/ivectors_${x}
data_dir=data_ae/${x}
post_dir=exp/post_mono/post_${x}
local/get_phone_post.sh --nj $nj --cmd run.pl \
                        --remove-word-position-dependency true --online-ivector-dir $ivectors_dir \
                        exp/chain_cleaned/tree_sp exp/chain_cleaned/tdnn_1d_sp data/lang $data_dir $post_dir
done
fi


