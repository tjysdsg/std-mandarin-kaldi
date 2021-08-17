set -e 

aug_affix=_augment
do_adjust_speed=true
do_adjust_volume=true

. path.sh
. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 <train-dir>"
fi

# dir
aug_affix=_augment
train_dir=$1
train_final=${train_dir}${aug_affix}

data_src=
if $do_adjust_speed; then
  utils/data/perturb_data_dir_speed.sh 0.9 ${train_dir} ${train_dir}_speed0.9 || exit 1
  utils/data/perturb_data_dir_speed.sh 1.1 ${train_dir} ${train_dir}_speed1.1 || exit 1
  data_src="${train_dir}_speed0.9 ${train_dir}_speed1.1"
fi

local/combine_data.sh $train_final $train_dir $data_src

if $do_adjust_volume; then
  utils/data/perturb_data_dir_volume.sh $train_final
fi
