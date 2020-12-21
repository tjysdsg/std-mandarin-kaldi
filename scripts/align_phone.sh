
. ./path.sh

ali_path=$1
stage=0

if [ $stage -le 0 ]; then

for f in `ls $ali_path | grep gz`; do
    gunzip -k $ali_path/$f
done

fi

mkdir -p $ali_path/tmp

if [ $stage -le 1 ]; then

for f in `ls $ali_path | grep ali | grep -v gz`; do
    ali-to-phones --ctm-output $ali_path/final.mdl ark:$ali_path/$f $ali_path/tmp/$f.ctm
done

fi

if [ $stage -le 2 ]; then

cat $ali_path/tmp/*ctm >$ali_path/phone_ctm
rm $ali_path/tmp/*ctm

fi


if [ $stage -le 3 ]; then

    python ali_to_phone.py $ali_path/phones.txt $ali_path/phone_ctm $ali_path/phone_ctm_phone

fi

