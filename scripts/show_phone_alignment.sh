. path.sh

ali_path=/NASdata/pc_backup/jiayan/close_talk_manderin/exp/tri14_sp_ali
stage=1

mkdir -p alignments
if [ $stage -le 1 ]; then
    for f in $ali_path/ali.*.gz; do
        filename="${f##*/}"
        ctmfile=alignments/$filename.ctm
        if [ ! -f $ctmfile ]; then
            echo "$f -> $ctmfile"
            alignfile=$ali_path/$filename.ali
            zcat $f > $alignfile
            ali-to-phones --ctm-output $ali_path/final.mdl ark:$alignfile $ctmfile
        else
            echo "Skipping $f because $ctmfile already exists"
        fi
    done
fi

if [ $stage -le 2 ]; then
    cat alignments/*ctm > phone_ctm.ctm
    python ali_to_phone.py $ali_path/phones.txt phone_ctm.ctm phone_ctm.txt
fi
