. path.sh

ali_path=exp/tri5a_ali
stage=2
for f in $ali_path/ali.*.gz; do
    if [ $stage -le 1 ]; then
        echo $f
        filename="${f##*/}"
        alignfile=$ali_path/$filename.ali
        zcat $f > $alignfile
        mkdir -p alignments
        ali-to-phones --ctm-output $ali_path/final.mdl ark:$alignfile alignments/$filename.ctm
    fi

    if [ $stage -le 2 ]; then
        cat alignments/*ctm > phone_ctm.ctm
        python ali_to_phone.py $ali_path/phones.txt phone_ctm.ctm phone_ctm.txt
    fi
done
