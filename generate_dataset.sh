pushd aishell2
./run.sh || exit 1
popd

pushd magicdata
./run.sh || exit 1
popd

python create_dataset.py || exit 1
