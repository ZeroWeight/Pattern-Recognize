#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

case ${DATASET} in
  name_card)
    TRAIN_IMDB="name_card_real_trainval"
    TEST_IMDB="name_card_real_test"
    STEPSIZE="[50000]"
    ITERS=200000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  name_card_fake)
    TRAIN_IMDB="name_card_fake_train"
    TEST_IMDB="name_card_fake_train"
    STEPSIZE="[50000]"
    ITERS=500000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;

  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.pth
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python3.5 ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.pth \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE}
fi

#./experiments/scripts/test_faster_rcnn.sh $@
