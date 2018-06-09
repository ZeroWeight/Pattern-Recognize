while true
do
  ID=0
  nvidia-smi | grep 'MiB /' | cat | while read LINE
  do
    TEMP=${LINE%%MiB*}
    USED=${TEMP##*| }
    TEMP=${LINE%MiB*}
    TOT=${TEMP##*/ }
    FREE=`expr $TOT - $USED`
    if [ $FREE -ge $1 ]; then
      ./experiments/scripts/train_faster_rcnn.sh $ID name_card_fake res101 
      kill $$
    fi
    ID=`expr $ID + 1`
  done
  sleep 1
done
