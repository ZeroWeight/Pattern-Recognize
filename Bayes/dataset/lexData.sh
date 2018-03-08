# -*- sh -*-
# one file in sci.crypt is skipped b/c rainbow thinks it isn't text
DIR=/afs/csail.mit.edu/group/tml/jrennie/20news-bydate
rainbow -i --skip-header --istext-avoid-uuencode --prune-vocab-by-occur-count=2 --no-stoplist $DIR/20news-bydate-{train,test}/*
mkdir $DIR/rainbow
rainbow --print-matrix=sin > $DIR/rainbow/frequencies
cp ~/.rainbow/vocabulary $DIR/rainbow
mkdir $DIR/matlab
curpwd=`/bin/pwd`
cd $DIR/matlab
for type in train test; do
    echo ../rainbow2matlab.py $type/ $type < ../rainbow/frequencies
    ../rainbow2matlab.py $type/ $type < ../rainbow/frequencies
done
cd $curpwd
