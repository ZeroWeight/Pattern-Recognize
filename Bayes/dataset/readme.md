### http://qwone.com/~jason/20Newsgroups/

Below is a processed version of the 20news-bydate data set which is easy to read into Matlab/Octave as a sparse matrix:

- 20news-bydate-matlab.tgz

You'll find six files:

- train.data
- train.label
- train.map
- test.data
- test.label
- test.map

The `.data` files are formatted "docIdx wordIdx count". The .label files are simply a list of label id's. The .map files map from label id's to label names. Rainbow was used to lex the data files. I used the following two scripts to produce the data files:

- lexData.sh
- rainbow2matlab.py

[Added 1/14/08] The following file contains the vocabulary for the indexed data. The line number corresponds to the index number of the word---word on the first line is word #1, word on the second line is word #2, etc.