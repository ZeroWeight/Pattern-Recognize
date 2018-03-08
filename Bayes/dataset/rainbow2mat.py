#!/usr/bin/env python

# Convert rainbow "--print-matrix=sin" output to Matlab sparse matrix
# format.  Rainbow frequency file expected on stdin. Two arguments to
# script: (1) regex to match lines of standard input, and (2) string
# to use as root for output file names.  Lines that don't match are
# ignored.  Output is two files, with suffixes '.data' and '.label'

# Examples: rainbow2matlab 20news-bydate-train train
#           rainbow2matlab 20news-bydate-test test

import sys
import re

assert(len(sys.argv) == 3)
regex = sys.argv[1]
rootName = sys.argv[2]
labels = {}
fd = open('%s.data' % rootName,'w')
fl = open('%s.label' % rootName,'w')
assert(fd and fl)
i = 0
for line in sys.stdin:
    a = line.split(None,2)
    # skip empty files
    if len(a) < 3: continue
    if not re.search(regex,a[0]): continue
    l = a[1]
    if not l in labels:
        labels[l] = len(labels)
    li = labels[l]
    b = a[2].strip().split('  ')
    # skip files with only one word
    if len(b) <= 1: continue
    i = i + 1
    def printEntry(s):
        j,cnt = map(int,s.split())
        fd.write('%d %d %d\n' % (i,j+1,cnt))
    map(printEntry,b)
    fl.write('%d\n' % (li+1))
fd.close()
fl.close()
print '%s: %d lines processed' % (sys.argv[0],i)
print '%s: %d unique labels' % (sys.argv[0],len(labels))
def srtfun(a,b):
    if labels[a] < labels[b]: return -1
    elif labels[a] > labels[b]: return 1
    else: return 0
fm = open('%s.map' % rootName,'w')
keys = labels.keys()
keys.sort(srtfun)
for k in keys:
    fm.write('%s %d\n' % (k,labels[k]+1))
fm.close()
