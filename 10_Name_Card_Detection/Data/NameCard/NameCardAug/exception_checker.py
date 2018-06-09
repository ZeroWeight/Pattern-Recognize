import os

for filename in ['train.txt', 'valid.txt']:
  with open('train.txt','r') as f:
    for line in f:
      words = line.split(' ')
      index = words[1].split('.')[0]
      if (int(words[2]) > int(words[4])) or (int(words[3]) > int(words[5])):
        print(line,words[2],words[4],words[3],words[5])
        print(words[2] > words[4],words[3] > words[5])

