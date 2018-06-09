import os
pre = ''
with open('train.txt','r') as f:
  with open(os.path.join('ImageSets','Main','trainval.txt'),'w') as g:
    for line in f:
      a  = line.split(' ' )[1].split('.')[0]
      if a != pre:
        pre = a
        g.write('{}\n'.format((line.split(' ' )[1].split('.')[0])))
pre = ''
with open('valid.txt','r') as f:
  with open(os.path.join('ImageSets','Main','test.txt'),'w') as g:
    for line in f:
      a = line.split(' ' )[1].split('.')[0]
      if a != pre:
        pre = a 
        g.write('{}\n'.format((line.split(' ')[1].split('.')[0])))

