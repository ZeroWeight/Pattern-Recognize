import os

original_file = os.path.join('..','NameCardReal','train.txt')
with open('train.txt','w') as des:
  with open(original_file,'r') as ori:
    for line in ori:
      des.write(line)
  
    for idx in range(256):
      with open(original_file,'r') as ori:
        for line in ori:
          words = line.split(' ')
          new_line = '{} {} {} {} {} {} {} {} {}'.format(str(idx + 1) + words[0], str(idx + 1) + words[1],
            words[2], words[3], words[4], words[5], words[6], words[7], words[8])
          des.write(new_line)
