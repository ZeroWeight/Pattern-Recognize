# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
__sets = {}

from datasets.pascal_voc import pascal_voc
import numpy as np

for split in ['train', 'val', 'trainval', 'test']:
  name = 'voc_2007_{}'.format(split)
  __sets[name] = (lambda split=split: pascal_voc(split))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
