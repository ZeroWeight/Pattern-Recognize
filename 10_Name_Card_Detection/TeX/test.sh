cd pytorch-faster-rcnn/tools

#testing in batch
CUDA_VISIBLE_DEVICES=$GPU_ID python3.5 test.py namelist.txt >> log.txt

#testing per sample
CUDA_VISIBLE_DEVICES=$GPU_ID python3.5 test.py test.jpg >> log.txt