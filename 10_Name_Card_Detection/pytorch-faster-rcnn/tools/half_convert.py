from nets.resnet_v1 import resnetv1
import torch

net = resnetv1(num_layers=101)
net.create_architecture(4, tag='default', anchor_scales=[8, 16, 32])

net.load_state_dict(torch.load(os.path.join('../output', 'res101', 'NameCardtrainvalNameCardReal', 'default',
                              'res101_faster_rcnn_iter_200000.pth'), map_location=lambda storage, loc: storage))
			      net.eval()
			      half_net = net.to(torch.float16)

			      torch.save(half_net.state_dict(), 'params.pth')

