require 'torch'
require 'nn'
require 'cunn'
require 'rnn'


criterion1 = nn.AbsCriterion()
criterion2 = nn.MaskZeroCriterion(criterion1, 1):cuda()


x1 = torch.Tensor({{1,2,3,4,5,1},{1,2,0,4,5,1},{1,1,0,0,0,0}})
x2 = torch.Tensor({{1,0,3,4,5,0},{1,2,0,4,5,0},{0,0,0,0,0,0}})
--y1=criterion1:forward(x1,x2)
y2=criterion2:forward(x1,x2)
print(y1)
print(y2)