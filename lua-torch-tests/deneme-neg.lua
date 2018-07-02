require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

torch.seed()
x1 = torch.Tensor(1,5):random(1,3059):cuda()
x2 = torch.Tensor(1,5):random(1,3059):cuda()
xn = torch.Tensor(1,5):random(1,3059):cuda()


additivePri = nn.Sequential()
ltPri = nn.LookupTableMaskZero(3059,64) -- MaskZero
additivePri:add( nn.SplitTable(2))
additivePri:add( nn.Sequencer(ltPri))
additivePri:add( nn.CAddTable())
additivePri:add( nn.MulConstant(1))
additivePri:cuda()

additiveSec = nn.Sequential()
ltSec = nn.LookupTableMaskZero(3059,64) -- MaskZero
additiveSec:add( nn.SplitTable(2))
additiveSec:add( nn.Sequencer(ltSec))
additiveSec:add( nn.CAddTable())
additiveSec:add( nn.MulConstant(1))
additiveSec:cuda()

additivePri:getParameters():uniform(-1*0.001,0.001)
additiveSec:getParameters():uniform(-1*0.001,0.001)

additivePri:zeroGradParameters()
additiveSec:zeroGradParameters()

o1 = additivePri:forward(x1)
o2 = additiveSec:forward(x2)
n1 = additivePri:forward(xn)

abs = nn.Abs():cuda()
criterion = nn.L1HingeEmbeddingCriterion(1):cuda()
a1 = abs:forward(o1 - o2)
a2 = abs:forward(o1 - n1)
tb = {a1,a2}

err = criterion:forward( tb,1)
gradOutputs = criterion:backward({a1,a2},1)
additivePri:backward(o1, gradOutputs)

print("end")
