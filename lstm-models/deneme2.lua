require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

bsize=1
esize=64
hsize=64

torch.seed()
x1a = torch.Tensor(bsize,105):random(1,3059):cuda()
x1b1 = torch.Tensor(bsize,62):random(1,5252):cuda()
x1b2 = torch.Tensor(bsize,62):random(1,5252):cuda()
x1b = torch.cat(x1b1,x1b2,1)

--x1c = torch.Tensor(1,62):fill(1):cuda()
split = nn.SplitTable(2)
x2a = split:forward(x1a)
x2b = split:forward(x1b)

mdlPri = nn.Sequential()
ltPri = nn.LookupTableMaskZero(3059,esize) -- MaskZero
rr1 = nn.Sequencer(ltPri):forward(x2a)
mdlPri:add(nn.Sequencer(ltPri))

modPri = nn.ConcatTable()
for i = 1, 10, 1 do -- seq length
    recPri = nn.NarrowTable(i,1)
    modPri:add(recPri)
end
rr2 = modPri:forward(rr1)


mdlPri:add(modPri)
add_lstmPri = nn:Sequential()
add_lstmPri:add(nn.Sequencer(nn.CAddTable()))
    
add_lstmPri:add(nn.Sequencer(nn.Linear(esize, hsize)))
lstmP1 = nn.LSTM(hsize,hsize):maskZero(1)
lstmPri = nn.Sequential():add(lstmP1):add(nn.NormStabilizer())
lstmPri:add(nn.Dropout(0.6)) 


ex1 = nn.Identity()
ar = ex1:forward(rr2)
  
add_lstmPri:add(nn.Sequencer(lstmPri))
rr3 = add_lstmPri:forward(rr2)

add_lstmPri2 = nn.Sequential():add(nn.ConcatTable():add(add_lstmPri):add(nn.Sequencer(ex1)))
rr4 = add_lstmPri2:forward(rr2,rr2)

tt = nn.CAddTable():forward(nn.FlattenTable():forward(rr4))

add_lstmPri:add(nn.CAddTable()) -- add linear
add_lstmPri:add( nn.MulConstant(1/10))
mdlPri:add(add_lstmPri)
nn.MaskZero(mdlPri,1)
mdlPri:cuda()




--ra = mdlPri:forward(x2a)
--rr = mdlSec:forward(x2b)
--rb = rr[{{1,bsize},{}}]
--rc = rr[{{bsize+1,2*bsize},{}}] 


mlp = nn.Sequential()
mlp:add(nn.CMul(5, 1))

y = torch.Tensor(5, 4)
sc = torch.Tensor(5, 4)
for i = 1, 5 do sc[i] = 2*i; end -- scale input with this

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   return err
end

for i = 1, 10000 do
   x = torch.rand(5, 4)
   y:copy(x)
   y:cmul(sc)
   err = gradUpdate(mlp, x, y, nn.MSECriterion(), 0.01)
end

print(mlp:get(1).weight)

print("end")
