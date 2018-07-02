require 'nn'
require 'cunn'
require 'rnn'
require 'TripletEmbedding'
require 'TripletEmbeddingAbs'

require("io")
require("os")
require("paths")

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

bsize=1
esize=64
hsize=128

torch.seed()
x1a = torch.Tensor(bsize,10):random(1,3059):cuda()
x1b1 = torch.Tensor(bsize,62):random(1,5252):cuda()
x1b2 = torch.Tensor(bsize,62):random(1,5252):cuda()
x1b = torch.cat(x1b1,x1b2,1)

--x1c = torch.Tensor(1,62):fill(1):cuda()
split = nn.SplitTable(2)
x2a = split:forward(x1a)
x2b = split:forward(x1b)

mdlPri = nn.Sequential()
ltPri = nn.LookupTableMaskZero(3059,esize) -- MaskZero
sp = nn.Sequencer(ltPri)
mdlPri:add(nn.Sequencer(ltPri))

modPri = nn.ConcatTable()
for i = 1, 10, 1 do -- seq length
    recPri = nn.NarrowTable(i,1)
    modPri:add(recPri)
end

mdlPri:add(modPri)
add_lstmPri = nn:Sequential()
add_lstmPri:add(nn.Sequencer(nn.CAddTable()))
    
add_lstmPri:add(nn.Sequencer(nn.Linear(esize, hsize)))
lstmP1 = nn.LSTM(hsize,hsize):maskZero(1)
lstmPri = nn.Sequential():add(lstmP1):add(nn.NormStabilizer())
lstmPri:add(nn.Dropout(0.6)) 
  
add_lstmPri:add(nn.Sequencer(lstmPri))
add_lstmPri:add(nn.CAddTable()) -- add linear
add_lstmPri:add( nn.MulConstant(1/10))
mdlPri:add(add_lstmPri)
nn.MaskZero(mdlPri,1)
mdlPri:cuda()



mdlSec = nn.Sequential()
ltSec = nn.LookupTableMaskZero(5252,esize) -- MaskZero
ss = nn.Sequencer(ltSec)
mdlSec:add(nn.Sequencer(ltSec))

modSec = nn.ConcatTable()
for i = 1, 10, 1 do -- seq length
    recSec = nn.NarrowTable(i,1)
    modSec:add(recSec)
end

mdlSec:add(modSec)
add_lstmSec = nn:Sequential()
add_lstmSec:add(nn.Sequencer(nn.CAddTable()))
    
add_lstmSec:add(nn.Sequencer(nn.Linear(esize, hsize)))
lstmS1 = nn.LSTM(hsize,hsize):maskZero(1)
lstmSec = nn.Sequential():add(lstmS1):add(nn.NormStabilizer())
lstmSec:add(nn.Dropout(0.6)) 
  
add_lstmSec:add(nn.Sequencer(lstmSec))
add_lstmSec:add(nn.CAddTable()) -- add linear
add_lstmSec:add( nn.MulConstant(1/10))
mdlSec:add(add_lstmSec)
nn.MaskZero(mdlSec,1)
mdlSec:cuda()

mdlPri:getParameters():uniform(-1*1,1)
mdlSec:getParameters():uniform(-1*1,1)

mdlPri:training()
mdlSec:training()
mdlPri:zeroGradParameters()
mdlSec:zeroGradParameters()
mdlPri:forget()
mdlSec:forget()

ra = mdlPri:forward(x2a)
rr = mdlSec:forward(x2b)
rb = rr[{{1,bsize},{}}]
rc = rr[{{bsize+1,2*bsize},{}}] 

if false then
local a = ra -- ancor
local p = rb[{{1,bsize},{}}] -- positive
local n = rb[{{bsize+1,2*bsize},{}}] -- negative
local N = a:size(1)

t1 = torch.Tensor(N):zero():type(torch.type(a))
nm1 = (a - p):norm(2,2):pow(2)
nm2 = (a - n):norm(2,2):pow(2)
t2 = nm1 - nm2 + 2
t3 = torch.cat(t1 , t2, 2)
t4 = torch.max(t3, 2)
output = t4:sum() / N

o1 = t4:gt(0):repeatTensor(1,a:size(2)):type(a:type())
o2 = (n - p):cmul(o1 * 2/N)
g1 = (n - p):cmul(self.Li:gt(0):repeatTensor(1,a:size(2)):type(a:type()) * 2/N)
g2 = (p - a):cmul(self.Li:gt(0):repeatTensor(1,a:size(2)):type(a:type()) * 2/N)
g3 = (a - n):cmul(self.Li:gt(0):repeatTensor(1,a:size(2)):type(a:type()) * 2/N)
end


loss1 = nn.TripletEmbeddingCriterion(.2):cuda()

print(colour.red('loss: '), loss1:forward({ra, rb, rc}), '\n')
gradInput1 = loss1:backward({ra, rb, rc})
print(b('gradInput[1]:')); print(gradInput1[1])
print(b('gradInput[2]:')); print(gradInput1[2])
print(b('gradInput[3]:')); print(gradInput1[3])

loss2 = nn.TripletEmbeddingCriterionAbs(.2):cuda()

print(colour.red('loss: '), loss2:forward({ra, rb, rc}), '\n')
gradInput2 = loss2:backward({ra, rb, rc})
print(b('gradInput[1]:')); print(gradInput2[1])
print(b('gradInput[2]:')); print(gradInput2[2])
print(b('gradInput[3]:')); print(gradInput2[3])


print("end")
