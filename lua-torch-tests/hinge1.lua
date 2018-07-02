require 'nn'
require 'cunn'
require 'rnn'
require 'optim'
require 'torch'

require("io")
require("os")
require("paths")

local norm,sign= torch.norm,torch.sign

additivePri = nn.Sequential()
ltPri = nn.LookupTableMaskZero(3059,64) -- MaskZero
additivePri:add( nn.SplitTable(2))
additivePri:add( nn.Sequencer(ltPri))
additivePri:add( nn.CAddTable())
additivePri:add( nn.MulConstant(1))

additiveSec = nn.Sequential()
ltSec = nn.LookupTableMaskZero(3059,64) 
additiveSec:add( nn.SplitTable(2))
additiveSec:add( nn.Sequencer(ltSec))
additiveSec:add( nn.CAddTable())
additiveSec:add( nn.MulConstant(1))

negSmpPri = nn.Sequential()
negSmpPri:add( nn.SplitTable(2))
ltNegPri = nn.LookupTableMaskZero(3059,64) 
ltNegPri.weight:set(ltSec.weight)
negSmpPri:add( nn.Sequencer(ltNegPri))
negSmpPri:add( nn.CAddTable())
negSmpPri:add( nn.MulConstant(1))

negSmpSec = nn.Sequential()
negSmpSec:add( nn.SplitTable(2))
ltNegSec = nn.LookupTableMaskZero(3059,64) 
ltNegSec.weight:set(ltPri.weight)
negSmpSec:add( nn.Sequencer(ltNegSec))
negSmpSec:add( nn.CAddTable())
negSmpSec:add( nn.MulConstant(1))

prPri1 = nn.ParallelTable()
prPri1:add(additivePri)
prPri2 = nn.ParallelTable()
prPri2:add(additiveSec)
prPri2:add(negSmpPri)
sqPri1 = nn.Sequential()
sqPri1:add(prPri2)
sqPri1:add(nn.CAddTable())
prPri1:add(sqPri1)

mlpPri = nn.Sequential()
mlpPri:add(prPri1)
mlpPri:add(nn.PairwiseDistance(1))
mlpPri:cuda()


prSec1 = nn.ParallelTable()
prSec1:add(additiveSec)
prSec2 = nn.ParallelTable()
prSec2:add(additivePri)
prSec2:add(negSmpSec)
sqSec1 = nn.Sequential()
sqSec1:add(prSec2)
sqSec1:add(nn.CAddTable())
prSec1:add(sqSec1)

mlpSec = nn.Sequential()
mlpSec:add(prSec1)
mlpSec:add(nn.PairwiseDistance(1))
mlpSec:cuda()


mlpPri:getParameters():uniform(-1*1,1)
mlpSec:getParameters():uniform(-1*1,1)
prmPri, gPrmSec = mlpPri:parameters()
prmSec, gPrmPri = mlpSec:parameters()

--mlp:getParameters():uniform(-1*0.001,0.001)
margin = 5
crit = nn.HingeEmbeddingCriterion(margin):cuda()
--crit = nn.L1HingeEmbeddingCriterion(margin):cuda()

torch.seed()
smp_num = 10
x1 = torch.Tensor(smp_num,1,5):random(1,3059):cuda()
x2 = torch.Tensor(smp_num,1,5):random(1,3059):cuda()
xn1 = torch.Tensor(smp_num,1,5):random(1,2000):cuda()
xn2 = torch.Tensor(smp_num,1,5):random(1,2000):cuda()
x0 = torch.Tensor(1,5):fill(0):cuda()


-- Use a typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
local pred = mlp:forward(x)
local err = criterion:forward(pred, y)
local gradCriterion = criterion:backward(pred, y)
--print(pred,err,gradCriterion)
mlp:zeroGradParameters()
mlp:backward(x, gradCriterion)
mlp:updateParameters(learningRate)
return err
end

-- push the pair x and y together, notice how then the distance between them given
-- by print(mlp:forward({x, y})[1]) gets smaller
lr = 0.0001
lrd = 10
coefL2 = 0.00001

local optimState = {learningRate = lr,momentum=0.5}
paramsPri, gradParamsPri = mlpPri:getParameters()
paramsSec, gradParamsSec = mlpSec:getParameters()

logger = optim.Logger('hinge_loss_log.txt')

function fevalPri(params)
      _nidx1_ = (_nidx1_ or 0) + 1
      if _nidx1_ > smp_num then _nidx1_ = 1 end
      
      gradParamsPri:zero()

      local outputs = mlpPri:forward( {x1[_nidx1_], {x2[_nidx1_],xn1[_nidx1_]}})
      local loss = crit:forward(outputs, 1)
      local dloss_doutputs = crit:backward(outputs, 1)
      mlpPri:backward( {x1[_nidx1_], {x2[_nidx1_],xn1[_nidx1_]}}, dloss_doutputs)
      --f = f --+ opt.coefL1 * norm(parameters,1)
      loss = loss + coefL2 * norm(params,2)^2/2
      gradParamsPri:add(paramsPri:clone():mul(coefL2) )
     
      return loss, gradParamsPri
end

function fevalSec(params)
      _nidx2_ = (_nidx2_ or 0) + 1
      if _nidx2_ > smp_num then _nidx2_ = 1 end
      
      gradParamsSec:zero()

      local outputs = mlpSec:forward( {x2[_nidx2_], {x1[_nidx2_],xn2[_nidx2_]}})
      local loss = crit:forward(outputs, 1)
      local dloss_doutputs = crit:backward(outputs, 1)
      mlpSec:backward( {x2[_nidx2_], {x1[_nidx2_],xn2[_nidx2_]}}, dloss_doutputs)
      --f = f --+ opt.coefL1 * norm(parameters,1)
      loss = loss + coefL2 * norm(params,2)^2/2
      gradParamsSec:add(paramsSec:clone():mul(coefL2) )
     
      return loss, gradParamsSec
end



for i = 1, 5000 do
   --gradUpdate(mlp, {x1, {x2,xn}}, 1, crit, lr)
   current_lossPri = 0
   current_lossSec = 0
   
  for j = 1,smp_num do
      _,fs = optim.sgd(fevalPri, paramsPri, optimState)
      current_lossPri = current_lossPri + fs[1] 
      _,fs = optim.sgd(fevalSec, paramsSec, optimState)
      current_lossSec = current_lossSec + fs[1] 
  end
   --print(i,lr,err,mlp:forward({x1, {x2,x0}})[1])
   
  current_lossPri = current_lossPri / smp_num
  current_lossSec = current_lossSec / smp_num
  print('current loss pri = ' .. current_lossPri)
  print('current loss sec = ' .. current_lossSec)
  logger:add{['training error pri'] = current_lossPri ,['training error sec'] = current_lossSec }
  logger:style{['training error pri'] = '-' ,['training error sec'] = '-'}
  logger:plot()  
end

-- pull apart the pair x and y, notice how then the distance between them given
-- by print(mlp:forward({x, y})[1]) gets larger

for i = 1, 100 do
   gradUpdate(mlp, {x1, x2}, -1, crit, 0.001)
   print(mlp:forward({x1, x2})[1])
end