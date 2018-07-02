
math.randomseed(1)

require 'nn'
require 'cunn'
require 'rnn'
require 'lfs'
require 'utils'
require 'optim'
require 'io'
require 'os'
require 'paths'

local stringx = require('pl.stringx')


local file = require('pl.file')


path = "/home/saban/work/additive/"
test_doc_data_path = path.."/cdlc-new/lstm/data/"

allPri=torch.load(test_doc_data_path.."lstm.english.10000.tok.1e-05_1150.lstm.512.60.lstm-SC-avg-drop-10000.arts.train.input.pch")
targetsPri=torch.load(test_doc_data_path.."lstm.english.10000.tok.1e-05_1150.lstm.512.60.lstm-SC-avg-drop-10000.arts.train.targets.pch")

for i=1,targetsPri:size()[1] do
  if targetsPri[i] == 0 then
    targetsPri[i] = -1
  end
end

function count(x)
  local pcnt = 0
  local ncnt = 0
  for i = 1,x:size()[1] do
    if x[i][1] >= 0 then
      pcnt = pcnt + 1
    else
      ncnt = ncnt + 1
    end
  end
  return pcnt,ncnt
end

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

mlp = nn.Sequential()
mlp:add(nn.Linear(allPri:size()[2], 1))
--mlp:add(nn.Linear(5, 1))
mlp:cuda()
--x1 = torch.rand(10,5)
--x1_target = torch.Tensor(10):fill(1)
--x2 = torch.rand(10,5)
--x2_target = torch.Tensor(10):fill(-1)
criterion=nn.MarginCriterion(1):cuda()
--criterion=nn.MarginCriterion(1)


n = allPri:size()[1]
allPri = allPri[{{1,n},{}}]:cuda()
targetsPri = targetsPri[{{1,n}}]:cuda()

for i = 1, 10000 do
   gradUpdate(mlp, allPri, targetsPri, criterion, 0.01)
   gradUpdate(mlp, allPri, targetsPri, criterion, 0.01)
--   gradUpdate(mlp, x1, x1_target, criterion, 0.01)
--   gradUpdate(mlp, x2, x2_target, criterion, 0.01)
end

pcnt = 0
ncnt = 0
pred = mlp:forward(allPri)
for i=1,targetsPri:size()[1] do
  if pred[i][1] >= 0 then
    pcnt = pcnt + 1
  else
    ncnt = ncnt + 1
  end
end


print(pcnt)
print(ncnt)

--res1 = mlp:forward(x1)
--res2 = mlp:forward(x2)
--x3 = torch.randn(10,5)
--res3 = mlp:forward(x3)

--r1,r2 = count(res1)
--print("sample 1 :"..r1.."-"..r2)
--r1,r2 = count(res2)
--print("sample 2 :"..r1.."-"..r2)
--r1,r2 = count(res3)
--print("sample 3 :"..r1.."-"..r2)
--print(criterion:forward(mlp:forward(x1), x1_target))
--print(criterion:forward(mlp:forward(x2), x2_target))

print("end")

