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
math.randomseed(0)

-- hyper-parameters 
batchSize = 1
rho = 10 -- sequence length
esize=64
hiddenSize = 64
nIndex = 3059
lr = 0.1
maxIter = 100


x1a = torch.Tensor(batchSize,rho):random(1,3059):cuda()
x1b1 = torch.Tensor(batchSize,rho):random(1,5252):cuda()
x1b2 = torch.Tensor(batchSize,rho):random(1,5252):cuda()
--x1b = torch.cat(x1b1,x1b2,1)

targets = torch.Tensor(batchSize,2*hiddenSize):random():cuda()

x1a_r = x1a:index(2 ,torch.linspace(rho,1,rho):long())

split = nn.SplitTable(2)

x2a = split:forward(x1a)
x2a_r = split:forward(x1a_r)
x2b = split:forward(x1b1)

local sharedLookupTable = nn.LookupTableMaskZero(nIndex, esize)

-- forward rnn
local fwd = nn.Sequential()
   :add(sharedLookupTable)
   :add(nn.Linear(esize, hiddenSize))
   :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
fwdSeq = nn.Sequencer(fwd)

-- backward rnn (will be applied in reverse order of input sequence)
local bwd = nn.Sequential()
   :add(sharedLookupTable:sharedClone())
   :add(nn.Linear(esize, hiddenSize))
   :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))
bwdSeq = nn.Sequencer(bwd)

-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
local merge = nn.JoinTable(1, 1) 
mergeSeq = nn.Sequencer(merge)

-- Assume that two input sequences are given (original and reverse, both are right-padded).
-- Instead of ConcatTable, we use ParallelTable here.
local parallel = nn.ParallelTable()
parallel:add(fwdSeq):add(bwdSeq)
local brnn = nn.Sequential()
   :add(parallel)
   :add(nn.ZipTable())
   :add(mergeSeq)
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/rho))

--print(brnn)

criterion = nn.AbsCriterion():cuda()
nn.MaskZeroCriterion(criterion, 1)

brnn:cuda()

--criterion = nn.SequencerCriterion(criterion):cuda()

brnn:zeroGradParameters() 

local outputs = brnn:forward({x2a,x2a_r})
local err = criterion:forward(outputs, targets)

local gradOutputs = criterion:backward(outputs, targets)
local gradInputs = brnn:backward({x2a,x2a_r}, gradOutputs)
   
brnn:updateParameters(0.1)



print("end")
