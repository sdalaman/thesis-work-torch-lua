require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

LSTMAvgNarrow = false
LSTMAvgNoNarrow = false
LSTMAvgAttnNoNarrow = true
LSTMScAvgNarrow = false
LSTMScAvgNoNarrow = false
LSTMScAvgAttnNoNarrow =false
BiLSTMAvg = false
BiLSTMAvgAttn = true
BiLSTMScAvgOld = false
BiLSTMScAvg = false
BiLSTMScAvgAttn = false

emb_size = 64
hidden_size = 128
win_size = 1
seq_lenPri = 60
vsizePri = 3059
bsize = 100

nn.FastLSTM.bn = true
rho=nil
eps=nil
momentum=nil
affine=nil
p=0.25
mono=nil

lstm_flag = false
gru_flag = true
rnn_cnt = 2

local split = nn.SplitTable(2)

x1 = torch.Tensor(bsize,seq_lenPri):random(1,vsizePri):cuda()
x1r = x1:index(2,torch.linspace(seq_lenPri,1,seq_lenPri):long())
x2 = torch.Tensor(bsize,seq_lenPri):random(1,vsizePri):cuda()

xs1 = split:forward(x1)
xs1r = split:forward(x1r)
xs2 = split:forward(x2)

w1Out=30
w2Out=10

local attnW1 = nn.Linear(hidden_size,w1Out)
local attnW2 = nn.Linear(w1Out,w2Out)

attnW1:getParameters():uniform(-1,1)
attnW2:getParameters():uniform(-1,1)


function AttnLayerOld(attnLayerSize)
  
  local attnLayer1 = nn.Sequential()
      :add(nn.Sequencer(nn.Unsqueeze(1)))
      :add(nn.Sequencer(nn.SplitTable(2)))
      :add(nn.ZipTable())
      :add(nn.Sequencer(nn.JoinTable(1,2)))
      
  local attnLayer2 = nn.Sequential()
      :add(nn.Sequencer(nn.Linear(attnLayerSize,1)))
      :add(nn.Sequencer(nn.Transpose({1,2})))

  local attnLayer3 = nn.Sequential()
    :add(attnLayer2) -- attn weights
    :add(nn.Sequencer(nn.SoftMax()))
    :add(nn.Sequencer(nn.Transpose({1,2})))

  local con = nn.ConcatTable()
        :add(nn.Sequencer(nn.Transpose({1,2})))
        :add(attnLayer3)

  local attnLayer = nn.Sequential()
   :add(attnLayer1)
   :add(con)
   :add(nn.ZipTable())
   :add(nn.Sequencer(nn.MM()))
   :add(nn.JoinTable(2))
   :add(nn.Transpose({1,2}))

  return attnLayer
end

function AttnLayer(w1,w2)
  
  local attnLayerEntry = nn.Sequential()
      :add(nn.Sequencer(nn.Unsqueeze(1)))
      :add(nn.Sequencer(nn.SplitTable(2)))
      :add(nn.ZipTable())
      :add(nn.Sequencer(nn.JoinTable(1,2)))
      
  local conLeg1a = nn.Sequential()
    :add(w1)
    :add(nn.Tanh())
    :add(w2)
    :add(nn.SoftMax())
    :add(nn.Transpose({1,2}))
    :add(nn.Unsqueeze(1))
  
  local conLeg1 = nn.Sequential()
    :add(nn.Sequencer(conLeg1a))
    :add(nn.JoinTable(1))

  local conLeg2 = nn.Sequential()
    :add(nn.Sequencer(nn.Unsqueeze(1)))
    :add(nn.JoinTable(1))

  local con = nn.ConcatTable()
        :add(conLeg1)
        :add(conLeg2)

  local attnLayer = nn.Sequential()
   :add(attnLayerEntry)
   :add(con)
   :add(nn.MM())

  return attnLayer
  
end


-----------------------------------------------------------------------

-- LSTM with output of hidden states averaged with NarrowTable
if LSTMAvgNarrow == true then
local ltPri = nn.LookupTableMaskZero(vsizePri,emb_size) -- MaskZero
local modPri = nn.ConcatTable()
for i = 1, seq_lenPri, win_size do -- seq length
    local recPri = nn.NarrowTable(i,win_size)
    modPri:add(recPri)
end

local lstmPri = nn.Sequential()
  :add(nn.LSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1))
  :add(nn.NormStabilizer())
local add_lstmPri = nn:Sequential()
  --:add(nn.Sequencer(nn.CAddTable()))
  :add(nn.FlattenTable())
  :add(nn.Sequencer(nn.Linear(emb_size,hidden_size)))
  :add(nn.Sequencer(lstmPri))
  :add(nn.CAddTable()) -- add linear
  :add(nn.MulConstant(1/seq_lenPri))
  
LSTMAvgNarrow = nn.Sequential()  
  :add(nn.Sequencer(ltPri))
  :add(modPri)
  :add(add_lstmPri)
nn.MaskZero(LSTMAvgNarrow,1)
LSTMAvgNarrow:cuda()
out = LSTMAvgNarrow:forward(xs1)
end
----------------------------------------------------------------

-- LSTM with output of hidden states averaged without NarrowTable
if LSTMAvgNoNarrow == true then
local ltPri2 = nn.LookupTableMaskZero(vsizePri,emb_size) 
local fwdPri2 = nn.Sequential()
   :add(ltPri2)
   :add(nn.Linear(emb_size, hidden_size))
   :add(nn.Dropout(p))
   :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
   :add(nn.NormStabilizer())
   :add(nn.Dropout(p))
   :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
   :add(nn.NormStabilizer())
   --:add(nn.Linear(hidden_size,emb_size))

LSTMAvgNoNarrow = nn.Sequential()
   :add(nn.Sequencer(fwdPri2))
   :add(nn.CAddTable()) -- add linear
   :add(nn.MulConstant(1/seq_lenPri))
LSTMAvgNoNarrow:cuda()
out = LSTMAvgNoNarrow:forward(xs1)
end


----------------------------------------------------------------

-- LSTM with output of hidden states averaged without NarrowTable with attention layer

if LSTMAvgAttnNoNarrow == true then
  
local ltPri2 = nn.LookupTableMaskZero(vsizePri,emb_size) 
local fwdPri2 = nn.Sequential()
   :add(ltPri2)
   :add(nn.Linear(emb_size, hidden_size))
   :add(nn.Dropout(p))
   :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
   :add(nn.NormStabilizer())
   :add(nn.Dropout(p))
   :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
   :add(nn.NormStabilizer())
   --:add(nn.Linear(hidden_size,emb_size))

LSTMAvgAttnNoNarrow = nn.Sequential()
   :add(nn.Sequencer(fwdPri2))
   :add(AttnLayer(attnW1,attnW2))
   :add(nn.View(bsize,w2Out*hidden_size))

if false then 
out1 = LSTMAvgAttnNoNarrow:forward(xs1)

out2a = nn.Sequencer(nn.Unsqueeze(1)):forward(out1)
out2b = nn.Sequencer(nn.SplitTable(2)):forward(out2a)
out2c = nn.ZipTable():forward(out2b)
out2d = nn.Sequencer(nn.JoinTable(1,2)):forward(out2c)


bb = nn.Sequencer(nn.Unsqueeze(1)):forward(out2d)
bb = nn.JoinTable(1):forward(bb)

d=4
r=1

w1 = nn.Linear(hidden_size,d)
w2 = nn.Linear(d,r)
out3b1 = nn.Sequencer(w1):forward(out2d)
out3b2 = nn.Sequencer(nn.Tanh()):forward(out3b1)
out3b3 = nn.Sequencer(w2):forward(out3b2)
out3b5 = nn.Sequencer(nn.SoftMax()):forward(out3b3)
out3b6 = nn.Sequencer(nn.Transpose({1,2})):forward(out3b5)
aa = nn.Sequencer(nn.Unsqueeze(1)):forward(out3b6)
aa = nn.JoinTable(1):forward(aa)

out5 = nn.MM():forward({aa,bb})
end

LSTMAvgAttnNoNarrow:cuda()
  
outLSTMAvgAttnNoNarrow = LSTMAvgAttnNoNarrow:forward(xs1)


--a1 = nn.Sequencer(w1):forward(outLSTMAvgAttnNoNarrow[2])
--a2 = nn.Sequencer(nn.Tanh()):forward(a1)
--a3 = nn.Sequencer(w2):forward(a2)
--a4 = nn.Sequencer(nn.SoftMax()):forward(a3)
--a5 = nn.Sequencer(nn.Transpose({1,2})):forward(a4)
--a6 = nn.Sequencer(nn.Unsqueeze(1)):forward(a5)
--a7 = nn.JoinTable(1):forward(a6)

--out5 = nn.MM():forward({outLSTMAvgAttnNoNarrow[1],outLSTMAvgAttnNoNarrow[2]})


out={}
end


----------------------------------------------------------------

-- LSTM with short-cut connections and output of hidden states averaged with NarrowTable

if LSTMScAvgNarrow == true then
local duplicateInputSC = nn.ConcatTable()
  :add(nn.Identity())
  :add(nn.Identity())
local sqntlSC = nn.Sequential()
  :add(nn.Linear(emb_size,hidden_size))
  :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
  :add(nn.NormStabilizer())
  :add(nn.Linear(hidden_size,emb_size))
local prlSC = nn.ParallelTable()
  :add(sqntlSC)
  :add(nn.Identity())

 
local ltPriSC = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,emb_size))

local modPriSC = nn.ConcatTable()
for i = 1, seq_lenPri, win_size do -- seq length
    local recPriSC = nn.NarrowTable(i,win_size)
    modPriSC:add(recPriSC)
end

local add_lstmPriSC = nn:Sequential()
  :add(duplicateInputSC)
  :add(prlSC)

LSTMScAvgNarrow = nn.Sequential()
  :add(ltPriSC)
  :add(modPriSC)
  :add(nn.FlattenTable())
  :add(nn.Sequencer(add_lstmPriSC))
  :add(nn.FlattenTable()) 
  :add(nn.CAddTable()) -- add linear
  :add( nn.MulConstant(1/(2*seq_lenPri)))
nn.MaskZero(LSTMScAvgNarrow,1)
LSTMScAvgNarrow:cuda()
out = LSTMScAvgNarrow:forward(xs1)
end

----------------------------------------------------------------

-- LSTM with short-cut connections and output of hidden states averaged  without NarrowTable
if LSTMScAvgNoNarrow == true then
local ltPriSC2 = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,emb_size))
local duplicateInputSC2 = nn.ConcatTable()
  :add(nn.Identity())
  :add(nn.Identity())

local sqntlSC2 = nn.Sequential()
  :add(nn.Linear(emb_size,hidden_size))
  :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
  :add(nn.NormStabilizer())
  :add(nn.Linear(hidden_size,emb_size))
local prlSC2 = nn.ParallelTable()
  :add(sqntlSC2)
  :add(nn.Identity())


local fwdPriSC2 = nn.Sequential()
   :add(duplicateInputSC2)
   :add(prlSC2)
   :add(nn.CAddTable())

LSTMScAvgNoNarrow = nn.Sequential()
   :add(ltPriSC2)
   :add(nn.Sequencer(fwdPriSC2))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add(nn.MulConstant(1/seq_lenPri))
   
LSTMScAvgNoNarrow:cuda()

out = LSTMScAvgNoNarrow:forward(xs1)

end

----------------------------------------------------------------

-- LSTM with short-cut connections and output of hidden states averaged  without NarrowTable with attention layer
if LSTMScAvgAttnNoNarrow == true then
  
local ltPriSC2 = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,emb_size))
local duplicateInputSC2 = nn.ConcatTable()
  :add(nn.Identity())
  :add(nn.Identity())

local sqntlSC2 = nn.Sequential()
  :add(nn.Linear(emb_size,hidden_size))
  :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
  :add(nn.NormStabilizer())
  :add(nn.Linear(hidden_size,emb_size))
local prlSC2 = nn.ParallelTable()
  :add(sqntlSC2)
  :add(nn.Identity())


local fwdPriSC2 = nn.Sequential()
   :add(duplicateInputSC2)
   :add(prlSC2)
   :add(nn.CAddTable())

LSTMScAvgAttnNoNarrow = nn.Sequential()
   :add(ltPriSC2)
   :add(nn.Sequencer(fwdPriSC2))
   :add(AttnLayer(emb_size))

LSTMScAvgAttnNoNarrow:cuda()

outLSTMScAvgAttnNoNarrow = LSTMScAvgAttnNoNarrow:forward(xs1)

end

---------------------------------------

-- BiDirectional LSTM and output of hidden states averaged  NOT USED

if BiLSTMAvgOld == true then
local sharedLookupTablePri = nn.LookupTableMaskZero(vsizePri, emb_size)

local fwdPri = nn.Sequential()
   :add(sharedLookupTablePri)
   :add(nn.Linear(emb_size, hidden_size))
   :add(nn.Sequential():add(nn.LSTM(hidden_size,hidden_size):maskZero(1)):add(nn.NormStabilizer()))
   :add(nn.Linear(hidden_size,emb_size))

local fwdPriSeq = nn.Sequential()
   :add(nn.Sequencer(fwdPri))

local bwdPri = nn.Sequential()
   :add(sharedLookupTablePri:sharedClone())
   :add(nn.Linear(emb_size,hidden_size))
   :add(nn.Sequential():add(nn.LSTM(hidden_size,hidden_size):maskZero(1)):add(nn.NormStabilizer()))
   :add(nn.Linear(hidden_size,emb_size))
  
local bwdPriSeq = nn.Sequencer(bwdPri)

local mergePri = nn.JoinTable(1, 1)
local mergePriSeq = nn.Sequencer(mergePri)

local parallelPri = nn.ParallelTable()
  :add(fwdPriSeq)
  :add(bwdPriSeq)
BiLSTMAvgOld = nn.Sequential()
   :add(parallelPri)
   :add(nn.ZipTable())
   :add(mergePriSeq)
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

BiLSTMAvgOld:cuda()

out = BiLSTMAvgOld:forward({xs1,xs1r})
end

---------------------------------------------------------------------
--BiLSTMAvg GRU / LSTM  and output of hidden states averaged

if BiLSTMAvg == true then  
  
local lt = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,emb_size))
local lstmBlF = nn.Sequential()
  :add(nn.Linear(emb_size,hidden_size))
for i=1,rnn_cnt do
     if lstm_flag == true then
      lstmBlF:add(nn.LSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1))
     end
     if gru_flag == true then
      lstmBlF:add(nn.GRU(hidden_size,hidden_size,rho, p, mono):maskZero(1))
     end
    lstmBlF:add(nn.NormStabilizer())
end
lstmBlF:add(nn.Linear(hidden_size,emb_size))

local lstmBlB = nn.Sequential()
  :add(nn.Linear(emb_size,hidden_size))
for i=1,rnn_cnt do
     if lstm_flag == true then
      lstmBlB:add(nn.LSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1))
     end
     if gru_flag == true then
      lstmBlB:add(nn.GRU(hidden_size,hidden_size,rho, p, mono):maskZero(1))
     end
    lstmBlB:add(nn.NormStabilizer())
end
lstmBlB:add(nn.Linear(hidden_size,emb_size))

local fwdSeq = nn.Sequencer(lstmBlF)
local bwdSeq = nn.Sequencer(lstmBlB)
local mergeSeq = nn.Sequencer(nn.JoinTable(1, 1))
   
local backward = nn.Sequential()
  :add(nn.ReverseTable()) -- reverse
  :add(bwdSeq)
  :add(nn.ReverseTable()) -- unreverse
   
local concat = nn.ConcatTable()
  :add(fwdSeq)
  :add(backward)
   
BiLSTMAvg = nn.Sequential()
  :add(lt)
  :add(concat)
  :add(nn.ZipTable())
  :add(mergeSeq)
  :add(nn.CAddTable())
  :add( nn.MulConstant(1/(2*seq_lenPri)))

out = BiLSTMAvg:forward(xs1)
end


---------------------------------------------------------------------
--BiLSTMAvg GRU / LSTM  and output of hidden states averaged

if BiLSTMAvgAttn == true then  
  
local lt = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,emb_size))
local lstmBlF = nn.Sequential()
  :add(nn.Linear(emb_size,hidden_size))
for i=1,rnn_cnt do
     if lstm_flag == true then
      lstmBlF:add(nn.LSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1))
     end
     if gru_flag == true then
      lstmBlF:add(nn.GRU(hidden_size,hidden_size,rho, p, mono):maskZero(1))
     end
    lstmBlF:add(nn.NormStabilizer())
end
--lstmBlF:add(nn.Linear(hidden_size,emb_size))

local lstmBlB = nn.Sequential()
  :add(nn.Linear(emb_size,hidden_size))
for i=1,rnn_cnt do
     if lstm_flag == true then
      lstmBlB:add(nn.LSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1))
     end
     if gru_flag == true then
      lstmBlB:add(nn.GRU(hidden_size,hidden_size,rho, p, mono):maskZero(1))
     end
    lstmBlB:add(nn.NormStabilizer())
end
--lstmBlB:add(nn.Linear(hidden_size,emb_size))

local fwdSeq = nn.Sequencer(lstmBlF)
local bwdSeq = nn.Sequencer(lstmBlB)
local mergeSeq = nn.Sequencer(nn.JoinTable(1, 1))
   
local backward = nn.Sequential()
  :add(nn.ReverseTable()) -- reverse
  :add(bwdSeq)
  :add(nn.ReverseTable()) -- unreverse
   
local concat = nn.ConcatTable()
  :add(fwdSeq)
  :add(backward)
   
BiLSTMAvgAttn = nn.Sequential()
  :add(lt)
  :add(concat)
  :add(nn.ZipTable())
  :add(mergeSeq)
  :add(AttnLayer(2*hidden_size))

BiLSTMAvgAttn:cuda()

outBiLSTMAvgAttn = BiLSTMAvgAttn:forward(xs1)
end


--------------------------------------------------------

-- BiDirectional LSTM  SC with short-cut connections and output of hidden states averaged

if BiLSTMScAvgOld == true then
  
local sharedLookupTablePriSC = nn.LookupTableMaskZero(vsizePri, emb_size)

local fwdDuplicatePriSC = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
fwdLstmPriSC = nn.Sequential()
   :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
   :add(nn.NormStabilizer())
local fwdSqntlPriSC = nn.Sequential()
   :add(nn.Linear(emb_size,hidden_size))
   :add(fwdLstmPriSC)
   :add(nn.Linear(hidden_size,emb_size))
local fwdPrlPriSC = nn.ParallelTable()
   :add(fwdSqntlPriSC)
   :add(nn.Identity())


local fwdPriSC = nn.Sequential()
   :add(sharedLookupTablePriSC)
   :add(fwdDuplicatePriSC)
   :add(fwdPrlPriSC)


local fwdPriSeqSC = nn.Sequential()
   :add(nn.Sequencer(fwdPriSC))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

local bwdDuplicatePriSC = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
local bwdLstmPriSC = nn.Sequential()
   :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
   :add(nn.NormStabilizer())
local bwdSqntlPriSC = nn.Sequential()
   :add(nn.Linear(emb_size,hidden_size))
   :add(bwdLstmPriSC)
   :add(nn.Linear(hidden_size,emb_size))
local bwdPrlPriSC = nn.ParallelTable()
   :add(bwdSqntlPriSC)
   :add(nn.Identity())


local bwdPriSC = nn.Sequential()
   :add(sharedLookupTablePriSC:sharedClone())
   :add(bwdDuplicatePriSC)
   :add(bwdPrlPriSC)


local bwdPriSeqSC = nn.Sequential()
   :add(nn.Sequencer(bwdPriSC))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

local mergePriSC = nn.JoinTable(1, 1)

local parallelPriSC = nn.ParallelTable()
  :add(fwdPriSeqSC):add(bwdPriSeqSC)
  
BiLSTMScAvgOld = nn.Sequential()
   :add(parallelPriSC)
   :add(mergePriSC)

BiLSTMScAvgOld:cuda()

out = BiLSTMScAvgOld:forward({xs1,xs1r})

end

--------------------------------------------------------

-- BiDirectional LSTM  SC with short-cut connections and output of hidden states averaged version 2

if BiLSTMScAvg == true then

local gate = nn.LSTM(hidden_size,hidden_size):maskZero(1)

local sharedLookupTablePriSc = nn.LookupTableMaskZero(vsizePri, emb_size)

local   fwdDuplicatePriSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
   
local   fwdSqntlPriSc = nn.Sequential()
  
  if emb_size ~= hidden_size then
    fwdSqntlPriSc:add(nn.Linear(emb_size,hidden_size))
  end
  
  fwdSqntlPriSc:add(gate:clone())
        :add(nn.NormStabilizer())
        
  if emb_size ~= hidden_size then
    fwdSqntlPriSc:add(nn.Linear(hidden_size,emb_size))
  end
  
  local fwdPrlPriSc = nn.ParallelTable()
   :add(fwdSqntlPriSc)
   :add(nn.Identity())

  local fwdPriSc = nn.Sequential()
   :add(fwdDuplicatePriSc)
   :add(fwdPrlPriSc)
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add(nn.MulConstant(1/2))

  local bckPriSeqSc = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(nn.Sequencer(fwdPriSc:clone()))
    :add(nn.ReverseTable()) -- unreverse
   
  local concatPriSc = nn.ConcatTable()
    :add(nn.Sequencer(fwdPriSc))
    :add(bckPriSeqSc)

  local mergePriSc = nn.JoinTable(1, 1)

  BiLSTMScAvg = nn.Sequential()
    :add(nn.Sequencer(sharedLookupTablePriSc))
    :add(concatPriSc)
    :add(nn.ZipTable())
    :add(nn.Sequencer(mergePriSc))
    :add(nn.CAddTable())
    :add(nn.MulConstant(1/seq_lenPri))

  BiLSTMScAvg:cuda()
  
  out = BiLSTMScAvg:forward(xs1)
  
  out1 = {}
  
end

--------------------------------------------------------

-- BiDirectional LSTM  SC with short-cut connections and output of hidden states averaged version 2 and attention layer

if BiLSTMScAvgAttn == true then

local gate = nn.LSTM(hidden_size,hidden_size):maskZero(1)

local sharedLookupTablePriSc = nn.LookupTableMaskZero(vsizePri, emb_size)

local   fwdDuplicatePriSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
   
local   fwdSqntlPriSc = nn.Sequential()
  
  if emb_size ~= hidden_size then
    fwdSqntlPriSc:add(nn.Linear(emb_size,hidden_size))
  end
  
  fwdSqntlPriSc:add(gate:clone())
        :add(nn.NormStabilizer())
        
  if emb_size ~= hidden_size then
    fwdSqntlPriSc:add(nn.Linear(hidden_size,emb_size))
  end
  
  local fwdPrlPriSc = nn.ParallelTable()
   :add(fwdSqntlPriSc)
   :add(nn.Identity())

  local fwdPriSc = nn.Sequential()
   :add(fwdDuplicatePriSc)
   :add(fwdPrlPriSc)
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add(nn.MulConstant(1/2))

  local bckPriSeqSc = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(nn.Sequencer(fwdPriSc:clone()))
    :add(nn.ReverseTable()) -- unreverse
   
  local concatPriSc = nn.ConcatTable()
    :add(nn.Sequencer(fwdPriSc))
    :add(bckPriSeqSc)

  local mergePriSc = nn.JoinTable(1, 1)

  BiLSTMScAvgAttn = nn.Sequential()
    :add(nn.Sequencer(sharedLookupTablePriSc))
    :add(concatPriSc)
    :add(nn.ZipTable())
    :add(nn.Sequencer(mergePriSc))
    :add(AttnLayer(2*emb_size))

  BiLSTMScAvgAttn:cuda()
  
  outBiLSTMScAvgAttn = BiLSTMScAvgAttn:forward(xs1)
  
  out1 = {}
  
end

outLSTMAvgNarrow = {}
outLSTMAvgNoNarrow = {}
outLSTMAvgAttnNoNarrow = {}
outLSTMScAvgNarrow = {}
outLSTMScAvgNoNarrow = {}
outLSTMScAvgAttnNoNarrow = {}
outBiLSTMAvg = {}
outBiLSTMAvgAttn = {}
outBiLSTMScAvg = {}
outBiLSTMScAvgAttn = {}


print("end")

