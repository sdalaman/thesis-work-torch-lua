require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

x = torch.ones(2)
y = torch.ones(3)

net = nn.Sequential()
triple = nn.ParallelTable()
duplicate = nn.ConcatTable()
duplicate:add(nn.Identity())
duplicate:add(nn.Identity())
triple:add(duplicate)
triple:add(nn.Identity())
net:add(triple)
net:add(nn.FlattenTable())

-- at this point the network transforms {X,Y} into {X,X,Y}
separate = nn.ConcatTable()
separate:add(nn.SelectTable(1))
separate:add(nn.NarrowTable(2,2))
net:add(separate)
-- now you get {X,{X,Y}}
parallel_XY = nn.ParallelTable()
parallel_XY:add(nn.Identity()) -- preserves X
parallel_XY:add(...) -- whatever you want to do from {X,Y}
net:add(parallel_XY)
parallel_Xresult = nn.ParallelTable()
parallel_Xresult:add(...)  -- whatever you want to do from {X,result}
net:add(parallel_Xresult)

--output = net:forward({X,Y})


net = nn.Sequential()
duplicate = nn.ConcatTable()
duplicate:add(nn.Identity())
duplicate:add(nn.Identity())
net:add(duplicate)
par = nn.ParallelTable()
sq = nn.Sequential()
sq:add(nn.Linear(2,10))
sq2 = nn.Sequential()
sq2:add(nn.LSTM(10,10):maskZero(1))
--sq2:add(nn.NormStabilizer())
sq:add(sq2)
sq:add(nn.Linear(10,2))
--lstm = nn.Sequential():add(nn.LSTM(10,10):maskZero(1)):add(nn.NormStabilizer())
--sq:add(lstm)
par:add(sq)
par:add(nn.Identity())
net:add(par)
net:add(nn.CAddTable())

--out = net:forward(x)

emb_size = 64
hidden_size = 128
win_size = 1
seq_lenPri = 60
vsizePri = 3059
bsize = 100

x1 = torch.Tensor(bsize,seq_lenPri):random(1,vsizePri):cuda()
x1r=x1:index(2,torch.linspace(seq_lenPri,1,seq_lenPri):long())
x2 = torch.Tensor(bsize,seq_lenPri):random(1,vsizePri):cuda()

local split = nn.SplitTable(2)
xs1 = split:forward(x1)
xs1r = split:forward(x1r)
xs2 = split:forward(x2)

-- LSTM with output of hidden states averaged 
mdlPri = nn.Sequential()
ltPri = nn.LookupTableMaskZero(vsizePri,emb_size) -- MaskZero
mdlPri:add(nn.Sequencer(ltPri))
modPri = nn.ConcatTable()
for i = 1, seq_lenPri, win_size do -- seq length
    recPri = nn.NarrowTable(i,win_size)
    modPri:add(recPri)
end

mdlPri:add(modPri)
add_lstmPri = nn:Sequential()
add_lstmPri:add(nn.Sequencer(nn.CAddTable()))
add_lstmPri:add(nn.Sequencer(nn.Linear(emb_size,hidden_size)))
lstmP1 = nn.LSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1)
lstmPri = nn.Sequential():add(lstmP1):add(nn.NormStabilizer())
add_lstmPri:add(nn.Sequencer(lstmPri))
add_lstmPri:add(nn.CAddTable()) -- add linear
add_lstmPri:add( nn.MulConstant(1/seq_lenPri))
mdlPri:add(add_lstmPri)
nn.MaskZero(mdlPri,1)
mdlPri:cuda()

-- LSTM with output of hidden states averaged 2 without NarrowTable
ltPri2 = nn.LookupTableMaskZero(vsizePri,emb_size) 
fwdPri2 = nn.Sequential()
   :add(ltPri2)
   :add(nn.Linear(emb_size, hidden_size))
   :add(nn.LSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1))
   :add(nn.NormStabilizer())
   --:add(nn.Linear(hidden_size,emb_size))

mdlPri2 = nn.Sequential()
   :add(nn.Sequencer(fwdPri2))
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))
mdlPri2:cuda()


-- LSTM with short-cut connections and output of hidden states averaged 

duplicateInputSC = nn.ConcatTable()
  :add(nn.Identity())
  :add(nn.Identity())
sqntlSC = nn.Sequential()
  :add(nn.Linear(emb_size,hidden_size))
  :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
  :add(nn.NormStabilizer())
  :add(nn.Linear(hidden_size,emb_size))
prlSC = nn.ParallelTable()
  :add(sqntlSC)
  :add(nn.Identity())

 
ltPriSC = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,emb_size))

modPriSC = nn.ConcatTable()
for i = 1, seq_lenPri, win_size do -- seq length
    recPriSC = nn.NarrowTable(i,win_size)
    modPriSC:add(recPriSC)
end

add_lstmPriSC = nn:Sequential()
  :add(duplicateInputSC)
  :add(prlSC)

mdlPriSC = nn.Sequential()
mdlPriSC:add(ltPriSC)
mdlPriSC:add(modPriSC)
mdlPriSC:add(nn.Sequencer(add_lstmPriSC))
mdlPriSC:add(nn.FlattenTable()) 
mdlPriSC:add(nn.CAddTable()) -- add linear
mdlPriSC:add( nn.MulConstant(1/(2*seq_lenPri)))
nn.MaskZero(mdlPriSC,1)
mdlPriSC:cuda()

----------------
-- LSTM with short-cut connections and output of hidden states averaged  version 2 no NarrowTable

ltPriSC2 = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,emb_size))
duplicateInputSC2 = nn.ConcatTable()
duplicateInputSC2:add(nn.Identity())
duplicateInputSC2:add(nn.Identity())
prlSC2 = nn.ParallelTable()
sqntlSC2 = nn.Sequential()
sqntlSC2:add(nn.Linear(emb_size,hidden_size))
sqntlSC2:add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
sqntlSC2:add(nn.NormStabilizer())
sqntlSC2:add(nn.Linear(hidden_size,emb_size))
prlSC2:add(sqntlSC2)
prlSC2:add(nn.Identity())


fwdPriSC2 = nn.Sequential()
   :add(duplicateInputSC2)
   :add(prlSC2)

mdlPriSC2 = nn.Sequential()
   :add(ltPriSC2)
   :add(nn.Sequencer(fwdPriSC2))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))
mdlPriSC2:cuda()

--out = mdlPriSC:forward(xs1)
--out2 = mdlPriSC2:forward(xs1)

---------------------------------------

-- BiDirectional LSTM and output of hidden states averaged

nn.FastLSTM.bn = true
rho=nil
eps=nil
momentum=nil
affine=nil
p=0.25
mono=nil

sharedLookupTablePri = nn.LookupTableMaskZero(vsizePri, emb_size)

fwdPri = nn.Sequential()
   :add(sharedLookupTablePri)
   :add(nn.Linear(emb_size, hidden_size))
   --:add(nn.FastLSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1))
   :add(nn.Sequential():add(nn.LSTM(hidden_size,hidden_size):maskZero(1)):add(nn.NormStabilizer()))
   :add(nn.Linear(hidden_size,emb_size))

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
fwdPriSeq = nn.Sequential()
   :add(nn.Sequencer(fwdPri))

-- backward rnn (will be applied in reverse order of input sequence)
bwdPri = nn.Sequential()
   :add(sharedLookupTablePri:sharedClone())
   :add(nn.Linear(emb_size,hidden_size))
   --:add(nn.FastLSTM(hidden_size,hidden_size,rho, eps, momentum, affine, p, mono):maskZero(1))
   :add(nn.Sequential():add(nn.LSTM(hidden_size,hidden_size):maskZero(1)):add(nn.NormStabilizer()))
   :add(nn.Linear(hidden_size,emb_size))
  
bwdPriSeq = nn.Sequencer(bwdPri)

-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
mergePri = nn.JoinTable(1, 1)
--addTable = nn.CAddTable() 
mergePriSeq = nn.Sequencer(mergePri)

-- Assume that two input sequences are given (original and reverse, both are right-padded).
-- Instead of ConcatTable, we use ParallelTable here.
parallelPri = nn.ParallelTable()
parallelPri:add(fwdPriSeq):add(bwdPriSeq)
brnnPri = nn.Sequential()
   :add(parallelPri)
   :add(nn.ZipTable())
   :add(mergePriSeq)
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

brnnPri:cuda()

--out = brnnPri:forward({xs1,xs1r})

---------------------------------------------------------------------
--BiSeq GRU

emb_size = 64
hidden_size = 128
win_size = 1
seq_lenPri = 60
vsizePri = 3059
bsize = 10
lstm_flag = false
gru_flag = true
rnn_cnt = 2

lt = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,emb_size))
lstmBlF = nn.Sequential()
lstmBlF:add(nn.Linear(emb_size,hidden_size))
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

lstmBlB = nn.Sequential()
lstmBlB:add(nn.Linear(emb_size,hidden_size))
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

fwdSeq = nn.Sequencer(lstmBlF)
bwdSeq = nn.Sequencer(lstmBlB)
mergeSeq = nn.Sequencer(nn.JoinTable(1, 1))
   
local backward = nn.Sequential()
  :add(nn.ReverseTable()) -- reverse
  :add(bwdSeq)
  :add(nn.ReverseTable()) -- unreverse
   
local concat = nn.ConcatTable()
  :add(fwdSeq)
  :add(backward)
   
local biSeq = nn.Sequential()
  :add(lt)
  :add(concat)
  :add(nn.ZipTable())
  :add(mergeSeq)
  :add(nn.CAddTable())
  :add( nn.MulConstant(1/(2*seq_lenPri)))

x1 = torch.Tensor(bsize,seq_lenPri):random(1,vsizePri)
xs1 = split:forward(x1)

--l1 = lt:forward(xs1)

--a1 = fwdSeq:forward(l1)
--a2 = backward:forward(l1)

--o1 = concat:forward(l1)
--o2 = nn.ZipTable():forward(o1)
--o3 = mergeSeq:forward(o2)
--o4 = nn.CAddTable():forward(o3)
--o5 = nn.MulConstant(1/(2*seq_lenPri)):forward(o4)

out = biSeq:forward(xs1)

--------------------------------------------------------



-- BiDirectional LSTM  SC with short-cut connections and output of hidden states averaged

nn.FastLSTM.bn = true
rho=nil
eps=nil
momentum=nil
affine=nil
p=0.25
mono=nil

sharedLookupTablePriSC = nn.LookupTableMaskZero(vsizePri, emb_size)

fwdDuplicatePriSC = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
fwdLstmPriSC = nn.Sequential()
   :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
   :add(nn.NormStabilizer())
fwdSqntlPriSC = nn.Sequential()
   :add(nn.Linear(emb_size,hidden_size))
   :add(fwdLstmPriSC)
   :add(nn.Linear(hidden_size,emb_size))
fwdPrlPriSC = nn.ParallelTable()
   :add(fwdSqntlPriSC)
   :add(nn.Identity())


fwdPriSC = nn.Sequential()
   :add(sharedLookupTablePriSC)
   :add(fwdDuplicatePriSC)
   :add(fwdPrlPriSC)


-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
fwdPriSeqSC = nn.Sequential()
   :add(nn.Sequencer(fwdPriSC))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

-- backward rnn (will be applied in reverse order of input sequence)

bwdDuplicatePriSC = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
bwdLstmPriSC = nn.Sequential()
   :add(nn.LSTM(hidden_size,hidden_size):maskZero(1))
   :add(nn.NormStabilizer())
bwdSqntlPriSC = nn.Sequential()
   :add(nn.Linear(emb_size,hidden_size))
   :add(bwdLstmPriSC)
   :add(nn.Linear(hidden_size,emb_size))
bwdPrlPriSC = nn.ParallelTable()
   :add(bwdSqntlPriSC)
   :add(nn.Identity())


bwdPriSC = nn.Sequential()
   :add(sharedLookupTablePriSC:sharedClone())
   :add(bwdDuplicatePriSC)
   :add(bwdPrlPriSC)


-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
bwdPriSeqSC = nn.Sequential()
   :add(nn.Sequencer(bwdPriSC))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
mergePriSC = nn.JoinTable(1, 1)
--addTable = nn.CAddTable() 
--mergePriSeqSC = nn.Sequencer(mergePriSC)

-- Assume that two input sequences are given (original and reverse, both are right-padded).
-- Instead of ConcatTable, we use ParallelTable here.
parallelPriSC = nn.ParallelTable()
parallelPriSC:add(fwdPriSeqSC):add(bwdPriSeqSC)
brnnPriSC = nn.Sequential()
   :add(parallelPriSC)
   --:add(nn.ZipTable())
   :add(mergePriSC)
   --:add(nn.CAddTable()) -- add linear
   --:add( nn.MulConstant(1/seq_lenPri))

brnnPriSC:cuda()



--out = mdlPri:forward(xs)
--outSC = mdlPriSC:forward(xs)
out = brnnPriSC:forward({xs1,xs1r})

print("end")

