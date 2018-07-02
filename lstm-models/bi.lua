require 'nn'
require 'cunn'
require 'rnn'
require 'optim'

--emb_size = 4
--hidden_size = 5
--win_size = 1
--seq_lenPri = 5--60

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

cutorch.setHeapTracking(true)
collectgarbage()

initTotalGPUMem = {} 
initFreeGPUMem = {} 

function initGPUTotalMem()
  --print('## Initial Mem on GPU ##') 
  for i=1,cutorch.getDeviceCount() do 
    initFreeGPUMem[i], initTotalGPUMem[i] = cutorch.getMemoryUsage(i) 
    --print(i, initFreeGPUMem[i]) 
  end
end

--initGPUTotalMem()

function printGPUMemUsage(step)
  print('## Mem Used on GPU ## '..step) 
  for i=1,cutorch.getDeviceCount() do 
    free, _ = cutorch.getMemoryUsage(i) 
    print("GPU : "..i.." free mem : "..free..", used :"..(initTotalGPUMem[i] - free).." , total mem : "..initTotalGPUMem[i]) 
  end
end

initGPUTotalMem()
printGPUMemUsage(10)


vsizePri = 3059
--bsize = 2
p=0.5

prm_list = { lr = 0.0001;prev_lr = 0;alpha = 0.95;lr_decay = 0;weight_decay = 0; lr_list = nil ;emb_size = 10;hidden_size = 15;num_of_hidden=3;out_gate=10;gate_type = 'LSTM';momentum=0.5; threshold = 5000 ;dump_frq = 50;max_epoch = 5000;epoch=0;batch_size = 1;init = 1;max_seq_len = 5;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}

split = nn.SplitTable(2)

x1 = torch.Tensor(prm_list["batch_size"],prm_list["max_seq_len"]):random(1,vsizePri):cuda()
xs1 = split:forward(x1)

--bi = nn.BiSequencer(nn.GRU(10, 10, nil, 0), nn.GRU(10, 10, nil, 0):sharedClone(), nn.JoinTable(1, 1))


if prm_list["gate_type"] == "GRU" then
  gate = nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
else
  gate = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
end

ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"])
mergeSeq = nn.Sequencer(nn.JoinTable(1, 1))
  
local forward = nn.Sequential()
  :add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  
for i=1,prm_list["num_of_hidden"]-1 do
  forward:add(gate:clone())
    :add(nn.NormStabilizer())
    :add(nn.Dropout(p))
end

forward:add(gate:clone())
  :add(nn.NormStabilizer())

local backward = nn.Sequential()
  :add(nn.ReverseTable()) -- reverse
  :add(nn.Sequencer(forward:clone()))
  :add(nn.ReverseTable()) -- unreverse
   
local concat = nn.ConcatTable()
  :add(nn.Sequencer(forward))
  :add(backward)
   
local biSeq = nn.Sequential()
  :add(nn.SplitTable(2))
  :add(nn.Sequencer(ltPri))
  :add(concat)
  :add(nn.ZipTable())
  :add(mergeSeq)
  :add(nn.CAddTable())
  :add( nn.MulConstant(1/(2*prm_list["max_seq_len"])))

--inp = nn.Sequencer(ltPri):forward(xs1)
--out1 = concat:forward(inp)
--out1f = nn.Sequencer(forward):forward(inp)
--out1b = backward:forward(inp)
--out2 = nn.ZipTable():forward(out1)
--out3 = mergeSeq:forward(out2)
--out4 = nn.CAddTable():forward(out3)
--out5 = nn.MulConstant(1/(2*seq_lenPri)):forward(out4)
--out = biSeq:forward(x1)

print("ok")


--LSTM with SC bw 2 lstms and AVG 

  ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"]) 
  duplicateInputPriSc = nn.ConcatTable()
  duplicateInputPriSc:add(nn.Identity())
  duplicateInputPriSc:add(nn.Identity())
  
  lstmBlockPriSc = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockPriSc:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  lstmBlockPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
    --:add(nn.Dropout(p))
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockPriSc:add(nn.Dropout(p))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  
  prlPriSc = nn.ParallelTable()
    :add(lstmBlockPriSc:clone())
    :add(nn.Identity())

  prllScPriSc = nn.Sequential()
    :add(duplicateInputPriSc:clone())
    :add(prlPriSc:clone())
    :add(nn.CAddTable()) 
    :add(nn.MulConstant(1/2))
  
  fwdPriSc = nn.Sequential()
    --:add(ltPri)
    :add(prllScPriSc:clone())

  merge = nn.Sequential()
    :add(nn.FlattenTable()) 
    :add(nn.CAddTable()) -- add linear
    :add(nn.MulConstant(1/prm_list["max_seq_len"]))

  mdlPri2 = nn.Sequential()
    :add(nn.Sequencer(ltPri))
    :add(nn.Sequencer(prllScPriSc:clone()))
    :add(nn.Sequencer(lstmBlockPriSc:clone()))
    :add(nn.Sequencer(nn.MulConstant(1/2)))
    :add(merge:clone())

    
  nn.MaskZero(mdlPri2,1)
  mdlPri2:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  mdlPri2:cuda()

--out1 = nn.Sequencer(ltPri):forward(xs1)
--out2 = nn.Sequencer(prllScPriSc:clone()):cuda():forward(out1)
--out3 = nn.Sequencer(lstmBlockPriSc:clone()):cuda():forward(out2)
--out4 = nn.Sequencer(nn.MulConstant(1/2)):cuda():forward(out3)
--out5 = merge:clone():cuda():forward(out4)

--out = mdlPri:forward(xs1)


--LSTM with SC bw 3 lstms and AVG 

  ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"]) 
  duplicateInputPriSc = nn.ConcatTable()
  duplicateInputPriSc:add(nn.Identity())
  duplicateInputPriSc:add(nn.Identity())
  
  lstmBlockPriSc = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockPriSc:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  lstmBlockPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockPriSc:add(nn.Dropout(p))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  
  prlPriSc = nn.ParallelTable()
    :add(lstmBlockPriSc:clone())
    :add(nn.Identity())

  prllScPriSc = nn.Sequential()
    :add(duplicateInputPriSc:clone())
    :add(prlPriSc:clone())
    :add(nn.CAddTable()) 
    :add(nn.MulConstant(1/2))
  
  fwdPriSc = nn.Sequential()
    :add(prllScPriSc:clone())

  merge = nn.Sequential()
    :add(nn.FlattenTable()) 
    :add(nn.CAddTable()) -- add linear
    :add(nn.MulConstant(1/prm_list["max_seq_len"]))

  mdlPri3y = nn.Sequential()
    :add(nn.Sequencer(ltPri))
    :add(nn.Sequencer(prllScPriSc:clone()))
    :add(nn.Sequencer(lstmBlockPriSc:clone()))
    :add(nn.Sequencer(nn.MulConstant(1/2)))
    :add(merge:clone())


lstmPrl = nn.ParallelTable()
      :add(lstmBlockPriSc:clone())
      :add(nn.Identity())

concat = nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Identity())

lstmExit = nn.Sequential()
      :add(concat:clone())
      :add(lstmPrl:clone())
      :add(nn.CAddTable())
      :add(nn.MulConstant(1/2))
      :add(lstmBlockPriSc:clone())

lstmPrl2= nn.ParallelTable()
      :add(lstmExit:clone())
      :add(nn.Identity())

lstmExit2 = nn.Sequential()
      :add(concat:clone())
      :add(lstmPrl2:clone())
      :add(nn.CAddTable()) 
      :add(nn.MulConstant(1/2))

lstmPrl3= nn.ParallelTable()
      :add(lstmExit2:clone())
      :add(nn.Identity())

fullBlock = nn.Sequential()
          :add(concat:clone())
          :add(lstmPrl3:clone())    
          :add(nn.CAddTable()) 
          :add(nn.MulConstant(1/2))
          :add(lstmBlockPriSc:clone())


mdlPri3 = nn.Sequential()
    :add(nn.Sequencer(ltPri))
    :add(nn.Sequencer(fullBlock))
    :add(merge:clone())

nn.MaskZero(mdlPri3,1)
mdlPri3:cuda()
--mdlPri3:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

-------------------------------------------------------------------------------
-- LSTM 2 layers with SC between layers and AVG

lstmBlockScPriN = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScPriN:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  lstmBlockScPriN:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScPriN:add(nn.Dropout(p))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  
prlSlctN = nn.ParallelTable()
      :add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
      :add(nn.NarrowTable(1,1))

concatN = nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Identity())

lstmLayerLastN = nn.ParallelTable()
      :add(lstmBlockScPriN:clone())
      :add(nn.Identity())

lstmLayerFirstN = nn.Sequential()
      :add(concatN:clone())
      :add(
          nn.ParallelTable()
          :add(lstmBlockScPriN:clone())
          :add(nn.Identity())
        )
      :add(concatN:clone())
      :add(prlSlctN:clone())
      :add(nn.FlattenTable()) 

lstmLayersN= nn.Sequential()
  :add(lstmLayerFirstN:clone())
  :add(lstmLayerLastN:clone())

mergeAvgN = nn.Sequential()
    :add(nn.FlattenTable()) 
    :add(nn.CAddTable()) 
    :add(nn.MulConstant(1/prm_list["max_seq_len"]))

mdlPri4 = nn.Sequential()
    :add(nn.Sequencer(ltPri))
    :add(nn.Sequencer(lstmLayersN))
    :add(mergeAvgN)

nn.MaskZero(mdlPri4,1)

--parameters,gradParameters = mdlPri4:parameters()
--for i=1,#parameters do
  --print(parameters[i]:size())
  --parameters[i]:uniform(-1*prm_list["init"],prm_list["init"])
--end


--mdlPri4:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
mdlPri4:cuda()
mdlPri4:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

--prl = nn.ParallelTable()
--          :add(lstmBlockPriScN:clone())
--          :add(nn.Identity())

--prl:cuda()
--mergeAvgN:cuda()
--prlSlctN:cuda()
--lstmLayerLastN:cuda()

--inp1 = nn.Sequencer(ltPri):forward(xs1)
--o1 = concatN:forward(inp1[1])
--o2 = prl:forward(o1)            
--o3 = concatN:forward(o2)
--o4 = prlSlctN:forward(o3)
--o5 = nn.FlattenTable():forward(o4)
--o6 = lstmLayerLastN:clone():forward(o5)
--o7 = mergeAvgN:forward(o6)

--out = mdlPri4:forward(xs1)

-------------------------------------------------------------------------------
-- LSTM 3 layers with SC between layers and AVG

lstmBlockScPriN2 = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScPriN2:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  lstmBlockScPriN2:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScPriN2:add(nn.Dropout(p))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  
prlSlctPriN2 = nn.ParallelTable()
      :add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
      :add(nn.NarrowTable(1,1))

concatPriN2 = nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Identity())

lstmLayerMidPriN2 = nn.ParallelTable()
      :add(lstmBlockScPriN2:clone())
      :add(nn.Identity())

lstmLayerLastPriN2 = nn.Sequential()
      :add(concatPriN2:clone())
      :add(prlSlctPriN2:clone())
      :add(lstmLayerMidPriN2:clone())

lstmLayerFirstPriN2 = nn.Sequential()
      :add(concatPriN2:clone())
      :add(
          nn.ParallelTable()
          :add(lstmBlockScPriN2:clone())
          :add(nn.Identity())
        )
      :add(concatPriN2:clone())
      :add(prlSlctPriN2:clone())
      :add(nn.FlattenTable()) 

lstmLayersPriN2= nn.Sequential()
  :add(lstmLayerFirstPriN2:clone())
  :add(lstmLayerMidPriN2:clone())
  :add(lstmLayerLastPriN2:clone())
  
mergeAvgPriN2 = nn.Sequential()
    :add(nn.FlattenTable()) 
    :add(nn.CAddTable()) 
    :add(nn.MulConstant(1/prm_list["max_seq_len"]))

mdlPri5 = nn.Sequential()
    :add(nn.Sequencer(ltPri))
    :add(nn.Sequencer(lstmLayersPriN2))
    :add(mergeAvgPriN2)

nn.MaskZero(mdlPri5,1)
mdlPri5:cuda()
mdlPri5:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

prlPriN2 = nn.ParallelTable()
          :add(lstmBlockScPriN2:clone())
          :add(nn.Identity())
          
prlPriN2:cuda()
prlSlctPriN2:cuda()
lstmLayerLastPriN2:cuda()
lstmLayerMidPriN2:cuda()

--inp = nn.Sequencer(ltPri):forward(xs1)
--o1 = concatN:forward(inp[1])
--o2 = prlPriN2:forward(o1)            
--o3 = concatPriN2:forward(o2)
--o4 = prlSlctPriN2:forward(o3)
--o5 = nn.FlattenTable():forward(o4)
--o6 = lstmLayerMidPriN2:clone():forward(o5)
--o9 = lstmLayerLastPriN2:forward(o6)

--o7 = concatPriN2:forward(o6)
--o8 = prlSlctPriN2:forward(o7)
--o9 = lstmLayerLastPriN2:clone():forward(o8)
--o10 = mergeAvgPriN2:forward(o9)

--inp = nn.Sequencer(ltPri):forward(xs1)
--out = mdlPri5:forward(xs1)


-----------------------------
--BiLangModelBiLSTMScAvg2

sharedLookupTablePriSc = nn.LookupTableMaskZero(vsizePri, prm_list["emb_size"])

  fwdDuplicatePriSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
  
  fwdSqntlPriSc = nn.Sequential()
  
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    fwdSqntlPriSc:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  end
  
  for i=1,prm_list["num_of_hidden"]-1 do
    fwdSqntlPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
      :add(nn.NormStabilizer())
      :add(nn.Dropout(dropoutProb))
  end

  fwdSqntlPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
        :add(nn.NormStabilizer())
        
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    fwdSqntlPriSc:add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))
  end
  
  fwdPrlPriSc = nn.ParallelTable()
   :add(fwdSqntlPriSc)
   :add(nn.Identity())


  fwdPriSc = nn.Sequential()
   --:add(self.sharedLookupTablePriSc)
   :add(fwdDuplicatePriSc)
   :add(fwdPrlPriSc)
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add(nn.MulConstant(1/2))


-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
 -- fwdPriSeqSc = nn.Sequential()
--   :add(nn.Sequencer(fwdPriSc))
   --:add(nn.CAddTable())
   --:add(nn.MulConstant(1/prm_list["max_seq_len"]))

   
-----------------
  bckPriSeqSc = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(nn.Sequencer(fwdPriSc:clone()))
    :add(nn.ReverseTable()) -- unreverse
    --:add(nn.CAddTable())
    --:add(nn.MulConstant(1/prm_list["max_seq_len"]))

   
  concatPriSc = nn.ConcatTable()
    :add(nn.Sequencer(fwdPriSc))
    :add(bckPriSeqSc)

  mergePriSc = nn.JoinTable(1, 1)

  biSeqPri = nn.Sequential()
    --:add(nn.SplitTable(2))
    :add(nn.Sequencer(sharedLookupTablePriSc))
    :add(concatPriSc)
    :add(nn.ZipTable())
    :add(nn.Sequencer(mergePriSc))
    :add(nn.CAddTable())
    :add(nn.MulConstant(1/prm_list["max_seq_len"]))
  
  biSeqPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  biSeqPri:cuda()
  
fwdPriSc:cuda()
concatPriSc:cuda()
fwdDuplicatePriSc:cuda()
fwdPrlPriSc:cuda()

inp = nn.Sequencer(sharedLookupTablePriSc):forward(xs1)

o1 = fwdDuplicatePriSc:forward(inp[1])
o2 = fwdPrlPriSc:forward(o1)
o3 = nn.FlattenTable():forward(o2)
o4 = nn.CAddTable():cuda():forward(o3)
o5 = nn.MulConstant(1/2):cuda():forward(o4)




o7 = concatPriSc:forward(inp)
o8 = nn.ZipTable():forward(o7)
o9 = nn.Sequencer(mergePriSc):forward(o8)

out = biSeqPri:forward(xs1)

print("end")


----------------------