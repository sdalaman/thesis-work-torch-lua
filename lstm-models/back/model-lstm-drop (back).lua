local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'

local dropoutProb = 0.5

local BiLangModelLSTM = torch.class("BiLangModelLSTM")

local BiLangModelLSTMAvg = torch.class("BiLangModelLSTMAvg")
local BiLangModelLSTMScAvg = torch.class("BiLangModelLSTMScAvg")

local BiLangModelBiLSTMAvg = torch.class("BiLangModelBiLSTMAvg")
local BiLangModelBiLSTMAvg2 = torch.class("BiLangModelBiLSTMAvg2")
local BiLangModelBiLSTMScAvg = torch.class("BiLangModelBiLSTMScAvg")

function BiLangModelLSTMOld:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  self.mdlPri = nn.Sequential()
  self.ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"]) -- MaskZero
  self.mdlPri:add(nn.Sequencer(self.ltPri))

  modPri = nn.ConcatTable()
  for i = 1, seq_lenPri, prm_list["win_size"] do -- seq length
    recPri = nn.NarrowTable(i,prm_list["win_size"])
    modPri:add(recPri)
  end

  self.mdlPri:add(modPri)
  add_lstmPri = nn:Sequential()
  add_lstmPri:add(nn.Sequencer(nn.CAddTable()))
    
  add_lstmPri:add(nn.Sequencer(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"])))
  lstmP1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  --lstmP1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"],nil,nil,dropoutProb,nil):maskZero(1)
  
  lstmPri = nn.Sequential():add(lstmP1):add(nn.NormStabilizer())
  lstmPri:add(nn.Dropout(dropoutProb)) 
  
  add_lstmPri:add(nn.Sequencer(lstmPri))
--  if prm_list["hidden_size"] > 0 then
--    add_lstmPri:add(nn.Sequencer(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"])))
--  end
  add_lstmPri:add(nn.SelectTable(-1)) -- add linear
  self.mdlPri:add(add_lstmPri)
  nn.MaskZero(self.mdlPri,1)
  self.mdlPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

  self.mdlPri:cuda()

  -- Build the second model
  self.mdlSec = nn.Sequential()
  self.ltSec = nn.LookupTableMaskZero(vsizeSec,prm_list["emb_size"]) -- MaskZero
  self.mdlSec:add(nn.Sequencer(self.ltSec))

  modSec = nn.ConcatTable()
  for i = 1, seq_lenSec, prm_list["win_size"] do -- seq length
    recSec = nn.NarrowTable(i,prm_list["win_size"])
    modSec:add(recSec)
  end

  self.mdlSec:add(modSec)
  add_lstmSec = nn:Sequential()
  add_lstmSec:add(nn.Sequencer(nn.CAddTable()))
  --lstmS = nn.FastLSTM(prm_list["emb_size"],prm_list["emb_size"])
  add_lstmSec:add(nn.Sequencer(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"])))
  lstmS1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  --lstmS1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"],nil,nil,dropoutProb,nil):maskZero(1)
  lstmSec = nn.Sequential():add(lstmS1):add(nn.NormStabilizer()) 
  lstmSec:add(nn.Dropout(dropoutProb)) 
  
  add_lstmSec:add(nn.Sequencer(lstmSec))
  
--  if prm_list["hidden_size"] > 0 then
--    add_lstmSec:add(nn.Sequencer(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"])))
--  end
  add_lstmSec:add(nn.SelectTable(-1))
  self.mdlSec:add(add_lstmSec)
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  --return self.mdlPri,self.mdlSec,self.ltPri,self.ltSec
end

function BiLangModelLSTM:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  
  if prm_list["gate_type"] == "GRU" then
    gate = nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  else
    gate = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  end
  
  self.ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"]) 
  fwdPri = nn.Sequential()
   :add(self.ltPri)
   
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
   fwdPri:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  
  for i=1,prm_list["num_of_hidden"]-1 do
    fwdPri:add(gate:clone())
      :add(nn.NormStabilizer())
      :add(nn.Dropout(p))
  end
  
  fwdPri:add(gate:clone())
    :add(nn.NormStabilizer())

  self.mdlPri = nn.Sequential()
   :add(nn.Sequencer(fwdPri))
   :add(nn.SelectTable(-1))
  
  nn.MaskZero(self.mdlPri,1)
  self.mdlPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.mdlPri:cuda()

  
  -- Build the second model
  self.ltSec = nn.LookupTableMaskZero(vsizeSec,prm_list["emb_size"]) 
  fwdSec = nn.Sequential()
   :add(self.ltSec)
   
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
   fwdSec:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  
  for i=1,prm_list["num_of_hidden"]-1 do
    fwdSec:add(gate:clone())
      :add(nn.NormStabilizer())
      :add(nn.Dropout(p))
  end
  
  fwdSec:add(gate:clone())
    :add(nn.NormStabilizer())

  self.mdlSec = nn.Sequential()
   :add(nn.Sequencer(fwdSec))
   :add(nn.SelectTable(-1))
  
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  --return self.mdlPri,self.mdlSec,self.ltPri,self.ltSec
end

function BiLangModelLSTM:modelPri()
  return self.mdlPri
end

function BiLangModelLSTM:modelSec()
  return self.mdlSec
end

function BiLangModelLSTM:getCriterion()
  return s3333elf.criterion
end

function BiLangModelLSTM:getLookupTablePri()
  return self.ltPri
end

function BiLangModelLSTM:getLookupTableSec()
  return self.ltSec
end


---------------------------------------------

function BiLangModelLSTMAvg:__initOld(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  p=0.25
  mono=nil

  
  self.mdlPri = nn.Sequential()
  self.ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"]) -- MaskZero
  self.mdlPri:add(nn.Sequencer(self.ltPri))

  modPri = nn.ConcatTable()
  for i = 1, seq_lenPri, prm_list["win_size"] do -- seq length
    recPri = nn.NarrowTable(i,prm_list["win_size"])
    modPri:add(recPri)
  end

  self.mdlPri:add(modPri)
  add_lstmPri = nn:Sequential()
  add_lstmPri:add(nn.Sequencer(nn.CAddTable()))
    
  add_lstmPri:add(nn.Sequencer(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"])))
  lstmP1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"],rho, eps, momentum, affine, p, mono):maskZero(1)
  --lstmP1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"],nil,nil,dropoutProb,nil):maskZero(1)
  
  lstmPri = nn.Sequential():add(lstmP1):add(nn.NormStabilizer())
  --lstmPri:add(nn.Dropout(dropoutProb)) 
  
  add_lstmPri:add(nn.Sequencer(lstmPri))
--  if prm_list["hidden_size"] > 0 then
--    add_lstmPri:add(nn.Sequencer(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"])))
--  end
  add_lstmPri:add(nn.CAddTable()) -- add linear
  add_lstmPri:add( nn.MulConstant(1/seq_lenPri))
  self.mdlPri:add(add_lstmPri)
  nn.MaskZero(self.mdlPri,1)
  self.mdlPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

  self.mdlPri:cuda()

  -- Build the second model
  self.mdlSec = nn.Sequential()
  self.ltSec = nn.LookupTableMaskZero(vsizeSec,prm_list["emb_size"]) -- MaskZero
  self.mdlSec:add(nn.Sequencer(self.ltSec))

  modSec = nn.ConcatTable()
  for i = 1, seq_lenSec, prm_list["win_size"] do -- seq length
    recSec = nn.NarrowTable(i,prm_list["win_size"])
    modSec:add(recSec)
  end

  self.mdlSec:add(modSec)
  add_lstmSec = nn:Sequential()
  add_lstmSec:add(nn.Sequencer(nn.CAddTable()))
  --lstmS = nn.FastLSTM(prm_list["emb_size"],prm_list["emb_size"])
  add_lstmSec:add(nn.Sequencer(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"])))
  lstmS1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"],rho, eps, momentum, affine, p, mono):maskZero(1)
  --lstmS1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"],nil,nil,dropoutProb,nil):maskZero(1)
  lstmSec = nn.Sequential():add(lstmS1):add(nn.NormStabilizer()) 
  --lstmSec:add(nn.Dropout(dropoutProb)) 
  
  add_lstmSec:add(nn.Sequencer(lstmSec))
  
--  if prm_list["hidden_size"] > 0 then
--    add_lstmSec:add(nn.Sequencer(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"])))
--  end
  add_lstmSec:add(nn.CAddTable()) -- add linear
  add_lstmSec:add( nn.MulConstant(1/seq_lenSec))
  self.mdlSec:add(add_lstmSec)
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  --return self.mdlPri,self.mdlSec,self.ltPri,self.ltSec
end

function BiLangModelLSTMAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  p=0.5
  mono=nil
  cell2gate = nil

  
  if prm_list["gate_type"] == "GRU" then
    gate = nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  else
    gate = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  end
  
  self.ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"]) 
  fwdPri = nn.Sequential()
   :add(self.ltPri)
   
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
   fwdPri:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  
  for i=1,prm_list["num_of_hidden"]-1 do
    fwdPri:add(gate:clone())
      :add(nn.NormStabilizer())
      :add(nn.Dropout(p))
  end
  
  fwdPri:add(gate:clone())
    :add(nn.NormStabilizer())

  self.mdlPri = nn.Sequential()
   :add(nn.Sequencer(fwdPri))
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))
  nn.MaskZero(self.mdlPri,1)
  self.mdlPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.mdlPri:cuda()
  
  self.ltSec = nn.LookupTableMaskZero(vsizeSec,prm_list["emb_size"]) 
  fwdSec = nn.Sequential()
   :add(self.ltSec)
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
   fwdSec:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  
  for i=1,prm_list["num_of_hidden"]-1 do
    fwdSec:add(gate:clone())
      :add(nn.NormStabilizer())
      :add(nn.Dropout(p))
  end
  
  fwdSec:add(gate:clone())
    :add(nn.NormStabilizer())

  self.mdlSec = nn.Sequential()
   :add(nn.Sequencer(fwdSec))
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenSec))
  self.mdlSec:cuda()
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
end


function BiLangModelLSTMAvg:modelPri()
  return self.mdlPri
end

function BiLangModelLSTMAvg:modelSec()
  return self.mdlSec
end

function BiLangModelLSTMAvg:getCriterion()
  return self.criterion
end

function BiLangModelLSTMAvg:getLookupTablePri()
  return self.ltPri
end

function BiLangModelLSTMAvg:getLookupTableSec()
  return self.ltSec
end

---------------------------------------------------------------------

function BiLangModelLSTMScAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  p=0.5
  mono=nil
  cell2gate = nil

  self.ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"]) 
  self.duplicateInputPriSc = nn.ConcatTable()
  self.duplicateInputPriSc:add(nn.Identity())
  self.duplicateInputPriSc:add(nn.Identity())
  self.prlPriSc = nn.ParallelTable()
  self.sqntlPriSc = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    self.sqntlPriSc:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  self.sqntlPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
    :add(nn.Dropout(p))
    :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
    :add(nn.Dropout(p))
    :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    self.sqntlPriSc:add(nn.Dropout(p))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  self.prlPriSc:add(self.sqntlPriSc)
  self.prlPriSc:add(nn.Identity())

  self.fwdPriSc = nn.Sequential()
    :add(self.ltPri)
    :add(self.duplicateInputPriSc)
    :add(self.prlPriSc)

  self.mdlPri = nn.Sequential()
    :add(nn.Sequencer(self.fwdPriSc))
    :add(nn.FlattenTable()) 
    :add(nn.CAddTable()) -- add linear
    :add( nn.MulConstant(1/(2*seq_lenPri)))
  nn.MaskZero(self.mdlPri,1)
  self.mdlPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.mdlPri:cuda()

  -- Build the second model
  
  self.ltSec = nn.LookupTableMaskZero(vsizeSec,prm_list["emb_size"]) 
  self.duplicateInputSecSc = nn.ConcatTable()
  self.duplicateInputSecSc:add(nn.Identity())
  self.duplicateInputSecSc:add(nn.Identity())
  self.prlSecSc = nn.ParallelTable()
  self.sqntlSecSc = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    self.sqntlSecSc:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  self.sqntlSecSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
    :add(nn.Dropout(p))
    :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
    :add(nn.Dropout(p))
    :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    self.sqntlSecSc:add(nn.Dropout(p))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  self.prlSecSc:add(self.sqntlSecSc)
  self.prlSecSc:add(nn.Identity())


  self.fwdSecSc = nn.Sequential()
    :add(self.ltSec)
    :add(self.duplicateInputSecSc)
    :add(self.prlSecSc)

  self.mdlSec = nn.Sequential()
    :add(nn.Sequencer(self.fwdSecSc))
    :add(nn.FlattenTable()) 
    :add(nn.CAddTable()) -- add linear
    :add( nn.MulConstant(1/(2*seq_lenSec)))
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  
end


function BiLangModelLSTMScAvg:modelPri()
  return self.mdlPri
end

function BiLangModelLSTMScAvg:modelSec()
  return self.mdlSec
end

function BiLangModelLSTMScAvg:getCriterion()
  return self.criterion
end

function BiLangModelLSTMScAvg:getLookupTablePri()
  return self.ltPri
end

function BiLangModelLSTMScAvg:getLookupTableSec()
  return self.ltSec
end


---------------------------------------------

function BiLangModelBiLSTMAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
    
  self.sharedLookupTablePri = nn.LookupTableMaskZero(vsizePri, prm_list["emb_size"])

  -- forward rnn
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  p=0.25
  mono=nil

  fwdLstmPri = nn.Sequential()
    :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  fwdPri = nn.Sequential()
    :add(self.sharedLookupTablePri)
    :add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
    :add(fwdLstmPri)

  -- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  fwdPriSeq = nn.Sequencer(fwdPri)

  -- backward rnn (will be applied in reverse order of input sequence)
  bwdLstmPri = nn.Sequential()
    :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
    
  bwdPri = nn.Sequential()
   :add(self.sharedLookupTablePri:sharedClone())
   :add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
   :add(bwdLstmPri)
  
  bwdPriSeq = nn.Sequencer(bwdPri)

  -- merges the output of one time-step of fwd and bwd rnns.
  -- You could also try nn.AddTable(), nn.Identity(), etc.
  mergePri = nn.JoinTable(1, 1) 
  mergePriSeq = nn.Sequencer(mergePri)

  -- Assume that two input sequences are given (original and reverse, both are right-padded).
  -- Instead of ConcatTable, we use ParallelTable here.
  parallelPri = nn.ParallelTable()
  parallelPri:add(fwdPriSeq):add(bwdPriSeq)
  self.brnnPri = nn.Sequential()
   :add(parallelPri)
   :add(nn.ZipTable())
   :add(mergePriSeq)
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

  self.brnnPri:cuda()
     
   
  self.sharedLookupTableSec = nn.LookupTableMaskZero(vsizeSec, prm_list["emb_size"])

  -- forward rnn
  fwdLstmSec = nn.Sequential()
    :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
    
  fwdSec = nn.Sequential()
    :add(self.sharedLookupTableSec)
    :add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
    :add(fwdLstmSec)

  -- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  fwdSecSeq = nn.Sequencer(fwdSec)

  -- backward rnn (will be applied in reverse order of input sequence)
  bwdLstmSec = nn.Sequential()
    :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
    
  bwdSec = nn.Sequential()
   :add(self.sharedLookupTableSec:sharedClone())
   :add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
   :add(bwdLstmSec)
  
  bwdSecSeq = nn.Sequencer(bwdSec)

  -- merges the output of one time-step of fwd and bwd rnns.
  -- You could also try nn.AddTable(), nn.Identity(), etc.
  mergeSec = nn.JoinTable(1, 1) 
  mergeSecSeq = nn.Sequencer(mergeSec)

  -- Assume that two input sequences are given (original and reverse, both are right-padded).
  -- Instead of ConcatTable, we use ParallelTable here.
  parallelSec = nn.ParallelTable()
  parallelSec:add(fwdSecSeq):add(bwdSecSeq)
  self.brnnSec = nn.Sequential()
   :add(parallelSec)
   :add(nn.ZipTable())
   :add(mergeSecSeq)
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenSec))

  self.brnnSec:cuda()
   
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
end


function BiLangModelBiLSTMAvg:modelPri()
  return self.brnnPri
end

function BiLangModelBiLSTMAvg:modelSec()
  return self.brnnSec
end

function BiLangModelBiLSTMAvg:getCriterion()
  return self.criterion
end

function BiLangModelBiLSTMAvg:getLookupTablePri()
  return self.sharedLookupTablePri
end

function BiLangModelBiLSTMAvg:getLookupTableSec()
  return self.sharedLookupTableSec
end


function BiLangModelBiLSTMAvg2:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list) 
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

  
  self.ltPri = nn.Sequencer(nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"]))
  lstmFrwPri = nn.Sequential()
  lstmFrwPri:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  for i=1,rnn_cnt do
     if lstm_flag == true then
      lstmFrwPri:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
     end 
     if gru_flag == true then
      lstmFrwPri:add(nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
     end
     lstmFrwPri:add(nn.NormStabilizer())
  end
  lstmFrwPri:add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))

  lstmBwdPri = nn.Sequential()
  lstmBwdPri:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  for i=1,rnn_cnt do
     if lstm_flag == true then
      lstmBwdPri:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
     end 
     if gru_flag == true then
      lstmBwdPri:add(nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
     end
     lstmBwdPri:add(nn.NormStabilizer())
  end
  lstmBwdPri:add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))

  fwdPriSeq = nn.Sequencer(lstmFrwPri)
  bwdPriSeq = nn.Sequencer(lstmBwdPri)
  mergePriSeq = nn.Sequencer(nn.JoinTable(1, 1))
   
  backwardPri = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(bwdPriSeq)
    :add(nn.ReverseTable()) -- unreverse
   
  concatPri = nn.ConcatTable()
    :add(fwdPriSeq)
    :add(backwardPri)
   
  self.biSeqPri = nn.Sequential()
    :add(self.ltPri)
    :add(concatPri)
    :add(nn.ZipTable())
    :add(mergePriSeq)
    :add(nn.CAddTable())
    :add( nn.MulConstant(1/(2*seq_lenPri)))
  
  self.biSeqPri:cuda()
  
  self.ltSec = nn.Sequencer(nn.LookupTableMaskZero(vsizeSec,prm_list["emb_size"]))
  lstmFrwSec = nn.Sequential()
  lstmFrwSec:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  for i=1,rnn_cnt do
     if lstm_flag == true then
      lstmFrwSec:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
     end 
     if gru_flag == true then
      lstmFrwSec:add(nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
     end
     lstmFrwSec:add(nn.NormStabilizer())
  end
  lstmFrwSec:add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))

  lstmBwdSec = nn.Sequential()
  lstmBwdSec:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  for i=1,rnn_cnt do
     if lstm_flag == true then
      lstmBwdSec:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
     end 
     if gru_flag == true then
      lstmBwdSec:add(nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
     end
     lstmBwdSec:add(nn.NormStabilizer())
  end
  lstmBwdSec:add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))

  fwdSecSeq = nn.Sequencer(lstmFrwSec)
  bwdSecSeq = nn.Sequencer(lstmBwdSec)
  mergeSecSeq = nn.Sequencer(nn.JoinTable(1, 1))
   
  backwardSec = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(bwdSecSeq)
    :add(nn.ReverseTable()) -- unreverse
   
  concatSec = nn.ConcatTable()
    :add(fwdSecSeq)
    :add(backwardSec)
   
  self.biSeqSec = nn.Sequential()
    :add(self.ltSec)
    :add(concatSec)
    :add(nn.ZipTable())
    :add(mergeSecSeq)
    :add(nn.CAddTable())
    :add( nn.MulConstant(1/(2*seq_lenSec)))
  
  self.biSeqSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  
end

function BiLangModelBiLSTMAvg2:modelPri()
  return self.biSeqPri
end

function BiLangModelBiLSTMAvg2:modelSec()
  return self.biSeqSec
end

function BiLangModelBiLSTMAvg2:getCriterion()
  return self.criterion
end

function BiLangModelBiLSTMAvg2:getLookupTablePri()
  return self.ltPri
end

function BiLangModelBiLSTMAvg2:getLookupTableSec()
  return self.ltSec
end

---------------------------------------------------------------------

function BiLangModelBiLSTMScAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  p=0.25
  mono=nil

  self.sharedLookupTablePriSc = nn.LookupTableMaskZero(vsizePri, prm_list["emb_size"])

  fwdDuplicatePriSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
  fwdLstmPriSc = nn.Sequential()
   :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
   :add(nn.NormStabilizer())
  fwdSqntlPriSc = nn.Sequential()
   :add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
   :add(fwdLstmPriSc)
   :add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))
  fwdPrlPriSc = nn.ParallelTable()
   :add(fwdSqntlPriSc)
   :add(nn.Identity())


  fwdPriSc = nn.Sequential()
   :add(self.sharedLookupTablePri)
   :add(fwdDuplicatePriSc)
   :add(fwdPrlPriSc)


-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  fwdPriSeqSc = nn.Sequential()
   :add(nn.Sequencer(fwdPriSc))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

-- backward rnn (will be applied in reverse order of input sequence)

  bwdDuplicatePriSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
  bwdLstmPriSc = nn.Sequential()
   :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
   :add(nn.NormStabilizer())
  bwdSqntlPriSc = nn.Sequential()
   :add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
   :add(bwdLstmPriSc)
   :add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))
  bwdPrlPriSc = nn.ParallelTable()
   :add(bwdSqntlPriSc)
   :add(nn.Identity())


  bwdPriSc = nn.Sequential()
   :add(self.sharedLookupTablePri:sharedClone())
   :add(bwdDuplicatePriSc)
   :add(bwdPrlPriSc)


-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  bwdPriSeqSc = nn.Sequential()
   :add(nn.Sequencer(bwdPriSc))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
  mergePriSc = nn.JoinTable(1, 1)
--addTable = nn.CAddTable() 
--mergePriSeqSC = nn.Sequencer(mergePriSC)

-- Assume that two input sequences are given (original and reverse, both are right-padded).
-- Instead of ConcatTable, we use ParallelTable here.
  parallelPriSc = nn.ParallelTable()
  parallelPriSc:add(fwdPriSeqSc):add(bwdPriSeqSc)
  self.brnnPri = nn.Sequential()
   :add(parallelPriSc)
   --:add(nn.ZipTable())
   :add(mergePriSc)
   --:add(nn.CAddTable()) -- add linear
   --:add( nn.MulConstant(1/seq_lenPri))

  self.brnnPri:cuda()
  
    
  self.sharedLookupTableSec = nn.LookupTableMaskZero(vsizeSec, prm_list["emb_size"])

  fwdDuplicateSecSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
  fwdLstmSecSc = nn.Sequential()
   :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
   :add(nn.NormStabilizer())
  fwdSqntlSecSc = nn.Sequential()
   :add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
   :add(fwdLstmSecSc)
   :add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))
  fwdPrlSecSc = nn.ParallelTable()
   :add(fwdSqntlSecSc)
   :add(nn.Identity())


  fwdSecSc = nn.Sequential()
   :add(self.sharedLookupTableSec)
   :add(fwdDuplicateSecSc)
   :add(fwdPrlSecSc)


-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  fwdSecSeqSc = nn.Sequential()
   :add(nn.Sequencer(fwdSecSc))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenSec))

-- backward rnn (will be applied in reverse order of input sequence)

  bwdDuplicateSecSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
  bwdLstmSecSc = nn.Sequential()
   :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
   :add(nn.NormStabilizer())
  bwdSqntlSecSc = nn.Sequential()
   :add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
   :add(bwdLstmSecSc)
   :add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))
  bwdPrlSecSc = nn.ParallelTable()
   :add(bwdSqntlSecSc)
   :add(nn.Identity())


  bwdSecSc = nn.Sequential()
   :add(self.sharedLookupTableSec:sharedClone())
   :add(bwdDuplicateSecSc)
   :add(bwdPrlSecSc)


-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  bwdSecSeqSc = nn.Sequential()
   :add(nn.Sequencer(bwdSecSc))
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenSec))

-- merges the output of one time-step of fwd and bwd rnns.
-- You could also try nn.AddTable(), nn.Identity(), etc.
  mergeSecSc = nn.JoinTable(1, 1)
--addTable = nn.CAddTable() 
--mergePriSeqSC = nn.Sequencer(mergePriSC)

-- Assume that two input sequences are given (original and reverse, both are right-padded).
-- Instead of ConcatTable, we use ParallelTable here.
  parallelSecSc = nn.ParallelTable()
  parallelSecSc:add(fwdSecSeqSc):add(bwdSecSeqSc)
  self.brnnSec = nn.Sequential()
   :add(parallelSecSc)
   --:add(nn.ZipTable())
   :add(mergeSecSc)
   --:add(nn.CAddTable()) -- add linear
   --:add( nn.MulConstant(1/seq_lenPri))

  self.brnnSec:cuda()
  
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  
end


function BiLangModelBiLSTMScAvg:modelPri()
  return self.brnnPri
end

function BiLangModelBiLSTMScAvg:modelSec()
  return self.brnnSec
end

function BiLangModelBiLSTMScAvg:getCriterion()
  return self.criterion
end

function BiLangModelBiLSTMScAvg:getLookupTablePri()
  return self.sharedLookupTablePri
end

function BiLangModelBiLSTMScAvg:getLookupTableSec()
  return self.sharedLookupTableSec
end



