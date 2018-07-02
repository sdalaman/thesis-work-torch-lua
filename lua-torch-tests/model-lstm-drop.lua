local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'

local dropoutProb = 0.6

local BiLangModelLSTM = torch.class("BiLangModelLSTM")
local BiLangModelLSTMAvg = torch.class("BiLangModelLSTMAvg")
local BiLangModelLSTMBiAvg = torch.class("BiLangModelBiLSTMAvg")

function BiLangModelLSTM:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
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


function BiLangModelLSTM:modelPri()
  return self.mdlPri
end

function BiLangModelLSTM:modelSec()
  return self.mdlSec
end

function BiLangModelLSTM:getCriterion()
  return self.criterion
end

function BiLangModelLSTM:getLookupTablePri()
  return self.ltPri
end

function BiLangModelLSTM:getLookupTableSec()
  return self.ltSec
end


---------------------------------------------

function BiLangModelLSTMAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
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
  lstmS1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  --lstmS1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"],nil,nil,dropoutProb,nil):maskZero(1)
  lstmSec = nn.Sequential():add(lstmS1):add(nn.NormStabilizer()) 
  lstmSec:add(nn.Dropout(dropoutProb)) 
  
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

  fwdPri = nn.Sequential()
   :add(self.sharedLookupTablePri)
   :add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
   :add(nn.FastLSTM(prm_list["hidden_size"], prm_list["hidden_size"],rho, eps, momentum, affine, p, mono):maskZero(1))

  -- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  fwdPriSeq = nn.Sequencer(fwdPri)

  -- backward rnn (will be applied in reverse order of input sequence)
  bwdPri = nn.Sequential()
   :add(self.sharedLookupTablePri:sharedClone())
   :add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
   :add(nn.FastLSTM(prm_list["hidden_size"], prm_list["hidden_size"],rho, eps, momentum, affine, p, mono):maskZero(1))
  
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
  fwdSec = nn.Sequential()
   :add(self.sharedLookupTableSec)
   :add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
   :add(nn.FastLSTM(prm_list["hidden_size"], prm_list["hidden_size"],rho, eps, momentum, affine, p, mono):maskZero(1))

  -- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  fwdSecSeq = nn.Sequencer(fwdSec)

  -- backward rnn (will be applied in reverse order of input sequence)
  bwdSec = nn.Sequential()
   :add(self.sharedLookupTableSec:sharedClone())
   :add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
   :add(nn.FastLSTM(prm_list["hidden_size"], prm_list["hidden_size"],rho, eps, momentum, affine, p, mono):maskZero(1))
  
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



--[[function BiLangModelLSTM:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
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
  --lstmP = nn.FastLSTM(prm_list["emb_size"],prm_list["emb_size"])
  if prm_list["hidden_size"] > 0 then
    add_lstmPri:add(nn.Sequencer(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"])))
    lstmP1 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
    lstmP2 = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  else
    lstmP = nn.LSTM(prm_list["emb_size"],prm_list["emb_size"]):maskZero(1)
  end
  lstmPri = nn.Sequential():add(lstmP1):add(nn.NormStabilizer()):add(nn.Dropout(dropoutProb)) 
  lstmPri:add(lstmP2):add(nn.NormStabilizer()):add(nn.Dropout(dropoutProb)) 
  --lstmPri:maskZero(1)
  add_lstmPri:add(nn.Sequencer(lstmPri))
  if prm_list["hidden_size"] > 0 then
    add_lstmPri:add(nn.Sequencer(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"])))
  end
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
  if prm_list["hidden_size"] > 0 then
    add_lstmSec:add(nn.Sequencer(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"])))
    lstmS = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  else
    lstmS = nn.LSTM(prm_list["emb_size"],prm_list["emb_size"]):maskZero(1)
  end
  lstmSec = nn.Sequential():add(lstmS):add(nn.NormStabilizer())
  --lstmSec:maskZero(1)
  add_lstmSec:add(nn.Sequencer(lstmSec))
  add_lstmSec:add(nn.Sequencer(nn.Dropout(dropoutProb))) 
  if prm_list["hidden_size"] > 0 then
    add_lstmSec:add(nn.Sequencer(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"])))
  end
  add_lstmSec:add(nn.SelectTable(-1))
  self.mdlSec:add(add_lstmSec)
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  --return self.mdlPri,self.mdlSec,self.ltPri,self.ltSec
end
]]--

