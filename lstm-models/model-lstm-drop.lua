local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'

--local dropoutProb = 0.5

local BiLangModelLSTM = torch.class("BiLangModelLSTM")

local BiLangModelLSTMAvg = torch.class("BiLangModelLSTMAvg")
local BiLangModelLSTMAvgAttn = torch.class("BiLangModelLSTMAvgAttn")

local BiLangModelLSTMScAvg = torch.class("BiLangModelLSTMScAvg")
local BiLangModelLSTMScAttnAvg = torch.class("BiLangModelLSTMScAttnAvg")
local BiLangModelLSTM2LyScAvg = torch.class("BiLangModelLSTM2LyScAvg")
local BiLangModelLSTM3LyScAvg = torch.class("BiLangModelLSTM3LyScAvg")

local BiLangModelBiLSTMAvg = torch.class("BiLangModelBiLSTMAvg")
local BiLangModelBiLSTMAvg2 = torch.class("BiLangModelBiLSTMAvg2")

local BiLangModelBiLSTMScAvg = torch.class("BiLangModelBiLSTMScAvg")
local BiLangModelBiLSTMScAvg2 = torch.class("BiLangModelBiLSTMScAvg2")

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


function BiLangModelLSTM:__initOld(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
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
  lstmPri:add(nn.Dropout(prm_list["dropoutProb"])) 
  
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
  lstmSec:add(nn.Dropout(prm_list["dropoutProb"])) 
  
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
      :add(nn.Dropout(prm_list["dropoutProb"]))
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
      :add(nn.Dropout(prm_list["dropoutProb"]))
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
  return self.criterion
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
      :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  
  fwdPri:add(gate:clone())
    :add(nn.NormStabilizer())
    
  if prm_list["out_size"] ~= prm_list["hidden_size"] then
    fwdPri:add(nn.Linear(prm_list["hidden_size"], prm_list["out_size"]))
  end

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
      :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  
  fwdSec:add(gate:clone())
    :add(nn.NormStabilizer())

  if prm_list["out_size"] ~= prm_list["hidden_size"] then
    fwdSec:add(nn.Linear(prm_list["hidden_size"], prm_list["out_size"]))
  end
  
  self.mdlSec = nn.Sequential()
   :add(nn.Sequencer(fwdSec))
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenSec))
   
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

function BiLangModelLSTMAvgAttn:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
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
      :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  
  fwdPri:add(gate:clone())
    :add(nn.NormStabilizer())
  
  self.attnW1Pri = nn.Linear(prm_list["hidden_size"],prm_list["attnW1Out"])
  self.attnW2Pri = nn.Linear(prm_list["attnW1Out"],prm_list["attnW2Out"])

  self.attnLayerPri = AttnLayer(self.attnW1Pri,self.attnW2Pri)
    
  self.mdlPri = nn.Sequential()
   :add(nn.Sequencer(fwdPri))
   :add(self.attnLayerPri)
   :add(nn.View(prm_list["batch_size"],prm_list["attnW2Out"]*prm_list["hidden_size"]))
   
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
      :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  
  fwdSec:add(gate:clone())
    :add(nn.NormStabilizer())

  self.attnW1Sec = nn.Linear(prm_list["hidden_size"],prm_list["attnW1Out"])
  self.attnW2Sec = nn.Linear(prm_list["attnW1Out"],prm_list["attnW2Out"])

  self.attnLayerSec = AttnLayer(self.attnW1Sec,self.attnW2Sec)
  
  self.mdlSec = nn.Sequential()
   :add(nn.Sequencer(fwdSec))
   :add(self.attnLayerSec)
   :add(nn.View(prm_list["batch_size"],prm_list["attnW2Out"]*prm_list["hidden_size"]))

  self.mdlSec:cuda()
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
end

function BiLangModelLSTMAvgAttn:attnW1Pri()
  return self.attnW1Pri
end

function BiLangModelLSTMAvgAttn:attnW2Pri()
  return self.attnW2Pri
end

function BiLangModelLSTMAvgAttn:attnW1Sec()
  return self.attnW1Sec
end

function BiLangModelLSTMAvgAttn:attnW2Sec()
  return self.attnW2Sec
end


function BiLangModelLSTMAvgAttn:modelPri()
  return self.mdlPri
end

function BiLangModelLSTMAvgAttn:modelSec()
  return self.mdlSec
end

function BiLangModelLSTMAvgAttn:getCriterion()
  return self.criterion
end

function BiLangModelLSTMAvgAttn:getLookupTablePri()
  return self.ltPri
end

function BiLangModelLSTMAvgAttn:getLookupTableSec()
  return self.ltSec
end

---------------------------------------------------------------------



function BiLangModelLSTMScAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  dropoutProb = 0.9
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
  for i=1,prm_list["num_of_hidden"]-1 do 
      self.sqntlPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
        :add(nn.NormStabilizer())
        :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  self.sqntlPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  
  --self.sqntlPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
  --  :add(nn.NormStabilizer())
  --  :add(nn.Dropout(dropoutProb))
  --  :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
  --  :add(nn.NormStabilizer())
  --  :add(nn.Dropout(dropoutProb))
  --  :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
  --  :add(nn.NormStabilizer())
  
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    self.sqntlPriSc:add(nn.Dropout(prm_list["dropoutProb"]))
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
  for i=1,prm_list["num_of_hidden"]-1 do 
      self.sqntlSecSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
        :add(nn.NormStabilizer())
        :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  self.sqntlSecSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  
  --self.sqntlSecSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
  --  :add(nn.NormStabilizer())
  --  :add(nn.Dropout(dropoutProb))
  --  :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
  --  :add(nn.NormStabilizer())
  --  :add(nn.Dropout(dropoutProb))
  --  :add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
  --  :add(nn.NormStabilizer())
  
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    self.sqntlSecSc:add(nn.Dropout(prm_list["dropoutProb"]))
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


---------------------------------------------------------------------

function BiLangModelLSTMScAttnAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  dropoutProb = 0.9
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
  for i=1,prm_list["num_of_hidden"]-1 do 
      self.sqntlPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
        :add(nn.NormStabilizer())
        :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  self.sqntlPriSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    self.sqntlPriSc:add(nn.Dropout(prm_list["dropoutProb"]))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  self.prlPriSc:add(self.sqntlPriSc)
  self.prlPriSc:add(nn.Identity())

  self.fwdPriSc = nn.Sequential()
    :add(self.ltPri)
    :add(self.duplicateInputPriSc)
    :add(self.prlPriSc)
    :add(nn.CAddTable())

  local attnMapPri = nn.Linear(seq_lenPri,1)
  local attnLayerPri = nn.Sequential()
        :add(nn.Sequencer(nn.Unsqueeze(1)))
        :add(nn.Sequencer(nn.SplitTable(2)))
        :add(nn.ZipTable())
        :add(nn.Sequencer(nn.JoinTable(1,2)))
        :add(nn.Sequencer(nn.Transpose({1,2})))
        :add(nn.Sequencer(attnMapPri))
        :add(nn.Sequencer(nn.Transpose({1,2})))
        :add(nn.JoinTable(1))

  self.mdlPri = nn.Sequential()
    :add(nn.Sequencer(self.fwdPriSc))
    :add(attnLayerPri)
        
    --:add(nn.FlattenTable()) 
    --:add(nn.CAddTable()) -- add linear
    --:add( nn.MulConstant(1/(2*seq_lenPri)))
    
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
  for i=1,prm_list["num_of_hidden"]-1 do 
      self.sqntlSecSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
        :add(nn.NormStabilizer())
        :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  self.sqntlSecSc:add(nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1))
    :add(nn.NormStabilizer())
  
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    self.sqntlSecSc:add(nn.Dropout(prm_list["dropoutProb"]))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  self.prlSecSc:add(self.sqntlSecSc)
  self.prlSecSc:add(nn.Identity())


  self.fwdSecSc = nn.Sequential()
    :add(self.ltSec)
    :add(self.duplicateInputSecSc)
    :add(self.prlSecSc)

  local attnMapSec = nn.Linear(seq_lenPri,1)
  local attnLayerSec = nn.Sequential()
        :add(nn.Sequencer(nn.Unsqueeze(1)))
        :add(nn.Sequencer(nn.SplitTable(2)))
        :add(nn.ZipTable())
        :add(nn.Sequencer(nn.JoinTable(1,2)))
        :add(nn.Sequencer(nn.Transpose({1,2})))
        :add(nn.Sequencer(attnMapSec))
        :add(nn.Sequencer(nn.Transpose({1,2})))
        :add(nn.JoinTable(1))

  self.mdlSec = nn.Sequential()
    :add(nn.Sequencer(self.fwdSecSc))
    :add(attnLayerSec)

    --:add(nn.FlattenTable()) 
    --:add(nn.CAddTable()) -- add linear
    --:add( nn.MulConstant(1/(2*seq_lenSec)))
    
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  
end


function BiLangModelLSTMScAttnAvg:modelPri()
  return self.mdlPri
end

function BiLangModelLSTMScAttnAvg:modelSec()
  return self.mdlSec
end

function BiLangModelLSTMScAttnAvg:getCriterion()
  return self.criterion
end

function BiLangModelLSTMScAttnAvg:getLookupTablePri()
  return self.ltPri
end

function BiLangModelLSTMScAttnAvg:getLookupTableSec()
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
  self.biSeqPri = nn.Sequential()
   :add(parallelPri)
   :add(nn.ZipTable())
   :add(mergePriSeq)
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenPri))

  self.biSeqPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.biSeqPri:cuda()
     
   
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
  self.biSeqSec = nn.Sequential()
   :add(parallelSec)
   :add(nn.ZipTable())
   :add(mergeSecSeq)
   :add(nn.CAddTable()) -- add linear
   :add( nn.MulConstant(1/seq_lenSec))

  self.biSeqSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.biSeqSec:cuda()
   
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
end


function BiLangModelBiLSTMAvg:modelPri()
  return self.biSeqPri
end

function BiLangModelBiLSTMAvg:modelSec()
  return self.biSeqSec
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
  mono=nil

  
  if prm_list["gate_type"] == "GRU" then
    gate = nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  else
    gate = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  end

  mergeSeq = nn.Sequencer(nn.JoinTable(1, 1))
  
  self.ltPri = nn.LookupTableMaskZero(vsizePri,prm_list["emb_size"])
  
  local forwardPri = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    forwardPri:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  end  
    
  for i=1,prm_list["num_of_hidden"]-1 do
    forwardPri:add(gate:clone())
      :add(nn.NormStabilizer())
      :add(nn.Dropout(prm_list["dropoutProb"]))
  end

  forwardPri:add(gate:clone())
    :add(nn.NormStabilizer())

  if prm_list["hidden_size"] ~= prm_list["out_size"] then
    forwardPri:add(nn.Linear(prm_list["hidden_size"],prm_list["out_size"]))
  end  
  
  local backwardPri = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(nn.Sequencer(forwardPri:clone()))
    :add(nn.ReverseTable()) -- unreverse
   
  local concatPri = nn.ConcatTable()
    :add(nn.Sequencer(forwardPri))
    :add(backwardPri)
   
  self.biSeqPri = nn.Sequential()
    --:add(nn.SplitTable(2))
    :add(nn.Sequencer(self.ltPri))
    :add(concatPri)
    :add(nn.ZipTable())
    :add(mergeSeq)
    :add(nn.CAddTable())
    :add(nn.MulConstant(1/(2*seq_lenPri)))
  
  nn.MaskZero(self.biSeqPri,1)
  self.biSeqPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.biSeqPri:cuda()

  
  self.ltSec = nn.LookupTableMaskZero(vsizeSec,prm_list["emb_size"])
  
  local forwardSec = nn.Sequential()
  
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    forwardSec:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  end 
  
  for i=1,prm_list["num_of_hidden"]-1 do
    forwardSec:add(gate:clone())
      :add(nn.NormStabilizer())
      :add(nn.Dropout(prm_list["dropoutProb"]))
  end

  forwardSec:add(gate:clone())
    :add(nn.NormStabilizer())

  if prm_list["hidden_size"] ~= prm_list["out_size"] then
    forwardSec:add(nn.Linear(prm_list["hidden_size"],prm_list["out_size"]))
  end 
  
  local backwardSec = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(nn.Sequencer(forwardSec:clone()))
    :add(nn.ReverseTable()) -- unreverse
   
  local concatSec = nn.ConcatTable()
    :add(nn.Sequencer(forwardSec))
    :add(backwardSec)
   
  self.biSeqSec = nn.Sequential()
    --:add(nn.SplitTable(2))
    :add(nn.Sequencer(self.ltSec))
    :add(concatSec)
    :add(nn.ZipTable())
    :add(mergeSeq)
    :add(nn.CAddTable())
    :add(nn.MulConstant(1/(2*seq_lenSec)))
  
  nn.MaskZero(self.biSeqSec,1)
  self.biSeqSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
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
   :add(self.sharedLookupTablePriSc)
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
   :add(self.sharedLookupTablePriSc:sharedClone())
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
  self.biSeqPri = nn.Sequential()
   :add(parallelPriSc)
   --:add(nn.ZipTable())
   :add(mergePriSc)
   --:add(nn.CAddTable()) -- add linear
   --:add( nn.MulConstant(1/seq_lenPri))
  self.biSeqPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.biSeqPri:cuda()
  
    
  self.sharedLookupTableSecSc = nn.LookupTableMaskZero(vsizeSec, prm_list["emb_size"])

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
   :add(self.sharedLookupTableSecSc)
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
   :add(self.sharedLookupTableSecSc:sharedClone())
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
  self.biSeqSec = nn.Sequential()
   :add(parallelSecSc)
   --:add(nn.ZipTable())
   :add(mergeSecSc)
   --:add(nn.CAddTable()) -- add linear
   --:add( nn.MulConstant(1/seq_lenPri))
  self.biSeqSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.biSeqSec:cuda()
  
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  
end


function BiLangModelBiLSTMScAvg:modelPri()
  return self.biSeqPri
end

function BiLangModelBiLSTMScAvg:modelSec()
  return self.biSeqSec
end

function BiLangModelBiLSTMScAvg:getCriterion()
  return self.criterion
end

function BiLangModelBiLSTMScAvg:getLookupTablePri()
  return self.sharedLookupTablePriSc
end

function BiLangModelBiLSTMScAvg:getLookupTableSec()
  return self.sharedLookupTableSecSc
end


--------------------------------------------------------------------------

function BiLangModelBiLSTMScAvg2:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  mono=nil

  if prm_list["gate_type"] == "GRU" then
    gate = nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  else
    gate = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  end
  
  self.sharedLookupTablePriSc = nn.LookupTableMaskZero(vsizePri, prm_list["emb_size"])

  fwdDuplicatePriSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
   
  fwdSqntlPriSc = nn.Sequential()
  
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    fwdSqntlPriSc:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  end
  
  for i=1,prm_list["num_of_hidden"]-1 do
    fwdSqntlPriSc:add(gate:clone())
      :add(nn.NormStabilizer())
      :add(nn.Dropout(prm_list["dropoutProb"]))
  end
  
  fwdSqntlPriSc:add(gate:clone())
        :add(nn.NormStabilizer())
        
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    fwdSqntlPriSc:add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))
  end
  
  fwdPrlPriSc = nn.ParallelTable()
   :add(fwdSqntlPriSc)
   :add(nn.Identity())

  fwdPriSc = nn.Sequential()
   :add(fwdDuplicatePriSc)
   :add(fwdPrlPriSc)
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add(nn.MulConstant(1/2))

  bckPriSeqSc = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(nn.Sequencer(fwdPriSc:clone()))
    :add(nn.ReverseTable()) -- unreverse
   
  concatPriSc = nn.ConcatTable()
    :add(nn.Sequencer(fwdPriSc))
    :add(bckPriSeqSc)

  mergePriSc = nn.JoinTable(1, 1)

  self.biSeqPri = nn.Sequential()
    :add(nn.Sequencer(self.sharedLookupTablePriSc))
    :add(concatPriSc)
    :add(nn.ZipTable())
    :add(nn.Sequencer(mergePriSc))
    :add(nn.CAddTable())
    :add(nn.MulConstant(1/prm_list["max_seq_len"]))


  self.biSeqPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.biSeqPri:cuda()
  
    
  self.sharedLookupTableSecSc = nn.LookupTableMaskZero(vsizeSec, prm_list["emb_size"])

  fwdDuplicateSecSc = nn.ConcatTable()
   :add(nn.Identity())
   :add(nn.Identity())
   
  fwdSqntlSecSc = nn.Sequential()
  
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    fwdSqntlSecSc:add(nn.Linear(prm_list["emb_size"],prm_list["hidden_size"]))
  end
  
  for i=1,prm_list["num_of_hidden"]-1 do
    fwdSqntlSecSc:add(gate:clone())
      :add(nn.NormStabilizer())
      :add(nn.Dropout(prm_list["dropoutProb"]))
  end

  fwdSqntlSecSc:add(gate:clone())
        :add(nn.NormStabilizer())
        
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    fwdSqntlSecSc:add(nn.Linear(prm_list["hidden_size"],prm_list["emb_size"]))
  end
  
  fwdPrlSecSc = nn.ParallelTable()
   :add(fwdSqntlSecSc)
   :add(nn.Identity())

  fwdSecSc = nn.Sequential()
   :add(fwdDuplicateSecSc)
   :add(fwdPrlSecSc)
   :add(nn.FlattenTable()) 
   :add(nn.CAddTable()) -- add linear
   :add(nn.MulConstant(1/2))

  bckSecSeqSc = nn.Sequential()
    :add(nn.ReverseTable()) -- reverse
    :add(nn.Sequencer(fwdSecSc:clone()))
    :add(nn.ReverseTable()) -- unreverse
   
  concatSecSc = nn.ConcatTable()
    :add(nn.Sequencer(fwdSecSc))
    :add(bckSecSeqSc)

  mergeSecSc = nn.JoinTable(1, 1)

  self.biSeqSec = nn.Sequential()
    :add(nn.Sequencer(self.sharedLookupTableSecSc))
    :add(concatSecSc)
    :add(nn.ZipTable())
    :add(nn.Sequencer(mergeSecSc))
    :add(nn.CAddTable())
    :add(nn.MulConstant(1/prm_list["max_seq_len"]))

  self.biSeqSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])
  self.biSeqSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  
end


function BiLangModelBiLSTMScAvg2:modelPri()
  return self.biSeqPri
end

function BiLangModelBiLSTMScAvg2:modelSec()
  return self.biSeqSec
end

function BiLangModelBiLSTMScAvg2:getCriterion()
  return self.criterion
end

function BiLangModelBiLSTMScAvg2:getLookupTablePri()
  return self.sharedLookupTablePriSc
end

function BiLangModelBiLSTMScAvg2:getLookupTableSec()
  return self.sharedLookupTableSecSc
end



-------------------------------------------------------------------------------
-- LSTM 2 layers with SC between layers and AVG

function BiLangModelLSTM2LyScAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  mono=nil

  if prm_list["gate_type"] == "GRU" then
    gate = nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  else
    gate = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  end
  
  self.ltPri = nn.LookupTableMaskZero(vsizePri, prm_list["emb_size"])

  lstmBlockScPri = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
      lstmBlockScPri:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  lstmBlockScPri:add(gate:clone())
      :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScPri:add(nn.Dropout(prm_list["dropoutProb"]))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  
  prlSlctPri = nn.ParallelTable()
      :add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
      :add(nn.NarrowTable(1,1))

  concatPri = nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Identity())

  lstmLayerLastPri = nn.ParallelTable()
      :add(lstmBlockScPri:clone())
      :add(nn.Identity())

  lstmLayerFirstPri = nn.Sequential()
      :add(concatPri:clone())
      :add(
          nn.ParallelTable()
          :add(lstmBlockScPri:clone())
          :add(nn.Identity())
        )
      :add(concatPri:clone())
      :add(prlSlctPri:clone())
      :add(nn.FlattenTable()) 

  lstmLayersPri= nn.Sequential()
      :add(lstmLayerFirstPri:clone())
      :add(lstmLayerLastPri:clone())

  mergeAvgPri = nn.Sequential()
      :add(nn.FlattenTable()) 
      :add(nn.CAddTable()) 
      :add(nn.MulConstant(1/prm_list["max_seq_len"]))

  self.mdlPri = nn.Sequential()
      :add(nn.Sequencer(self.ltPri))
      :add(nn.Sequencer(lstmLayersPri))
      :add(mergeAvgPri)

  nn.MaskZero(self.mdlPri,1)
  self.mdlPri:cuda()
  self.mdlPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])


  self.ltSec = nn.LookupTableMaskZero(vsizeSec, prm_list["emb_size"])

  lstmBlockScSec = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
      lstmBlockScSec:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  lstmBlockScSec:add(gate:clone())
      :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScSec:add(nn.Dropout(prm_list["dropoutProb"]))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  
  prlSlctSec = nn.ParallelTable()
      :add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
      :add(nn.NarrowTable(1,1))

  concatSec = nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Identity())

  lstmLayerLastSec = nn.ParallelTable()
      :add(lstmBlockScSec:clone())
      :add(nn.Identity())

  lstmLayerFirstSec = nn.Sequential()
      :add(concatSec:clone())
      :add(
          nn.ParallelTable()
          :add(lstmBlockScSec:clone())
          :add(nn.Identity())
        )
      :add(concatSec:clone())
      :add(prlSlctSec:clone())
      :add(nn.FlattenTable()) 

  lstmLayersSec= nn.Sequential()
      :add(lstmLayerFirstSec:clone())
      :add(lstmLayerLastSec:clone())

  mergeAvgSec = nn.Sequential()
      :add(nn.FlattenTable()) 
      :add(nn.CAddTable()) 
      :add(nn.MulConstant(1/prm_list["max_seq_len"]))

  self.mdlSec = nn.Sequential()
      :add(nn.Sequencer(self.ltSec))
      :add(nn.Sequencer(lstmLayersSec))
      :add(mergeAvgSec)

  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:cuda()
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)

end

function BiLangModelLSTM2LyScAvg:modelPri()
  return self.mdlPri
end

function BiLangModelLSTM2LyScAvg:modelSec()
  return self.mdlSec
end

function BiLangModelLSTM2LyScAvg:getCriterion()
  return self.criterion
end

function BiLangModelLSTM2LyScAvg:getLookupTablePri()
  return self.ltPri
end

function BiLangModelLSTM2LyScAvg:getLookupTableSec()
  return self.ltSec
end



-------------------------------------------------------------------------------
-- LSTM 3 layers with SC between layers and AVG

function BiLangModelLSTM3LyScAvg:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,prm_list)  
  nn.FastLSTM.bn = true
  rho=nil
  eps=nil
  momentum=nil
  affine=nil
  mono=nil

  if prm_list["gate_type"] == "GRU" then
    gate = nn.GRU(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  else
    gate = nn.LSTM(prm_list["hidden_size"],prm_list["hidden_size"]):maskZero(1)
  end
  
  self.ltPri = nn.LookupTableMaskZero(vsizePri, prm_list["emb_size"])

  lstmBlockScPri = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScPri:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  lstmBlockScPri:add(gate:clone())
    :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScPri:add(nn.Dropout(prm_list["dropoutProb"]))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  
  prlSlctPri = nn.ParallelTable()
      :add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
      :add(nn.NarrowTable(1,1))

  concatPri = nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Identity())

  lstmLayerMidPri = nn.ParallelTable()
      :add(lstmBlockScPri:clone())
      :add(nn.Identity())

  lstmLayerLastPri = nn.Sequential()
      :add(concatPri:clone())
      :add(prlSlctPri:clone())
      :add(lstmLayerMidPri:clone())

  lstmLayerFirstPri = nn.Sequential()
      :add(concatPri:clone())
      :add(
          nn.ParallelTable()
          :add(lstmBlockScPri:clone())
          :add(nn.Identity())
        )
      :add(concatPri:clone())
      :add(prlSlctPri:clone())
      :add(nn.FlattenTable()) 

  lstmLayersPri= nn.Sequential()
      :add(lstmLayerFirstPri:clone())
      :add(lstmLayerMidPri:clone())
      :add(lstmLayerLastPri:clone())
  
  mergeAvgPri = nn.Sequential()
      :add(nn.FlattenTable()) 
      :add(nn.CAddTable()) 
      :add(nn.MulConstant(1/prm_list["max_seq_len"]))

  self.mdlPri = nn.Sequential()
      :add(nn.Sequencer(self.ltPri))
      :add(nn.Sequencer(lstmLayersPri))
      :add(mergeAvgPri)

  nn.MaskZero(self.mdlPri,1)
  self.mdlPri:cuda()
  self.mdlPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])



  self.ltSec = nn.LookupTableMaskZero(vsizeSec, prm_list["emb_size"])

  lstmBlockScSec = nn.Sequential()
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScSec:add(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"]))
  end
  lstmBlockScSec:add(gate:clone())
    :add(nn.NormStabilizer())
  if prm_list["emb_size"] ~= prm_list["hidden_size"] then
    lstmBlockScSec:add(nn.Dropout(prm_list["dropoutProb"]))
      :add(nn.Linear(prm_list["hidden_size"], prm_list["emb_size"]))
  end
  
  prlSlctSec = nn.ParallelTable()
      :add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CAddTable()))
      :add(nn.NarrowTable(1,1))

  concatSec = nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Identity())

  lstmLayerMidSec = nn.ParallelTable()
      :add(lstmBlockScSec:clone())
      :add(nn.Identity())

  lstmLayerLastSec = nn.Sequential()
      :add(concatSec:clone())
      :add(prlSlctSec:clone())
      :add(lstmLayerMidSec:clone())

  lstmLayerFirstSec = nn.Sequential()
      :add(concatSec:clone())
      :add(
          nn.ParallelTable()
          :add(lstmBlockScSec:clone())
          :add(nn.Identity())
        )
      :add(concatSec:clone())
      :add(prlSlctSec:clone())
      :add(nn.FlattenTable()) 

  lstmLayersSec= nn.Sequential()
      :add(lstmLayerFirstSec:clone())
      :add(lstmLayerMidSec:clone())
      :add(lstmLayerLastSec:clone())
  
  mergeAvgSec = nn.Sequential()
      :add(nn.FlattenTable()) 
      :add(nn.CAddTable()) 
      :add(nn.MulConstant(1/prm_list["max_seq_len"]))

  self.mdlSec = nn.Sequential()
      :add(nn.Sequencer(self.ltSec))
      :add(nn.Sequencer(lstmLayersSec))
      :add(mergeAvgSec)

  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:cuda()
  self.mdlSec:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)

end

function BiLangModelLSTM3LyScAvg:modelPri()
  return self.mdlPri
end

function BiLangModelLSTM3LyScAvg:modelSec()
  return self.mdlSec
end

function BiLangModelLSTM3LyScAvg:getCriterion()
  return self.criterion
end

function BiLangModelLSTM3LyScAvg:getLookupTablePri()
  return self.ltPri
end

function BiLangModelLSTM3LyScAvg:getLookupTableSec()
  return self.ltSec
end

