local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'
require 'nngraph'
require 'optim'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities


local dropoutProb = 0.6

local BiLangModelLSTM = torch.class("BiLangModelLSTM")

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
  --lstmP = nn.FastLSTM(prm_list["emb_size"],prm_list["emb_size"])
  add_lstmPri:add(nn.Sequencer(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"])))
  lstmP1 = nn.MaskZero(LSTM.lstm(prm_list["hidden_size"],prm_list["hidden_size"]),1)
  lstmPri = nn.Sequential():add(lstmP1):add(nn.NormStabilizer()):add(nn.Dropout(dropoutProb)) 
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
  add_lstmSec:add(nn.Sequencer(nn.Linear(prm_list["emb_size"], prm_list["hidden_size"])))
  lstmS1 = nn.MaskZero(LSTM.lstm(prm_list["hidden_size"],prm_list["hidden_size"]),1)
  lstmSec = nn.Sequential():add(lstmS1):add(nn.NormStabilizer()):add(nn.Dropout(dropoutProb)) 
  add_lstmSec:add(nn.Sequencer(lstmSec))
  
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

