local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'

local BiLangModelLSTM = torch.class("BiLangModelLSTM")

local BiLangModel2 = torch.class("BiLangModel2")

function BiLangModel2:__init(vocab_sizePri,vocab_sizeSec,config)
  
  self.additivePri = nn.Sequential()
  self.ltPri = nn.LookupTableMaskZero(vocab_sizePri,config.emb_size) -- MaskZero
  self.additivePri:add(nn.Sequencer( self.ltPri))
  self.lstmPri = nn.Sequential():add(nn.FastLSTM(config.emb_size, 128)):add(nn.NormStabilizer())
  self.additivePri:add(nn.SplitTable(2,3))
  self.additivePri:add(nn.Sequencer(self.lstmPri))
  self.additivePri:add(nn.SelectTable(-1))
  self.additivePri:add(nn.Dropout(0.5))
  self.additivePri:add(nn.Linear(128,config.emb_size))
  --self.additivePri:add(nn.ReLU())
  --self.additivePri:add(nn.Linear(config.emb_size, config.emb_size))
  self.additivePri:add(nn.HardTanh())
  
  self.additivePri:getParameters():uniform(-1*0.01 ,0.01 )
  self.additivePri:cuda()

  self.additiveSec = nn.Sequential()
  self.ltSec = nn.LookupTableMaskZero(vocab_sizeSec,config.emb_size) -- MaskZero
  self.additiveSec:add(nn.Sequencer( self.ltSec))
  self.lstmSec = nn.Sequential():add(nn.FastLSTM(config.emb_size, 128)):add(nn.NormStabilizer())
  self.additiveSec:add(nn.SplitTable(2,3))
  self.additiveSec:add(nn.Sequencer(self.lstmSec))
  self.additiveSec:add(nn.SelectTable(-1))
  self.additiveSec:add(nn.Dropout(0.5))
  self.additiveSec:add(nn.Linear(128,config.emb_size))
  --self.additiveSec:add(nn.ReLU())
  --self.additiveSec:add(nn.Linear(config.emb_size, config.emb_size))
  self.additiveSec:add(nn.HardTanh())
  
  self.additiveSec:getParameters():uniform(-1*0.01 ,0.01 )
  self.additiveSec:cuda()
    

  self.criterion = nn.AbsCriterion():cuda()
  --self.criterion = nn.MaskZeroCriterion(self.criterion, 1):cuda()
end

function BiLangModelLSTM:__init(vsizePri,vsizeSec,seq_lenPri,seq_lenSec,cfg)  
  self.mdlPri = nn.Sequential()
  self.ltPri = nn.LookupTableMaskZero(vsizePri,cfg.emb_size) -- MaskZero
  self.mdlPri:add(nn.Sequencer(self.ltPri))

  modPri = nn.ConcatTable()
  for i = 1, seq_lenPri, cfg.win_size do -- seq length
    recPri = nn.NarrowTable(i,cfg.win_size)
    modPri:add(recPri)
  end

  self.mdlPri:add(modPri)
  add_lstmPri = nn:Sequential()
  add_lstmPri:add(nn.Sequencer(nn.CAddTable()))
  --lstmP = nn.FastLSTM(prm_list["emb_size"],prm_list["emb_size"])
  if cfg.hdd_size > 0 then
    add_lstmPri:add(nn.Sequencer(nn.Linear(cfg.emb_size, cfg.hdd_size)))
    lstmP = nn.LSTM(cfg.hdd_size,cfg.hdd_size):maskZero(1)
  else
    lstmP = nn.LSTM(cfg.emb_size,cfg.emb_size):maskZero(1)
  end
  lstmPri = nn.Sequential():add(lstmP):add(nn.NormStabilizer())
  --lstmPri:maskZero(1)
  add_lstmPri:add(nn.Sequencer(lstmPri))
  if cfg.hdd_size > 0 then
    add_lstmPri:add(nn.Sequencer(nn.Linear(cfg.hdd_size, cfg.emb_size)))
  end
  add_lstmPri:add(nn.SelectTable(-1)) -- add linear
  self.mdlPri:add(add_lstmPri)
  nn.MaskZero(self.mdlPri,1)
  self.mdlPri:getParameters():uniform(-1*cfg.init,cfg.init)

  self.mdlPri:cuda()

  -- Build the second model
  self.mdlSec = nn.Sequential()
  self.ltSec = nn.LookupTableMaskZero(vsizeSec,cfg.emb_size) -- MaskZero
  self.mdlSec:add(nn.Sequencer(self.ltSec))

  modSec = nn.ConcatTable()
  for i = 1, seq_lenSec, cfg.win_size do -- seq length
    recSec = nn.NarrowTable(i,cfg.win_size)
    modSec:add(recSec)
  end

  self.mdlSec:add(modSec)
  add_lstmSec = nn:Sequential()
  add_lstmSec:add(nn.Sequencer(nn.CAddTable()))
  --lstmS = nn.FastLSTM(prm_list["emb_size"],prm_list["emb_size"])
  if cfg.hdd_size > 0 then
    add_lstmSec:add(nn.Sequencer(nn.Linear(cfg.emb_size, cfg.hdd_size)))
    lstmS = nn.LSTM(cfg.hdd_size,cfg.hdd_size):maskZero(1)
  else
    lstmS = nn.LSTM(cfg.emb_size,cfg.emb_size):maskZero(1)
  end
  lstmSec = nn.Sequential():add(lstmS):add(nn.NormStabilizer())
  --lstmSec:maskZero(1)
  add_lstmSec:add(nn.Sequencer(lstmSec))
  if cfg.hdd_size > 0 then
    add_lstmSec:add(nn.Sequencer(nn.Linear(cfg.hdd_size, cfg.emb_size)))
  end
  add_lstmSec:add(nn.SelectTable(-1))
  self.mdlSec:add(add_lstmSec)
  nn.MaskZero(self.mdlSec,1)
  self.mdlSec:getParameters():uniform(-1*cfg.init,cfg.init)

  self.mdlSec:cuda()
  
  self.criterion = nn.AbsCriterion():cuda()
  nn.MaskZeroCriterion(self.criterion, 1)
  --return self.mdlPri,self.mdlSec,self.ltPri,self.ltSec
end


function BiLangModelLSTM:getModelPri()
  return self.mdlPri
end

function BiLangModelLSTM:getModelSec()
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


