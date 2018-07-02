local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'

local win_size = 2
local w_init = 0.01  -- 0.01
local BiLangModel = torch.class("BiLangModel")

function BiLangModel:__init(vocab_sizePri,vocab_sizeSec,config)
  
  self.additivePri = nn.Sequential()
  self.ltPri = nn.LookupTableMaskZero(vocab_sizePri,config.emb_size) -- MaskZero
  self.additivePri:add(nn.Sequencer( self.ltPri))
  self.additivePri:add( nn.SplitTable(2))
  modPri = nn.ConcatTable()
  for i = 1, seq_lenPri-1, 1 do -- seq length
    recPri = nn.NarrowTable(i,win_size)
    modPri:add(recPri)
  end

  self.additivePri:add(modPri)
  add_tanPri = nn:Sequential()
  add_tanPri:add(nn.Sequencer(nn.CAddTable()))
  add_tanPri:add(nn.Sequencer(nn.Tanh()))
  add_tanPri:add(nn.CAddTable()) -- add linear layer
  self.additivePri:add(add_tanPri)
  --self.additivePri:add(nn.MaskZero(add_tanPri,1))
  
  self.additivePri:getParameters():uniform(-1*w_init,w_init)
  self.additivePri:cuda()

  self.additiveSec = nn.Sequential()
  self.ltSec = nn.LookupTableMaskZero(vocab_sizeSec,config.emb_size) -- MaskZero
  self.additiveSec:add(nn.Sequencer( self.ltSec))
  self.additiveSec:add( nn.SplitTable(2))
  modSec = nn.ConcatTable()
  for i = 1, seq_lenSec-1, 1 do -- seq length
    recSec = nn.NarrowTable(i,win_size)
    modSec:add(recSec)
  end

  self.additiveSec:add(modSec)
  add_tanSec = nn:Sequential()
  add_tanSec:add(nn.Sequencer(nn.CAddTable()))
  add_tanSec:add(nn.Sequencer(nn.Tanh()))
  add_tanSec:add(nn.CAddTable()) -- add linear layer
  self.additiveSec:add(add_tanSec)
  --self.additiveSec:add(nn.MaskZero(add_tanSec,1))
  
  self.additiveSec:getParameters():uniform(-1*w_init,w_init)
  self.additiveSec:cuda()
    

  self.criterion = nn.AbsCriterion():cuda()
  --self.criterion = nn.MaskZeroCriterion(self.criterion, 1):cuda()
end

function BiLangModel:getAdditivePri()
  return self.additivePri
end

function BiLangModel:getAdditiveSec()
  return self.additiveSec
end

function BiLangModel:getCriterion()
  return self.criterion
end

function BiLangModel:getLookupTablePri()
  return self.ltPri
end

function BiLangModel:getLookupTableSec()
  return self.ltSec
end
