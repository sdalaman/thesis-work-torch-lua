local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'

--local win_size = 6 --2
local w_init = 0.01  -- 0.01
local BiLangModelTanh = torch.class("BiLangModelTanh")

function BiLangModelTanh:__init(vocab_sizePri,vocab_sizeSec,config)
  if vocab_sizePri ~= 0 then
    self.legPri = nn.Sequential()
    self.ltPri = nn.LookupTableMaskZero(vocab_sizePri,config.emb_size) -- MaskZero
    self.legPri:add(nn.Sequencer( self.ltPri))
    self.legPri:add( nn.SplitTable(2))
    modPri = nn.ConcatTable()
    for i = 1, config.seq_lenPri-config.win_size+1, 1 do -- seq length
      recPri = nn.NarrowTable(i,config.win_size)
      modPri:add(recPri)
    end

    self.legPri:add(modPri)
    add_tanPri = nn:Sequential()
    add_tanPri:add(nn.Sequencer(nn.CAddTable()))
    add_tanPri:add(nn.Sequencer(nn.Tanh()))
    add_tanPri:add(nn.CAddTable()) -- add linear layer
    self.legPri:add(add_tanPri)
    --self.additivePri:add(nn.MaskZero(add_tanPri,1))
  
    self.legPri:getParameters():uniform(-1*w_init,w_init)
    self.legPri:cuda()
  else
    self.legPri = {}
    self.ltPri = {}
  end

  self.legSec = nn.Sequential()
  self.ltSec = nn.LookupTableMaskZero(vocab_sizeSec,config.emb_size) -- MaskZero
  self.legSec:add(nn.Sequencer( self.ltSec))
  self.legSec:add( nn.SplitTable(2))
  modSec = nn.ConcatTable()
  for i = 1, config.seq_lenSec-config.win_size+1, 1 do -- seq length
    recSec = nn.NarrowTable(i,config.win_size)
    modSec:add(recSec)
  end

  self.legSec:add(modSec)
  add_tanSec = nn:Sequential()
  add_tanSec:add(nn.Sequencer(nn.CAddTable()))
  add_tanSec:add(nn.Sequencer(nn.Tanh()))
  add_tanSec:add(nn.CAddTable()) -- add linear layer
  self.legSec:add(add_tanSec)
  --self.additiveSec:add(nn.MaskZero(add_tanSec,1))
  
  self.legSec:getParameters():uniform(-1*w_init,w_init)
  self.legSec:cuda()
    

  self.criterion = nn.AbsCriterion():cuda()
  --self.criterion = nn.MaskZeroCriterion(self.criterion, 1):cuda()
end

function BiLangModelTanh:getLegPri()
  return self.legPri
end

function BiLangModelTanh:getLegSec()
  return self.legSec
end

function BiLangModelTanh:getCriterion()
  return self.criterion
end

function BiLangModelTanh:getLookupTablePri()
  return self.ltPri
end

function BiLangModelTanh:getLookupTableSec()
  return self.ltSec
end
