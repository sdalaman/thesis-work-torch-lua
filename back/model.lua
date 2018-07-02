local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'


local BiLangModel = torch.class("BiLangModel")

function BiLangModel:__init(vocab_sizePri,vocab_sizeSec,config)
  self.additivePri = nn.Sequential()
  self.ltPri = nn.LookupTableMaskZero(vocab_sizePri,config.emb_size)
  self.additivePri:add( nn.SplitTable(2))
  self.additivePri:add( nn.Sequencer(self.ltPri))
  self.additivePri:add( nn.CAddTable())
  --self.additivePri:add( nn.MulConstant(1/corpus_1.longest_seq))
  
  self.additivePri:getParameters():uniform(-0.01,0.01)
  self.additivePri:cuda()

  self.additiveSec = nn.Sequential()
  self.ltSec = nn.LookupTableMaskZero(vocab_sizeSec,config.emb_size)
  self.additiveSec:add( nn.SplitTable(2))
  self.additiveSec:add( nn.Sequencer(self.ltSec))
  self.additiveSec:add( nn.CAddTable())
  --self.additiveSec:add( nn.MulConstant(1/corpus_2.longest_seq))
  
  self.additiveSec:getParameters():uniform(-0.01,0.01)
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
