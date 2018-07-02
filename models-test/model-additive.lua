local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'


local BiLangModelAdditive = torch.class("BiLangModelAdditive")

function BiLangModelAdditive:__init(vocab_sizePri,vocab_sizeSec,config)
  self.legPri = nn.Sequential()
  self.ltPri = nn.LookupTableMaskZero(vocab_sizePri,config.emb_size)
  self.legPri:add( nn.SplitTable(2))
  self.legPri:add( nn.Sequencer(self.ltPri))
  self.legPri:add( nn.CAddTable())
  --self.additivePri:add( nn.MulConstant(1/corpus_1.longest_seq))
  
  self.legPri:getParameters():uniform(-0.01,0.01)
  self.legPri:cuda()

  self.legSec = nn.Sequential()
  self.ltSec = nn.LookupTableMaskZero(vocab_sizeSec,config.emb_size)
  self.legSec:add( nn.SplitTable(2))
  self.legSec:add( nn.Sequencer(self.ltSec))
  self.legSec:add( nn.CAddTable())
  --self.additiveSec:add( nn.MulConstant(1/corpus_2.longest_seq))
  
  self.legSec:getParameters():uniform(-0.01,0.01)
  self.legSec:cuda()

  self.criterion = nn.AbsCriterion():cuda()
  --self.criterion = nn.MSECriterion():cuda()
  --self.criterion = nn.MaskZeroCriterion(self.criterion, 1):cuda()
end

function BiLangModelAdditive:getLegPri()
  return self.legPri
end

function BiLangModelAdditive:getLegSec()
  return self.legSec
end

function BiLangModelAdditive:getCriterion()
  return self.criterion
end

function BiLangModelAdditive:getLookupTablePri()
  return self.ltPri
end

function BiLangModelAdditive:getLookupTableSec()
  return self.ltSec
end
