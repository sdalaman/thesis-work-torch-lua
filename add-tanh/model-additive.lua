local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'nn'
require 'cunn'
require 'rnn'


local BiLangModelAdditive = torch.class("BiLangModelAdditive")

function BiLangModelAdditive:__init(vocab_sizePri,vocab_sizeSec,config)
  if vocab_sizePri ~= 0 then
    self.legPri = nn.Sequential()
    self.ltPri = nn.LookupTableMaskZero(vocab_sizePri,config.emb_size)
    self.legPri:add( nn.SplitTable(2))
    self.legPri:add( nn.Sequencer(self.ltPri))
    self.legPri:add( nn.CAddTable())
  --self.additivePri:add( nn.MulConstant(1/corpus_1.longest_seq))
  
    self.legPri:getParameters():uniform(-0.01,0.01)
    self.legPri:cuda()
  else
    self.legPri = {}
    self.ltPri = {}
  end

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


local BiLangModelAdditiveMulti = torch.class("BiLangModelAdditiveMulti")

function BiLangModelAdditiveMulti:__init(vocab_sizePri,vocab_sizeSec,vocab_sizeThrd,vocab_sizeFrth,config)
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

  self.legThrd = nn.Sequential()
  self.ltThrd = nn.LookupTableMaskZero(vocab_sizeThrd,config.emb_size)
  self.legThrd:add( nn.SplitTable(2))
  self.legThrd:add( nn.Sequencer(self.ltThrd))
  self.legThrd:add( nn.CAddTable())
  --self.additiveSec:add( nn.MulConstant(1/corpus_2.longest_seq))
  
  self.legThrd:getParameters():uniform(-0.01,0.01)
  self.legThrd:cuda()
  
  self.legFrth = nn.Sequential()
  self.ltFrth = nn.LookupTableMaskZero(vocab_sizeFrth,config.emb_size)
  self.legFrth:add( nn.SplitTable(2))
  self.legFrth:add( nn.Sequencer(self.ltFrth))
  self.legFrth:add( nn.CAddTable())
  --self.additiveSec:add( nn.MulConstant(1/corpus_2.longest_seq))
  
  self.legFrth:getParameters():uniform(-0.01,0.01)
  self.legFrth:cuda()

  self.criterionPri = nn.AbsCriterion():cuda()
  self.criterionSec = nn.AbsCriterion():cuda()
  self.criterionThrd = nn.AbsCriterion():cuda()
  self.criterionFrth = nn.AbsCriterion():cuda()
  --self.criterion = nn.MSECriterion():cuda()
  --self.criterion = nn.MaskZeroCriterion(self.criterion, 1):cuda()
end

function BiLangModelAdditiveMulti:getLegPri()
  return self.legPri
end

function BiLangModelAdditiveMulti:getLegSec()
  return self.legSec
end

function BiLangModelAdditiveMulti:getLegThrd()
  return self.legThrd
end

function BiLangModelAdditiveMulti:getLegFrth()
  return self.legFrth
end

function BiLangModelAdditiveMulti:getLookupTablePri()
  return self.ltPri
end

function BiLangModelAdditiveMulti:getLookupTableSec()
  return self.ltSec
end

function BiLangModelAdditiveMulti:getLookupTableThrd()
  return self.ltThrd
end

function BiLangModelAdditiveMulti:getLookupTableFrth()
  return self.ltFrth
end

function BiLangModelAdditiveMulti:getCriterionPri()
  return self.criterionPri
end

function BiLangModelAdditiveMulti:getCriterionSec()
  return self.criterionSec
end

function BiLangModelAdditiveMulti:getCriterionThrd()
  return self.criterionThrd
end

function BiLangModelAdditiveMulti:getCriterionFrth()
  return self.criterionFrth
end
