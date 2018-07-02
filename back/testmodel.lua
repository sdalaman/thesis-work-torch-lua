require 'torch'
require 'nn'
require 'cunn'
require 'rnn'

local TestModel = torch.class("TestModel")

function TestModel:__init(rownum,colnum)
  self.mm = nn.Sequential()
  self.inp = torch.Tensor(rownum,colnum)
  self.mm:add( nn.SplitTable(2))
  self.mm:add( nn.Sequencer(nn.Tanh(self.inp)))
  --self.mm:add( nn.Tanh(self.inp))
  self.mm:add( nn.CAddTable())
end
