require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

torch.seed()
x = torch.Tensor(1,5):random(1,3059):cuda()


additivePri = nn.Sequential()
ltPri = nn.LookupTableMaskZero(3059,64) -- MaskZero
additivePri:add(nn.Sequencer( ltPri))
additivePri:add( nn.SplitTable(2))
modPri = nn.ConcatTable()
lstmPri = nn.Sequencer(nn.LSTM(64,64))
for i = 1, 5, 1 do
    recPri = nn.Sequential()
    recPri:add(nn.NarrowTable(i,1))
    recPri:add(lstmPri:clone('weight','bias'))
    recPri:add(nn.CAddTable()) 
    recPri:add(nn.Linear(64, 64))
    recPri:add(nn.Tanh())
    modPri:add(recPri)
end
additivePri:add(modPri)
additivePri:add(nn.CAddTable())


rnn = nn.Sequential()
         :add(nn.FastLSTM(64, 64))
         :add(nn.NormStabilizer())

rnn = nn.Sequential()
          :add(nn.SplitTable(0,2))
          :add(nn.Sequencer(rnn))
          :add(nn.SelectTable(-1))
          :add(nn.Dropout(0.5))
          :add(nn.Linear(64,64))
          --:add(nn.ReLU())
          --:add(nn.Linear(64, 1))
          :add(nn.HardTanh())


o = additivePri:forward(x)

print("end")
