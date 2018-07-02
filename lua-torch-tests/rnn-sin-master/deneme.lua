require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

torch.seed()
x = torch.Tensor(10,5):random(1,3059):cuda()


ltPri = nn.LookupTableMaskZero(3059,64) -- MaskZero
ox = nn.Sequencer( ltPri):forward(x)

tt = nn.SplitTable(2,3):forward(ox)

rnn = nn.Sequential()
         :add(nn.FastLSTM(64, 64))
         :add(nn.NormStabilizer())

rnn = nn.Sequential()
          :add(nn.SplitTable(2,3))
          :add(nn.Sequencer(rnn))
          :add(nn.SelectTable(-1))
          :add(nn.Dropout(0.5))
          :add(nn.Linear(64,64))
          --:add(nn.ReLU())
          --:add(nn.Linear(64, 64))
          :add(nn.HardTanh())


o = rnn:forward(ox)

print("end")
