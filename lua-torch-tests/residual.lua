require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

torch.seed()

emb_size = 20
hidden_size = 20
vsizePri = 1000
seq_lenPri = 5
batch_size = 10

inp = {}
inp2 = {}
for i = 1,seq_lenPri do
  x = torch.Tensor(batch_size):random(1,vsizePri)
  table.insert(inp,x)
  table.insert(inp2,{x})
end

local net = net or nn.Sequential()

net = nn.Sequential()
ltPri = nn.LookupTableMaskZero(vsizePri,emb_size) -- MaskZero
net:add(nn.Sequencer(ltPri))

modPri = nn.ConcatTable()
for i = 1, seq_lenPri, 1 do -- seq length
  recPri = nn.NarrowTable(i,1)
  modPri:add(recPri)
end
net:add(modPri)
add_lstmPri = nn:Sequential()
add_lstmPri:add(nn.Sequencer(nn.CAddTable()))
lstmP = nn.LSTM(hidden_size,hidden_size):maskZero(1)
lstmPri = nn.Sequential():add(lstmP):add(nn.NormStabilizer()):add(nn.Dropout(dropoutProb)) 
add_lstmPri:add(nn.Sequencer(lstmPri))
net:add(add_lstmPri)


rr = net:forward(inp)


print("end")
