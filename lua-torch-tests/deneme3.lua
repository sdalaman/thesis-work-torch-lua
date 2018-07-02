require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

torch.seed()

emb_size = 64
hidden_size = 64
vsizePri = 1000
seq_lenPri = 5
batch_size = 5

inp = {}
for i = 1,seq_lenPri do
  x = torch.Tensor(batch_size):random(1,vsizePri):cuda()
  table.insert(inp,x)
end

mdlPri = nn.Sequential()
ltPri = nn.LookupTableMaskZero(vsizePri,emb_size) -- MaskZero
mdlPri:add(nn.Sequencer(ltPri))

modPri = nn.ConcatTable()
for i = 1, seq_lenPri, 1 do -- seq length
  recPri = nn.NarrowTable(i,1)
  modPri:add(recPri)
end

mdlPri:add(modPri)
add_lstmPri = nn:Sequential()
add_lstmPri:add(nn.Sequencer(nn.CAddTable()))
add_lstmPri:add(nn.Sequencer(nn.Linear(emb_size, hidden_size)))
lstmP1 = nn.LSTM(hidden_size,hidden_size):maskZero(1)
lstmP2 = nn.LSTM(hidden_size,hidden_size):maskZero(1)
lstmPri = nn.Sequential():add(lstmP1):add(nn.NormStabilizer()):add(nn.Dropout(dropoutProb)) 
lstmPri:add(lstmP2):add(nn.NormStabilizer()):add(nn.Dropout(dropoutProb)) 
add_lstmPri:add(nn.Sequencer(lstmPri))
add_lstmPri:add(nn.Sequencer(nn.Linear(hidden_size, emb_size)))
--add_lstmPri:add(nn.SelectTable(-1)) -- add linear
mdlPri:add(add_lstmPri)
--nn.MaskZero(self.mdlPri,1)
--mdlPri:getParameters():uniform(-1*prm_list["init"],prm_list["init"])

mdlPri:cuda()
 
rr = mdlPri:forward(inp)
--a1,b1 = mdlPri:getParameters()
--a2,b2 = mdlPri:parameters()
--lstma,b = lstmP1:parameters()


print("end")
