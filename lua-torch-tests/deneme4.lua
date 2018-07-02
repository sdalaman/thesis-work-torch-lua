require 'nn'
require 'cunn'
require 'rnn'

rho = 5
hiddenSize = 10
outputSize = 5 -- num classes
nIndex = 1000
batch_size = 1

inp = {}
inp2 = {}
for i = 1,1 do
  x = torch.Tensor(batch_size):random(1,nIndex)
  table.insert(inp,x)
  table.insert(inp2,{x})
end

x = torch.Tensor(1):random(1,nIndex)
y = torch.Tensor(hiddenSize):random(1,nIndex)
ii = {torch.ones(5), torch.ones(5)*2, torch.ones(5)*3}

lp = nn.LookupTable(nIndex, hiddenSize)
ln = nn.Linear(hiddenSize, hiddenSize)
ca = nn.CAddTable()

r1 = lp:forward(x)
r2 = ln:forward(y)
tb = {r1,r2}
r3 = ca:forward(tb)

-- recurrent module
rm = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.LookupTable(nIndex, hiddenSize))
      :add(nn.Linear(hiddenSize, hiddenSize)))
   :add(nn.CAddTable())
   :add(nn.Sigmoid())

rnn = nn.Sequencer(
   nn.Sequential()
      :add(nn.Recurrence(rm, hiddenSize, 1))
      :add(nn.Linear(hiddenSize, outputSize))
      :add(nn.LogSoftMax())
)




print("end")