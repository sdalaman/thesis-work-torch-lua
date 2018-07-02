require 'nn'
require 'cunn'
require 'nngraph'
--require 'rnn'


n=4
experts = nn.ConcatTable()
for i = 1, n do
   local expert = nn.Sequential()
   expert:add(nn.Linear(3, 4))
   expert:add(nn.Tanh())
   expert:add(nn.Linear(4, 5))
   expert:add(nn.Tanh())
   experts:add(expert)
end

gater = nn.Sequential()
gater:add(nn.Linear(3, 7))
gater:add(nn.Tanh())
gater:add(nn.Linear(7, n))
gater:add(nn.SoftMax())

trunk = nn.ConcatTable()
trunk:add(gater)
trunk:add(experts)

moe = nn.Sequential()
moe:add(trunk)
--moe:add(nn.MixtureTable())

mx = nn.MixtureTable()
local mxp, mxpdx = mx:getParameters()

inp = torch.randn(2, 3)
out1 = trunk:forward(inp)
out2 = mx:forward(out1)

mxp, mxpdx = mx:getParameters()


----------------------------

x1 = nn.Identity()()
x2 = nn.Identity()()
a = nn.CAddTable()({x1, x2})
m = nn.gModule({x1, x2}, {a})
x = {x1,x2,x2,x1}

a1 = nn.Sequencer(nn.Unsqueeze(1))(x1)
a2 = nn.Sequencer(nn.SplitTable(2))(a1)
a3 = nn.ZipTable()()a2
a4 = nn.Sequencer(nn.JoinTable(1,2))(a4)
      



moe = nn.Sequential()
moe:add(m)

x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)

out = moe:forward({x1,x2})

-----------------------

b=10
r=10
s=5
h=5
model = nn.MM()
model2 = nn.Linear(10, 5)
local mp, mpdx = model2:getParameters()
--A = torch.randn(b,r , s )
--B = torch.randn(b, s, h)
--C = model:forward({A, B}) 
y = torch.randn(b, 5)
x = torch.randn(b, 10)
mp, mpdx = model2:getParameters()

pred = model2:forward(x) 
criterion = nn.AbsCriterion()
err = criterion:forward( pred, y)
print("err : "..err)
gradOutputs = criterion:backward(pred, y)
model2:backward(x, gradOutputs)
model2:updateParameters(1)

mp, mpdx = model2:getParameters()

print("end")