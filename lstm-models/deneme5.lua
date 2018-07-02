require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

n = 60
d = 64
h = 128
da = 10

H = torch.exp(torch.abs(torch.randn(2*h,n)))
H2 = torch.exp(torch.abs(torch.randn(2*h,n)))
Ht = H:t()
l1 = nn.Linear(2*h,da)
l2 = nn.Linear(da,1)
tr = nn.Transpose({1,2})
mm1 = nn.MM()
mm2 = nn.CMulTable()

t1 = torch.Tensor({{1,2,4},{1,2,4}})
t2 = torch.Tensor({{1,2,2},{1,2,2}})

pd = nn.PairwiseDistance(1)

od = pd:forward({t2,t1})

o1 = l1:forward(Ht)
o2 = l2:forward(nn.Tanh():forward(o1))
o3 = H* o2
o4 = o2:t()*H:t()

om1 = mm1:forward({H,o2})


oo =  nn.Sequential()
      :add(nn.Transpose({1,2}))
      :add(nn.Linear(2*h,da))
      :add(nn.Tanh())
      :add(nn.Linear(da,1))
      
  
oo2 = nn.ConcatTable()
  :add(nn.Identity())
  :add(oo)

  
  
o5 = oo2:forward(H) 
      
Ht2 = tr:forward(H)

print("end")


