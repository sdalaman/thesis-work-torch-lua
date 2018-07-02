require 'torch'
require 'nn'
require 'cunn'
require 'rnn'
require 'testmodel'


function apply_to_slices(tensor, dimension, func, ...)
    for i, slice in ipairs(tensor:split(1, dimension)) do
        func(slice, i, ...)
    end
    return tensor
end

function power_fill(tensor, i, power)
    power = power or 1
    tensor:fill(i ^ power)
end

function prepMrx(A,step)
  A1 = A:narrow(2,1,A:size()[2]-1)
  A2 = A:narrow(2,2,A:size()[2]-1)
  Tot = torch.Tensor(A:size()[1],A:size()[2]-1)
  for i = 1 , A:size()[2]-1 do
    torch.add(Tot,A1, A2)
  end
  return Tot
end


A = torch.Tensor(100, 64)

-- B = apply_to_slices(A:clone(), 1, power_fill)
 
--C = apply_to_slices(A:clone(), 2, power_fill, 3)

for i = 1 , A:size()[2] do
  A:select(2,i):fill(i)
end
  
AT = prepMrx(A,2)
m1 = TestModel(AT:size()[1],AT:size()[2])
resT = m1.mm:forward(AT)

i = 0