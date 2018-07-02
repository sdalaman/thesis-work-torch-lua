require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'
require 'cunn'

input = torch.rand(10)
lstm = nn.LSTM(10,10)
out = lstm:forward(input)

function CreateModule(input_size)
    local input = nn.Identity()()   -- network input

    local nn_module_1 = nn.Linear(input_size, 10)(input)
    local nn_module_2 = nn.Linear(10, input_size)(nn_module_1)

    local output = nn.CMulTable()({input, nn_module_2})

    -- pack a graph into a convenient module with standard API (:forward(), :backward())
    return nn.gModule({input}, {output})
end

criterion = nn.MSECriterion():cuda()

rho = 5
hiddenSize = 10
input = torch.rand(10):cuda()
y = torch.rand(10):cuda()

my_module = CreateModule(input:size(1)):cuda()
my_module:getParameters():uniform(-1*0.001,0.001)
maxepoch=10000

prms, gradPrms= my_module:parameters()

for i=1,maxepoch do
  output = my_module:forward(input)
  err = criterion:forward( output, y)
  print("err : "..err)
  gradOutputs = criterion:backward(output, y)
  my_module:backward(input, gradOutputs)
  my_module:updateParameters(0.01)
end

pred = my_module:forward(input)
print(pred)
print(y)
print(pred-y)
print("end")





