-- multiple learning rates per network. Optimizes two copies of a model network and checks if the optimization steps (2) and (3) produce the same weights/parameters.
Example code for how to set different learning rates per layer. Note that when calling :parameters(), the weights and bias of a given layer are separate, consecutive tensors. Therefore, when calling :parameters(), a network with N layers will output a table with N*2 tensors, where the i'th and i'th+1 tensors belong to the same layer.


require 'torch'
require 'nn'
require 'optim'
torch.setdefaulttensortype('torch.FloatTensor')

-- (1) Define a model for this example.
local model = nn.Sequential()
model:add(nn.Linear(10,20))
model:add(nn.Linear(20,30))
model:add(nn.Linear(30,20))
model:add(nn.Linear(20,10))
local criterion = nn.CrossEntropyCriterion()

-- defines a single optimazation state for all of the model's layers
local optimState = {
        learningRate = 0.01,
        learningRateDecay = 0.0,
        momentum = 0.9,
        dampening = 0.0,
        weightDecay = 5e-4
  }
  
-- make two copies of the model: one copy will update its parameters by updating all layers at the same time, and another copy will update its parameters layer by layer. 
local model1 = model:clone()
local model2 = model:clone()

-- set some inputs/labels
local inputs = torch.FloatTensor(20,10):uniform()
local labels = torch.FloatTensor(20):random(1,3)


-- (2) optimize model1 (all layers at once)
local parameters1, gradParameters1 = model1:getParameters() 

-- forward + backward
local err, outputs
local feval1 = function(x)
  model1:zeroGradParameters()
  outputs = model1:forward(inputs)
  err = criterion:forward(outputs, labels)
  local gradOutputs = criterion:backward(outputs, labels)
  model1:backward(inputs, gradOutputs)
  return err, gradParameters1
end
-- optimize all layers parameters at once
optim.sgd(feval1, parameters1, optimState)


-- (3) optimize model1 layer by layer
local parameters2, gradParameters2 = model2:parameters() 

-- set optim states for each layer. To set different rates just change the learningRate field value.
optimState2 = {}
for i=1, #parameters2 do
  table.insert(optimState2, {
                              learningRate = 0.01,
                              learningRateDecay = 0.0,
                              momentum = 0.9,
                              dampening = 0.0,
                              weightDecay = 5e-4
                            }
              )
end

-- forward + backward
model2:zeroGradParameters()
outputs = model2:forward(inputs)
err = criterion:forward(outputs, labels)
gradOutputs = criterion:backward(outputs, labels)
model2:backward(inputs, gradOutputs)

-- optimize layer by layer
for i=1, #parameters2 do
  local feval2 = function(x)
    return err, gradParameters2[i]
  end
  
  optim.sgd(feval2, parameters2[i], optimState2[i])
end

-- (4) check if models parameters (tensors) are the same
local parameters2_, gradParameters2_ = model2:getParameters() 
if torch.add(parameters1, -parameters2_):sum() == 0 then
  print('Parameters are equal')
else
  print('Parameters are different')
end
