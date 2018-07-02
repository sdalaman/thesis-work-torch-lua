require "torch"
require "nn"
require "rnn"

-- Initialize random number generator
math.randomseed( os.time() )

-- hyper-parameters 
batch_size = 8
test_size = 4
rho = 5 -- sequence length
hidden_size = 100
output_size = 3
learning_rate = 0.001

-- Initialize synthetic dataset
-- dataset[index] returns table of the form: {inputs, targets}
-- where inputs is a set of points (x,y) of a randomly selected function: line, parabola, sqrt
-- and targets is a set of corresponding class of a function (1=line, 2=parabola, 3=sqrt)
local dataset = {}
dataset.k = math.random()
dataset.b = math.random()*5
dataset.size = function (self)
  return 1000
end
local mt = {}
mt.__index = function (self, i)
  local class = math.random(3)

  local t = torch.Tensor(3):zero()
  t[class] = 1
  local targets = {}
  for i = 1,batch_size do table.insert(targets, class) end

  local inputs = {}
  local k = self.k
  local b = self.b

  -- Line
  if class == 1 then
    for i = 1,batch_size do
      local x = math.random()*10 + 5
      local y = k*x + b
      input = torch.Tensor(2)
      input[1] = x
      input[2] = y
      table.insert(inputs, input)
    end

  -- Parabola
  elseif class == 2 then
    for i = 1,batch_size do
      local x = math.random()*10 + 5
      local y = k*x*x + b
      input = torch.Tensor(2)
      input[1] = x
      input[2] = y
      table.insert(inputs, input)
    end

  -- Sqrt
  else
    for i = 1,batch_size do
      local x = math.random()*5 + 5
      local y = k*math.sqrt(x) + b
      input = torch.Tensor(2)
      input[1] = x
      input[2] = y
      table.insert(inputs, input)
    end
  end

  return { inputs, targets }
end -- dataset.__index meta function
setmetatable(dataset, mt)


-- build simple recurrent neural network
local model = nn.Sequencer(
  nn.Sequential()
    :add( nn.LSTM(2, hidden_size, rho) )
    :add( nn.Linear(hidden_size, output_size) )
    :add( nn.LogSoftMax() )
)

print(model)

-- build criterion
local criterion = nn.SequencerCriterion( nn.ClassNLLCriterion() )


local epoch = 1
local err = 0
local pos = 0
local N = math.floor( dataset:size() * 0.1 )

while true do

  print ("Epoch "..tostring(epoch).." started")

  -- training
  model:training()
  for iteration = 1, dataset:size() do
    -- 1. Load minibatch of samples
    local sample = dataset[iteration] -- pick random sample (dataset always returns random set)
    local inputs = sample[1]
    local targets = sample[2]

    -- 2. Perform forward run and calculate error
    local outputs = model:forward(inputs)
    local _err = criterion:forward(outputs, targets)

    print(string.format("Epoch %d (pos=%f) Iteration %d Error = %f", epoch, pos, iteration, _err))

    -- 3. Backward sequence through model(i.e. backprop through time)
    local gradOutputs = criterion:backward(outputs, targets)
    -- Sequencer handles the backwardThroughTime internally
    model:backward(inputs, gradOutputs)
    model:updateParameters(learning_rate)
    model:zeroGradParameters()     

  end -- for training

  -- Testing
  model:evaluate()
  err = 0
  pos = 0
  for iteration = 1, N do
    -- 1. Load minibatch of samples
    local sample = dataset[ math.random(dataset:size()) ]
    local inputs = sample[1]
    local targets = sample[2]
    -- Drop last points to reduce to test_size
    for i = #inputs, test_size, -1 do
      inputs[i] = nil
      targets[i] = nil
    end

    -- 2. Perform forward run and calculate error
    local outputs = model:forward(inputs)
    err = err + criterion:forward(outputs, targets)

    local p = 0
    for i = 1, #outputs do
      local _, oi = torch.max(outputs[i], 1)
      if oi[1] == targets[i] then p = p + 1 end
    end
    pos = pos + p/#outputs

  end -- for testing
  err = err / N
  pos = pos / N
  print(string.format("Epoch %d testing results: pos=%f err=%f", epoch, pos, err))

  if (pos > 0.95) then break end

  epoch = epoch + 1
end -- while epoch