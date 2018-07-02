--[[
--   Element-Research Torch RNN Tutorial for recurrent neural nets : let's predict time series with a laptop GPU
--   https://christopher5106.github.io/deep/learning/2016/07/14/element-research-torch-rnn-tutorial.html
--]]

--[[
--   Part 1
--]]

require 'rnn'

-- Construct RNN for h_t = σ(W_hh . h_t−1 + W_xh . X_t) 
local r = nn.Recurrent(
  7,  -- Hidden state size
  nn.LookupTable(10, 7), -- W_xh . X_t: 7D embedding of a word from a dictionary of 10 words
  nn.Linear(7, 7), -- W_hh . h_t-1
  nn.Sigmoid(), -- Transfer function σ
  5 -- Truncated BPTT limit
)

-- Construct RNN (alternative with more general nn.Recurrence)
local rm = nn.Sequential()
  :add(nn.ParallelTable()
    :add(nn.LookupTable(10, 7))
    :add(nn.Linear(7, 7)))
  :add(nn.CAddTable())
  :add(nn.Sigmoid())

r = nn.Recurrence(rm, 7, 1) -- Arguments are recurrent module, hidden state size and input dimension

-- Construct output for o_t = W_ho . h_t
local rr = nn.Sequential()
  :add(r)
  :add(nn.Linear(7, 10))
  :add(nn.LogSoftMax()) -- Output log probabilities

-- Wrap non-recurrent modules with nn.Recursor
local rnn = nn.Recursor(rr, 5) -- Truncated BPTT limit

-- Create input and target sequences
local inputs = torch.LongTensor({{1}, {2}, {3}, {4}, {5}})
local targets = torch.LongTensor({{2}, {3}, {4}, {5}, {6}})

-- Apply each element of the sequence to the RNN step by step
local outputs, err = {}, 0
local criterion = nn.ClassNLLCriterion()
for step = 1, 5 do
  outputs[step] = rnn:forward(inputs[step])
  err = err + criterion:forward(outputs[step], targets[step])
end

-- Train the RNN with BPTT step by step
local gradOutputs, gradInputs = {}, {}
for step = 5, 1, -1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
end

-- Update the parameters
rnn:updateParameters(0.1) -- Learning rate
rnn:forget() -- Reset the hidden state after every training or evaluation sequence
rnn:zeroGradParameters() -- Reset the accumulated gradients after every training sequence

-- Alternatively, use nn.Sequencer to process a sequence in one step
rnn = nn.Sequencer(rr)
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- Perform forward and backward pass
outputs = rnn:forward(inputs)
err = criterion:forward(outputs, targets)
gradOutputs = criterion:backward(outputs, targets)
gradInputs = rnn:backward(inputs, gradOutputs)

-- Update the parameters
rnn:updateParameters(0.1)
-- nn.Sequencer calls forget before every forward call
rnn:zeroGradParameters()

--[[
--   Part 2
--]]

require 'gnuplot'

-- Create cos function to predict
local ii = torch.linspace(0, 200, 2000)
local oo = torch.cos(ii)
gnuplot.plot({'f(x)', ii, oo, '+-'})

-- Use GPU 1
local gpu = 1
require 'cutorch'
require 'cunn'
cutorch.setDevice(gpu)
local sequence = oo:cuda()

-- Set up hyperparameters
local nIters = 2000
local batchSize = 80
local rho = 10
local hiddenSize = 300
local nIndex = 1
local lr = 0.0001
local nPredict = 200

-- Set up network
rnn = nn.Sequential()
  :add(nn.Linear(nIndex, hiddenSize))
  :add(nn.FastLSTM(hiddenSize, hiddenSize))
  :add(nn.NormStabilizer()) -- Use the norm stabilisation criterion to regularise the hidden states
  :add(nn.Linear(hiddenSize, nIndex))
  :add(nn.HardTanh())
rnn = nn.Sequencer(rnn):cuda()
rnn:training()
print(rnn)

-- Set up criterion
criterion = nn.MSECriterion():cuda()

-- Create random offsets to create sequences of length rho to train with
local offsets = {}
for i = 1, batchSize do
  table.insert(offsets, math.ceil(math.random() * (sequence:size(1) - rho)))
end
offsets = torch.LongTensor(offsets):cuda()

-- Create zero targets (as only final prediction matters)
local gradOutputsZeroed = {}
for step = 1, rho do
  gradOutputsZeroed[step] = torch.zeros(batchSize, 1):cuda()
end

-- Train
local iteration = 1
while iteration < nIters do
  -- Create inputs and targets
  local inputs, targets = {}, {}
  for step = 1, rho do
    inputs[step] = sequence:index(1, offsets):view(batchSize, 1) -- Create input sequence using offsets
    offsets:add(1) -- Increment offsets (for t+1 target prediction)
    for j = 1, batchSize do
      -- Wrap offsets around if necessary
      if offsets[j] > sequence:size(1) then
        offsets[j] = 1
      end
    end
    targets[step] = sequence:index(1, offsets) -- Create target sequence using incremented offsets
  end

  rnn:zeroGradParameters() -- Zero gradients

  -- Forward propagate
  local outputs = rnn:forward(inputs)
  local err = criterion:forward(outputs[rho], targets[rho])
  print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
  -- Backward propagate
  local gradOutputs = criterion:backward(outputs[rho], targets[rho])
  gradOutputsZeroed[rho] = gradOutputs
  local gradInputs = rnn:backward(inputs, gradOutputsZeroed)

  -- Update parameters
  rnn:updateParameters(lr)
  iteration = iteration + 1
end

-- Test
rnn:evaluate()
-- Create sequence to predict
local predict = torch.CudaTensor(nPredict)
for step = 1, rho do
  predict[step] = sequence[step]
end

local start = {}
iteration = 0
while rho + iteration < nPredict do
  for step = 1, rho do
    start[step] = predict:index(1, torch.LongTensor({step + iteration})):view(1, 1)
  end

  output = rnn:forward(start)

  predict[iteration + rho + 1] = (output[rho]:float())[1][1] -- Retrieve prediction

  iteration = iteration + 1
end

-- Plot predictions
gnuplot.plot({'f(x)', predict, '+-'})