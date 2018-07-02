-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("neuralconvo.Seq2Seq")

function Seq2Seq:__init(vocabSize, hiddenSize)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")

  self:buildModel()
end

function Seq2Seq:buildModel()
  self.encoder = nn.Sequential()
  self.encoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  self.encoder:add(nn.SplitTable(1, 2))
  self.encoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize)
  self.encoder:add(nn.Sequencer(self.encoderLSTM))
  self.encoder:add(nn.SelectTable(-1))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  self.decoder:add(nn.SplitTable(1, 2))
  self.decoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize)
  self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.decoder:add(nn.Sequencer(nn.Linear(self.hiddenSize, self.vocabSize)))
  self.decoder:add(nn.Sequencer(nn.LogSoftMax()))

  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  self.zeroTensor = torch.Tensor(2):zero()
end

function Seq2Seq:cuda()
  self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end

  self.zeroTensor = self.zeroTensor:cuda()
end

function Seq2Seq:double()
  self.encoder:double()
  self.decoder:double()
  
  if self.criterion then
    self.criterion:double()
  end
  self.zeroTensor = self.zeroTensor:double()
end


--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function Seq2Seq:forwardConnect(inputSeqLen)
  self.decoderLSTM.userPrevOutput =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevOutput, self.encoderLSTM.outputs[inputSeqLen])
  self.decoderLSTM.userPrevCell =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevCell, self.encoderLSTM.cells[inputSeqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect()
  self.encoderLSTM.userNextGradCell =
    nn.rnn.recursiveCopy(self.encoderLSTM.userNextGradCell, self.decoderLSTM.userGradPrevCell)
  self.encoderLSTM.gradPrevOutput =
    nn.rnn.recursiveCopy(self.encoderLSTM.gradPrevOutput, self.decoderLSTM.userGradPrevOutput)
end

function Seq2Seq:validate(input, target)
  local encoderInput = input
  local decoderInput = target:sub(1, -2)
  local decoderTarget = target:sub(2, -1)

  -- Forward pass
  self.encoder:forward(encoderInput)
  self:forwardConnect(encoderInput:size(1))
  local decoderOutput = self.decoder:forward(decoderInput)
  local Edecoder = self.criterion:forward(decoderOutput, decoderTarget)

  return Edecoder
end

function Seq2Seq:train(input, target)
  local encoderInput = input
  local decoderInput = target:sub(1, -2)
  local decoderTarget = target:sub(2, -1)

  -- Forward pass
  self.encoder:forward(encoderInput)
  self:forwardConnect(encoderInput:size(1))
  local decoderOutput = self.decoder:forward(decoderInput)
  local Edecoder = self.criterion:forward(decoderOutput, decoderTarget)

  if Edecoder ~= Edecoder then -- Exist early on bad error
    return Edecoder
  end

  -- Backward pass
  local gEdec = self.criterion:backward(decoderOutput, decoderTarget)
  self.decoder:backward(decoderInput, gEdec)
  self:backwardConnect()
  self.encoder:backward(encoderInput, self.zeroTensor)

  self.encoder:updateGradParameters(self.momentum)
  self.decoder:updateGradParameters(self.momentum)
  self.decoder:updateParameters(self.learningRate)
  self.encoder:updateParameters(self.learningRate)
  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  self.decoder:forget()
  self.encoder:forget()

  return Edecoder
end

local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input, options)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  local sampling = options.sampling or 'argmax'
  local temperature = options.temperature or 1

  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}
  local outputs = {}

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = self.goToken
  for i = 1, MAX_OUTPUT_SIZE do
    local prediction = self.decoder:forward(torch.Tensor{output})[1]
    -- prediction contains the probabilities for each word IDs.
    -- The index of the probability is the word ID.
    local probs, wordIds = prediction:sort(1, true)
    
    -- log probabilities from the previous timestep
    if sampling == 'argmax' then
        -- use argmax
        output = wordIds[1]    -- First one is the most likely.
    else
        -- use sampling
        prediction:div(temperature) -- scale by temperature
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        output = torch.multinomial(probs:float(), 1)[1]
    end

    -- Terminate on EOS token
    if output == self.eosToken then
      break
    end

    table.insert(outputs, output)
    table.insert(predictions, wordIds)
    table.insert(probabilities, probs)
  end 

  self.decoder:forget()
  self.encoder:forget()

  return outputs, predictions, probabilities
end