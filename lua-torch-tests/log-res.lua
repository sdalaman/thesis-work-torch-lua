require 'torch'
    require 'math'
    require 'svm'
    require 'nn'
    require 'optim'
    require 'cudnn'
    
    TRAIN_NAME = "sample_train.txt"
    TEST_NAME = "sample_test.txt"
    
    
    -- i have:
    num_samples = 10000
    num_features = 500
    - dataset_inputs: feature tensor (dimension: num_samples X num_features)
    - dataset_outputs: labels tensor (dim: num_samples)
    num_labels = 2
    
    -- create the model
    linLayer = nn.Linear(num_features, num_labels)
    softMaxLayer = nn.LogSoftMax()  -- the input and output are a single tensor
    model = nn.Sequential()
    model:add(linLayer)
    model:add(softMaxLayer)
    cudnn.convert(model, cudnn)  -- converts the model
    print(model)
    
    -- loss function to be minimized: negative log-likelihood
    criterion = nn.ClassNLLCriterion()
    
    ----------------------------------------------------------------------
    -- Train the model (Using SGD)
    
    x, dl_dx = model:getParameters()
    
    feval = function(x_new)
       if x ~= x_new then
          x:copy(x_new)
       end
    
       _nidx_ = (_nidx_ or 0) + 1
       if _nidx_ > (#dataset_inputs)[1] then _nidx_ = 1 end
    
       local inputs = dataset_inputs[_nidx_]
       local target = dataset_outputs[_nidx_]
    
       dl_dx:zero()
    
       -- evaluate the loss function and its derivative wrt x, for that sample
       local loss_x = criterion:forward(model:forward(inputs), target)
       model:backward(inputs, criterion:backward(model.output, target))
    
       -- return loss(x) and dloss/dx
       return loss_x, dl_dx
    end
    
    -- Parameters train the model using SGD
    sgd_params = {
       learningRate = 1e-3,
       learningRateDecay = 1e-4,
       weightDecay = 0,
       momentum = 0
    }
    
    
    epochs = 1e2  -- number of cycles/iterations over our training data
    
    print('')
    print('============================================================')
    print('Training with SGD')
    print('')
    
    for i = 1,epochs do
    
       -- this variable is used to estimate the average loss
       current_loss = 0
    
       -- an epoch is a full loop over our training data
       for i = 1,(#dataset_inputs)[1] do
    
          _,fs = optim.sgd(feval,x,sgd_params) -- PROBLEM!! : this function call produces the error
    
          current_loss = current_loss + fs[1]
       end
    
       -- report average error on epoch
       current_loss = current_loss / (#dataset_inputs)[1]
       print('epoch = ' .. i .. ' of ' .. epochs .. ' current loss = ' .. current_loss)
    
    end
    
    
    -- Then I will use the "model" to predict on test samples
    
    
    print("---- DONE -----")
