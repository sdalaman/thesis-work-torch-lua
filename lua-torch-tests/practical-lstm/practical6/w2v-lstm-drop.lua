require 'data-lstm'
require 'model-lstm-drop'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'

data_path = "../data/"
model_save_path = "../../test-models/"
model_load_path = "../../test-models/"
corpus_path = "../../../models/corpus/"

test_size = 100
cv_max = 10

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

function implode(sep,tbl,names)
  local newstr
  newstr = ""
  for i=1,#names do
    newstr = newstr ..names[i].."="..tbl[names[i]] .. sep
  end
  return newstr
end

function test_model(model,epoch,inpPri,inpSec,test_size)
  mPri = model:modelPri()
  mSec = model:modelSec()

  local split = nn.SplitTable(2)
  inPri = torch.zeros(test_size,inpPri:size()[2]):cuda()
  inSec = torch.zeros(test_size,inpSec:size()[2]):cuda()
  cl = inpPri:size()[1]
  math.randomseed( os.time() )
  
  for i=1,test_size do
    ln = math.random(1,cl)
    inPri[i] = inpPri[ln]
    inSec[i] = inpSec[ln]
  end
  
  local inputPri = split:forward(inPri)
  local inputSec = split:forward(inSec)
  
  local outputPri = mPri:forward( inputPri)
  local outputSec = mSec:forward( inputSec)
  all_rootsPri = outputPri
  all_rootsSec = outputSec

  score = 0
  for idxPri = 1, all_rootsPri:size(1) do
    closest = idxPri
    for idxSec = 1, all_rootsSec:size(1) do
      if torch.dist(all_rootsPri[idxPri],all_rootsSec[idxSec]) < torch.dist(all_rootsPri[idxPri],all_rootsSec[closest]) then
        closest = idxSec
      end
    end
    
    if idxPri == closest then
      score = score + 1 
    else
      --print("Closest to: "..idxPri.." is: ".. closest)
    end
  end
  print("-------Epoch: "..epoch.." - Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  return score,all_rootsPri:size(1)
end

function cv_test(cv,model,maxepoch,insPri,insSec,test_size)
  tot = 0
  model.mdlPri:evaluate()
  model.mdlSec:evaluate()
  for i = 1,cv do
    sc,nsum = test_model(model,maxepoch,insPri,insSec,test_size)
    tot = tot + sc
  end
  print("-------"..cv.." Epochs avg: "..tot/cv.."\n")
  model.mdlPri:training()
  model.mdlSec:training()
  return sc,nsum,tot/cv
end

function model_train(model,inputsPri,inputsSec,prm_list,res_list)
  --criterion = nn.AbsCriterion():cuda()
  --nn.MaskZeroCriterion(criterion, 1)
  
  lr_list = prm_list["lr_list"]
  criterion = model:getCriterion()
  modelPri = model:modelPri()
  modelSec = model:modelSec()
  modelPri:training()
  modelSec:training()
  beginning_time = torch.tic()
  local split = nn.SplitTable(2)
  paramsPri, gradParamsPri = modelPri:getParameters()
  xPri2, dl_dxPri2= modelPri:parameters()
  paramsSec, gradParamsSec = modelSec:getParameters()
  xSec2, dl_dxSec2 = modelSec:parameters()
  
  local optimStatePriSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"]}
  local optimStateSecSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"]}
  
  local optimStatePriRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"]}
  local optimStateSecRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"]}
  
  if prm_list["opm_name"] == "sgd" then
    print("Optimization is SGD \n")
    statePri = optimStatePriSgd
    stateSec = optimStateSecSgd
  end
  
  local initstate_c = torch.zeros(prm_list["batch_size"], prm_list["hidden_size"])
  local initstate_h = initstate_c:clone()
  
  local dfinalstate_c = initstate_c:clone()
  local dfinalstate_h = initstate_c:clone()
  
  if prm_list["opm_name"] == "rmsprop" then
    print("Optimization is RmsProp \n")
    statePri = optimStatePriRmsProp
    stateSec = optimStateSecRmsProp
  end

  for i =prm_list["epoch"]+1,prm_list["max_epoch"] do
    current_lossPri = 0
    current_lossSec = 0
    totalGradWeightPri = 0
    totalGradWeightSec = 0
    local inds = torch.range(1, no_of_sents,prm_list["batch_size"])
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    local k = 0
    
    if lr_list ~= nil then
      if lr_list[i] ~= nil then
        statePri.learningRate = lr_list[i] 
        stateSec.learningRate = lr_list[i] 
      end
    end  

    for j = 1, math.floor(no_of_sents/prm_list["batch_size"]) do 
                --get input row and target
        local start = inds[shuffle[j]]
        local endd = inds[shuffle[j]]+prm_list["batch_size"]-1
        if((start > #corpus_pri.sequences) or (endd > #corpus_pri.sequences)) then
          k = k + 1
          endd = #corpus_pri.sequences
        end
        local inputPri = split:forward(inputsPri[{{start,endd},{}}])
        local inputSec = split:forward(inputsSec[{{start,endd},{}}])
        
        modelPri:zeroGradParameters()
        modelSec:zeroGradParameters()
        modelPri:forget()
        modelSec:forget()

        -- print( target)
        function fevalPri(params)
          local outputPri = modelPri:forward( inputPri)
          local outputSec = modelSec:forward( inputSec)
          local err = criterion:forward( outputPri, outputSec)
          local gradOutputs = criterion:backward(outputPri, outputSec)
          modelPri:backward(inputPri, gradOutputs)
          totalGradWeightPri = totalGradWeightPri + torch.sum(torch.pow(dl_dxPri2[1],2))
          return err, gradParamsPri
        end
        
        function fevalSec(params)
          local outputPri = modelPri:forward( inputPri)
          local outputSec = modelSec:forward( inputSec)
          local err = criterion:forward( outputSec, outputPri)
          local gradOutputs = criterion:backward(outputSec, outputPri)
          modelSec:backward(inputSec, gradOutputs)
          totalGradWeightSec = totalGradWeightSec + torch.sum(torch.pow(dl_dxSec2[1],2))
          return err, gradParamsSec
        end
        
        _,fsPri = prm_list["opm"](fevalPri, paramsPri, statePri)
        current_lossPri = current_lossPri + fsPri[1] 
        _,fsSec = prm_list["opm"](fevalSec, paramsSec, stateSec)
        current_lossSec = current_lossSec + fsSec[1] 
    end
    
    prm_list["epoch"] = i
    current_lossPri = current_lossPri / math.floor(no_of_sents/prm_list["batch_size"])
    current_lossSec = current_lossSec / math.floor(no_of_sents/prm_list["batch_size"])
    print("epoch "..i..", lr ".. statePri.learningRate.."/"..stateSec.learningRate..", loss (pri/sec) = "..current_lossPri.."/"..current_lossSec..", grad (pri/sec) = "..totalGradWeightPri.."/"..totalGradWeightSec)
    loss_logger:add{['lrPri'] = statePri.learningRate,['lrSec'] = stateSec.learningRate,['training error pri'] = current_lossPri ,['training error sec'] = current_lossSec}
    loss_logger:style{['lr']='+-',['training error pri'] = '+-' , ['training error sec'] = '+-' }
    grad_logger:add{['grad weight pri'] = totalGradWeightPri ,['grad weight sec'] = totalGradWeightSec }
    grad_logger:style{['grad weight pri'] = '+-' , ['grad weight sec'] = '+-' }
    
    if i % prm_list["dump_frq"] == 0 then
      sc,prm_list["num_of_samples"],prm_list["score"] = cv_test(cv_max,model,i,inputsPri,inputsSec,test_size)
      score_logger:add{["lrPri"] = statePri.learningRate,["lrSec"] = stateSec.learningRate, ['score '] = prm_list["score"] }
      score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['score '] = '+-'}
      score_logger:plot()  
      loss_logger:plot()  
      grad_logger:plot()  
      model_save(nm,sec_str,prm_list,model)
    end
  
    if i % prm_list["threshold"] == 0 then 
      statePri.learningRate = statePri.learningRate * prm_list["lr_decay"] 
      stateSec.learningRate = stateSec.learningRate * prm_list["lr_decay"]
      prm_list["lr"] = statePri.learningRate
    end
    
  end

  durationSeconds = torch.toc(beginning_time)
  print('time elapsed:'.. printtime(durationSeconds))
  return model
end

pri_corpus_name = "english.1000.tok"
sec_corpus_name = "turkish.1000.tok"
fhead = "lstm-drop-2layer-1000"

pri_str = string.sub(pri_corpus_name,1,2)
sec_str = string.sub(sec_corpus_name,1,2)

nm = string.sub(pri_corpus_name,1,2).."-"..string.sub(sec_corpus_name,1,2)
vocab1 = corpus_path..pri_corpus_name.."."..nm..".lstm.vocab" -- vocab file  for lang 1
vocab2 = corpus_path..sec_corpus_name.."."..nm..".lstm.vocab" -- vocab file  for lang 2
corpus_name_1 = corpus_path..pri_corpus_name.."."..nm..".lstm.corpus.tch"
corpus_name_2 = corpus_path..sec_corpus_name.."."..nm..".lstm.corpus.tch"
inputs_name_1 = corpus_path..pri_corpus_name.."."..nm..".lstm.inputs.tch"
inputs_name_2 = corpus_path..sec_corpus_name.."."..nm..".lstm.inputs.tch"

corpus_load = true
if corpus_load == false then
  corpus_pri = Corpus(pri_corpus_name)
  corpus_sec = Corpus(sec_corpus_name)
  corpus_pri:prepare_corpus(nil)
  corpus_sec:prepare_corpus(nil)
  inputsPri=corpus_pri:get_dataLSTM():cuda()
  inputsSec=corpus_sec:get_dataLSTM():cuda()

  torch.save(corpus_name_1,corpus_pri)
  torch.save(corpus_name_2,corpus_sec)
  torch.save(inputs_name_1,inputsPri)
  torch.save(inputs_name_2,inputsSec)
  torch.save(vocab1,corpus_pri.vocab_map)
  torch.save(vocab2,corpus_sec.vocab_map)
  print("corpus data prepared and saved\n")
else
  print("corpus data will be loaded\n")
  corpus_pri = torch.load(corpus_name_1)
  corpus_sec = torch.load(corpus_name_2)
  inputsPri = torch.load(inputs_name_1)
  inputsSec = torch.load(inputs_name_2)
  print("corpus data loaded\n")
end

no_of_sents = #corpus_pri.sequences
nm = pri_str.."-"..sec_str


vocab_sizePri = corpus_pri.no_of_words
vocab_sizeSec = corpus_sec.no_of_words

seq_lenPri = corpus_pri.longest_seq
seq_lenSec = corpus_sec.longest_seq

print(corpus_name_1.." "..corpus_pri.no_of_words.." words, "..#corpus_pri.sequences.." sents, "..corpus_pri.longest_seq.." sent max lenght\n")
print(corpus_name_2.." "..corpus_sec.no_of_words.." words, "..#corpus_sec.sequences.." sents, "..corpus_sec.longest_seq.." sent max lenght\n")
print(inputs_name_1.." "..inputs_name_2.."\n")


prm_names = {"lr","prev_lr","emb_size","hidden_size","lr_decay","threshold","dump_frq","max_epoch","epoch","batch_size","init","max_seq_len","win_size","opm_name","opm","score","num_of_samples"}

scenarios = {}

result_list = {}

lr_list_1000 = {}
lr_list_1000[1] = 0.001
lr_list_1000[51] = 0.00095
lr_list_1000[101] = 0.0009025

--for 1000, ok
--scenarios[#scenarios+1] = { lr = 0.0001;prev_lr = 0;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 256;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=0;batch_size = 100;init = 1;max_seq_len = 10;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; score = 0;num_of_samples = 0}

--scenarios[#scenarios+1] = { lr = 0.00001;prev_lr = 0.0001;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 256;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=1000;batch_size = 100;init = 1;max_seq_len = 10;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; score = 0;num_of_samples = 0}

--scenarios[#scenarios+1] = { lr = 0.000001;prev_lr = 0.00001;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 256;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=1100;batch_size = 100;init = 1;max_seq_len = 10;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; score = 0;num_of_samples = 0}

--scenarios[#scenarios+1] = { lr = 0.0000001;prev_lr = 0.000001;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 256;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=1150;batch_size = 100;init = 1;max_seq_len = 10;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; score = 0;num_of_samples = 0}

-- for 1000   acc = 51
--scenarios[#scenarios+1] = { lr = 0.00000001;prev_lr = 0.0000001;alpha = 0.95;lr_decay = 0.95; lr_list = nil ;emb_size = 64;hidden_size = 256;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=1200;batch_size = 100;init = 1;max_seq_len = 10;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; score = 0;num_of_samples = 0}

--for 1000  new
--scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0;alpha = 0.95;lr_decay = 0.95; lr_list = nil ;emb_size = 128;hidden_size = 512;momentum=0.5; threshold = 100 ;dump_frq = 100;max_epoch = 1000;epoch=0;batch_size = 50;init = 1;max_seq_len = 10;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; score = 0;num_of_samples = 0}

scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 128;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 1000;epoch=0;batch_size = 50;init = 1;max_seq_len = 10;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; score = 0;num_of_samples = 0}

model_save_flag = true
model_load_flag = false

function prepheadstr(prm_list)
  str = string.format("%s-%s-%d-lr=%3.7f-lr_decay=%3.7f-max_epoch=%d-batch_size=%d-init=%1.4f-max_seq_len=%d-win_size=%d" ,fhead,prm_list["opm_name"],prm_list["hidden_size"],prm_list["lr"],prm_list["lr_decay"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])
  return str
end

function model_save(nm,sec_str,prm_list,model)
  if model_save_flag == true then
    mname = model_save_path..nm..'_'..sec_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"].."."..fhead
    torch.save(mname..'.model',model:modelSec())
    torch.save(mname..'.LT',model:getLookupTableSec())
    mname = model_save_path..nm..'_'..pri_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"].."."..fhead
    torch.save(mname..'.model',model:modelPri())
    torch.save(mname..'.LT',model:getLookupTablePri())
    print("model saved \n")  
  end
end

function model_load(nm,sec_str,prm_list,model)
  mname = model_load_path..nm..'_'..sec_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"].."."..fhead
  print("model to be loaded : "..mname.."\n")  
  model.mdlSec = torch.load(mname..'.model')
  model.ltSec = torch.load(mname..'.LT')
  mname = model_load_path..nm..'_'..pri_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"].."."..fhead
  model.mdlPri = torch.load(mname..'.model')
  model.ltPri = torch.load(mname..'.LT')
  print("model loaded \n")  
end

for i=1,#scenarios do
  prm_list = scenarios[i]
  printf("lr %f, lr_decay %f,emb_size %d,hidden_size %d,max_epoch %d, batch_size %d, init %f, max_seq_len %d, win_size %d \n",prm_list["lr"],prm_list["lr_decay"],
    prm_list["emb_size"],prm_list["hidden_size"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])

  seq_lenPri = prm_list["max_seq_len"]
  seq_lenSec = prm_list["max_seq_len"]

  loss_logger = optim.Logger(fhead..'_loss_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"],true,prepheadstr(prm_list))
  grad_logger = optim.Logger(fhead..'_grad_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"],true,prepheadstr(prm_list))
  score_logger = optim.Logger(fhead..'_score_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"],true,prepheadstr(prm_list))


  --modelPri,modelSec,ltPri,ltSec = create_model(vocab_sizePri,vocab_sizeSec,seq_lenPri,seq_lenSec,prm_list)
  modelLSTM = BiLangModelLSTM(vocab_sizePri,vocab_sizeSec,seq_lenPri,seq_lenSec,prm_list)  
  if model_load_flag == true then
    model_load(nm,sec_str,prm_list,modelLSTM)
    sc,prm_list["num_of_samples"],prm_list["score"] = cv_test(cv_max,modelLSTM,prm_list["epoch"],inputsPri,inputsSec,test_size)
    score_logger:add{["lrPri"] = prm_list["prev_lr"],["lrSec"] = prm_list["prev_lr"], ['score '] = prm_list["score"] }
    score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['score '] = '+-'}
    score_logger:plot()  
  end
  
  modelLSTM = model_train(modelLSTM,inputsPri,inputsSec,prm_list,result_list)
  --prm_list["score"],prm_list["num_of_samples"] = test_model(modelPri,modelSec,prm_list["max_epoch"],inputsPri,inputsSec,100)
  sc,prm_list["num_of_samples"],prm_list["score"] = cv_test(cv_max,modelLSTM,prm_list["max_epoch"],inputsPri,inputsSec,test_size)
  score_logger:add{["lrPri"] = prm_list["lr"],["lrSec"] = prm_list["lr"], ['score '] = prm_list["score"] }
  score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['score '] = '+-'}
  score_logger:plot()  

  model_save(nm,sec_str,prm_list,modelLSTM)

end

print("end")
