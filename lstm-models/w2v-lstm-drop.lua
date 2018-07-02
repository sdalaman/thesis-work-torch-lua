require 'data-lstm-drop'
require 'model-lstm-drop'
require 'fileListTr'
require 'fileListTst'
require 'fileListTstSmp'
require 'tableutils'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'
require 'io'
require 'os'
require 'paths'

path = "/home/saban/work/additive/"
data_path = path.."data/"
model_save_path = path.."model-impl/lstm-models/model-files/"
model_load_path = path.."model-impl/lstm-models/model-files/"
corpus_path = path.."models/corpus/"

cutorch.setHeapTracking(true)
collectgarbage()

-- -dtype tok -dsize 10000 -corpus_load 1 -model_save 1 -model_load 0 -max_sntc_len 60
config = {}
config.dtype = "tok" -- data type
config.dsize = "" -- data type
config.corpus_load = 1
config.model_save = 1
config.model_load = 0
config.max_sntc_len = 60

cmd = torch.CmdLine()
cmd:option("-dtype", config.dtype)
cmd:option("-dsize", config.dsize)
cmd:option("-corpus_load", config.corpus_load)
cmd:option("-model_save", config.model_save)
cmd:option("-model_load", config.model_load)
cmd:option("-max_sntc_len", config.max_sntc_len)

params = cmd:parse(arg)
for param, value in pairs(params) do
    config[param] = value
end

print("data type "..config.dtype)
print("data size "..config.dsize)
print("corpus_load "..config.corpus_load)
print("model_save "..config.model_save)
print("model_load "..config.model_load)
print("max_stnc_len "..config.max_sntc_len)

corpus_load = config.corpus_load == 1
model_save_flag = config.model_save == 1
model_load_flag = config.model_load == 1
max_sentence_len = config.max_sntc_len
morph = config.dtype == "morph"

if morph == false then
  corpus_name_pri = "english."..config.dsize..".tok"
  corpus_name_sec = "turkish."..config.dsize..".tok"
else
  corpus_name_pri = "english."..config.dsize..".tok.morph"
  corpus_name_sec = "turkish."..config.dsize..".tok.morph"
end

if morph == false then
  test_data_pathEn = path.."data/cdlc_en_tr/test/englishTok"
  test_data_pathTr = path.."data/cdlc_en_tr/test/turkishTok"
  train_data_pathEn = path.."data/cdlc_en_tr/train/englishTok"
  train_data_pathTr = path.."data/cdlc_en_tr/train/turkishTok"
else
  test_data_pathEn = path.."data/cdlc_en_tr/test/englishMorph"
  test_data_pathTr = path.."data/cdlc_en_tr/test/turkishMorph"
  train_data_pathEn = path.."data/cdlc_en_tr/train/englishMorph"
  train_data_pathTr = path.."data/cdlc_en_tr/train/turkishMorph"
end

function printf(s,...)
  return io.write(s:format(...))
end

function printtensor(head,tsr)
  str = head.."\n"
  for i=1,tsr:size(1) do
    for j=1,tsr:size(2) do
      do str = str.." "..tsr[i][j] end
    end
    str=str.."\n"
  end
  print(str)
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end


initTotalGPUMem = {} 
initFreeGPUMem = {} 

function initGPUTotalMem()
  --print('## Initial Mem on GPU ##') 
  for i=1,cutorch.getDeviceCount() do 
    initFreeGPUMem[i], initTotalGPUMem[i] = cutorch.getMemoryUsage(i) 
    --print(i, initFreeGPUMem[i]) 
  end
end

--initGPUTotalMem()

function printGPUMemUsage(step)
  print('## Mem Used on GPU ## '..step) 
  for i=1,cutorch.getDeviceCount() do 
    free, _ = cutorch.getMemoryUsage(i) 
    print("GPU : "..i.." free mem : "..free..", used :"..(initTotalGPUMem[i] - free).." , total mem : "..initTotalGPUMem[i]) 
  end
end

initGPUTotalMem()
printGPUMemUsage(10)

function implode(sep,tbl,names)
  local newstr
  newstr = ""
  for i=1,#names do
    newstr = newstr ..names[i].."="..tbl[names[i]] .. sep
  end
  return newstr
end

function test_model(mPri,mSec,inpPri,inpSec,test_size)  
  local split = nn.SplitTable(2)
  local inPri = torch.zeros(test_size,inpPri:size()[2]):cuda()
  local inSec = torch.zeros(test_size,inpSec:size()[2]):cuda()
  cl = inpPri:size()[1]
  math.randomseed( os.time() )
  
  for i=1,test_size do
    ln = math.random(1,cl)
    inPri[i] = inpPri[ln]
    inSec[i] = inpSec[ln]
  end
  
  local inputPri = split:forward(inPri)
  local inputSec = split:forward(inSec)
  
  local all_rootsPri = mPri:forward( inputPri)
  local all_rootsSec = mSec:forward( inputSec)

  score = 0
  for idxPri = 1, all_rootsPri:size(1) do
    closest = idxPri
    for idxSec = 1, all_rootsSec:size(1) do
      if torch.dist(all_rootsPri[idxPri],all_rootsSec[idxSec]) <= torch.dist(all_rootsPri[idxPri],all_rootsSec[closest]) then
        closest = idxSec
      end
    end
    
    if idxPri == closest then
      score = score + 1 
    else
      --print("Closest to: "..idxPri.." is: ".. closest)
    end
  end
  return score,all_rootsPri:size(1)
end

function test_model_cv(model,insPri,insSec,prm_list)
  totscore = 0
  --model:modelPri():clearState()
  --model:modelSec():clearState()
  local mPri = model:modelPri()
  local mSec = model:modelSec()
  mPri:evaluate()
  mSec:evaluate()
  for i = 1,prm_list["max_cv"] do
    score,nsum = test_model(mPri,mSec,insPri,insSec,prm_list["test_size"])
    totscore = totscore + score
    print("-- score : "..score.."\n")
  end
  mPri:training()
  mSec:training()
  return totscore/prm_list["max_cv"]
end

function test_err(model,inpPri,inpSec,test_size) 
  local criterion = model:getCriterion()
  local split = nn.SplitTable(2)
  mPri = model:modelPri()
  mSec = model:modelSec()
  mPri:evaluate()
  mSec:evaluate()
  local inPri = torch.zeros(test_size,inpPri:size()[2]):cuda()
  local inSec = torch.zeros(test_size,inpSec:size()[2]):cuda()
  cl = inpPri:size()[1]
  math.randomseed( os.time() )
  
  for i=1,test_size do
    ln = math.random(1,cl)
    inPri[i] = inpPri[ln]
    inSec[i] = inpSec[ln]
  end
  local inputPri = split:forward(inPri)
  local inputSec = split:forward(inSec)
  local all_rootsPri = mPri:forward( inputPri)
  local all_rootsSec = mSec:forward( inputSec)

  tot_err = 0
  for i = 1, all_rootsPri:size(1) do
    local err = criterion:forward(all_rootsPri[i],all_rootsSec[i])
    tot_err = tot_err + err
  end
  mPri:training()
  mSec:training()
  return tot_err/test_size
end

function prepare_testdata(corpus1,corpus2,prm_list)
  totalLines = 0
  totscore = 0
  local totaltestInp1 = nil
  local totaltestInp2 = nil
  fileList = fileListTstSmp
  for fcnt = 1,#fileList do
    if morph == true then
      testcorpusfn1 = fileList[fcnt]..".tok.morph"
      testcorpusfn2 = fileList[fcnt]..".tok.morph"
    else 
      testcorpusfn1 = fileList[fcnt]..".tok"
      testcorpusfn2 = fileList[fcnt]..".tok"
    end
    sellist = {}
    f1,f1err = io.open(test_data_pathEn..testcorpusfn1,'r') 
    f2,f2err = io.open(test_data_pathTr..testcorpusfn2,'r')
    if f1 and f2 then
      f1:close()
      f2:close()
      --print("test corpus data will be prepared\n")
      testcorpus1 = Corpus(testcorpusfn1,test_data_pathEn)
      testcorpus2 = Corpus(testcorpusfn2,test_data_pathTr)
      testcorpus1:prepare_corpus(nil,prm_list["test_size"])
      testcorpus2:prepare_corpus(nil,prm_list["test_size"])
      testcorpus1.longest_seq = prm_list["max_seq_len"]
      testcorpus2.longest_seq = prm_list["max_seq_len"]
      testinputs1= testcorpus_shapedataLSTM(corpus1.vocab_map,testcorpus1):cuda()
      testinputs2= testcorpus_shapedataLSTM(corpus2.vocab_map,testcorpus2):cuda()
      if totalTestInp1 ~= nil then
        totalTestInp1 = torch.cat(totalTestInp1,testinputs1[{{},{1,prm_list["max_seq_len"]}}],1)
        totalTestInp2 = torch.cat(totalTestInp2,testinputs2[{{},{1,prm_list["max_seq_len"]}}],1)
      else
        totalTestInp1 = testinputs1[{{},{1,prm_list["max_seq_len"]}}]
        totalTestInp2 = testinputs2[{{},{1,prm_list["max_seq_len"]}}]
      end
    end
  end
  return totalTestInp1,totalTestInp2
end
        
function model_train(model,trInpPri,trInpSec,tstInpPri,tstInpSec,prm_list,stPri,stSec)
  collectgarbage()
  criterion = model:getCriterion()
  modelPri = model:modelPri()
  modelSec = model:modelSec()
  modelPri:training()
  modelSec:training()
  
  local w1Pri=modelPri:get(2):get(2):get(1):get(1):get(1):get(1):get(1)
  local w2Pri=modelPri:get(2):get(2):get(1):get(1):get(1):get(1):get(3)
  local w1pPri, w1pdxPri = w1Pri:getParameters()
  local w2pPri, w2pdxPri = w2Pri:getParameters()
  
  local lstm1Pri = modelPri:get(1):get(1):get(1):get(6)
  local lstm1pPri, lstm1pdxPri = lstm1Pri:getParameters()
  
  local w1Sec=modelPri:get(2):get(2):get(1):get(1):get(1):get(1):get(1)
  local w2Sec=modelPri:get(2):get(2):get(1):get(1):get(1):get(1):get(3)
  local w1pxSec, w1pdxSec = w1Sec:getParameters()
  local w2pxSec, w2pdxSec = w2Sec:getParameters()
  
  beginning_time = torch.tic()
  local split = nn.SplitTable(2)
  local paramsPri, gradParamsPri = modelPri:getParameters()
  local xPri, dl_dxPri = modelPri:parameters()
  local paramsSec, gradParamsSec = modelSec:getParameters()
  local xSec, dl_dxSec = modelSec:parameters()
  
  local optimStatePriSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"],weightDecay =prm_list["weight_decay"],nesterov=true,dampening=0}
  local optimStateSecSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"],weightDecay =prm_list["weight_decay"],nesterov=true,dampening=0}
  
  local optimStatePriRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"],weightDecay =prm_list["weight_decay"]}
  local optimStateSecRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"],weightDecay =prm_list["weight_decay"]}
  
  if prm_list["epoch"] == 0 then
    if prm_list["opm_name"] == "sgd" then
      print("Optimization is SGD \n")
      statePri = optimStatePriSgd
      stateSec = optimStateSecSgd
    end
  
    if prm_list["opm_name"] == "rmsprop" then
      print("Optimization is RmsProp \n")
      statePri = optimStatePriRmsProp
      stateSec = optimStateSecRmsProp
    end
  else
    statePri = stPri
    stateSec = stSec
    statePri.learningRate = prm_list["lr"]
    stateSec.learningRate = prm_list["lr"]
  end

  no_of_sents = trInpPri:size()[1]
  for i =prm_list["epoch"]+1,prm_list["max_epoch"] do
    current_lossPri = 0
    current_lossSec = 0
    totalGradWeightPri = 0
    totalGradWeightSec = 0
    local inds = torch.range(1, no_of_sents,prm_list["batch_size"])
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    local k = 0
    
    beginning_time = torch.tic()
    print('begin time : '..os.date("%HH-%MM-%SS"))
    for j = 1, math.floor(no_of_sents/prm_list["batch_size"]) do 
                --get input row and target
        local start = inds[shuffle[j]]
        local endd = inds[shuffle[j]]+prm_list["batch_size"]-1
        if((start > trInpPri:size(1)) or (endd > trInpPri:size(1))) then
          k = k + 1
          endd = trInpPri:size(1)
        end
        local inputPri = split:forward(trInpPri[{{start,endd},{1,prm_list["max_seq_len"]}}])
        local inputSec = split:forward(trInpSec[{{start,endd},{1,prm_list["max_seq_len"]}}])
        
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
          totalGradWeightPri = totalGradWeightPri + torch.sum(torch.pow(gradParamsPri,2))
          -- penalties (L1 and L2):
          if prm_list["coefL1"] ~= 0 or prm_list["coefL2"] ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign
            -- Loss:
            err = err + prm_list["coefL1"] * norm(paramsPri,1)
            err = err + prm_list["coefL2"] * norm(paramsPri,2)^2/2
            -- Gradients:
            gradParamsPri:add( sign(paramsPri):mul(prm_list["coefL1"]) + paramsPri:clone():mul(prm_list["coefL2"]) )
          end
          return err, gradParamsPri
        end
        
        function fevalSec(params)
          local outputPri = modelPri:forward( inputPri)
          local outputSec = modelSec:forward( inputSec)
          local err = criterion:forward( outputSec, outputPri)
          local gradOutputs = criterion:backward(outputSec, outputPri)
          modelSec:backward(inputSec, gradOutputs)
          totalGradWeightSec = totalGradWeightSec + torch.sum(torch.pow(gradParamsSec,2))
          -- penalties (L1 and L2):
          if prm_list["coefL1"] ~= 0 or prm_list["coefL2"] ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign
            -- Loss:
            err = err + prm_list["coefL1"] * norm(paramsSec,1)
            err = err + prm_list["coefL2"] * norm(paramsSec,2)^2/2
            -- Gradients:
            gradParamsSec:add( sign(paramsSec):mul(prm_list["coefL1"]) + paramsSec:clone():mul(prm_list["coefL2"]) )
          end
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
    durationSeconds = torch.toc(beginning_time)
    
    w1Pri=modelPri:get(2):get(2):get(1):get(1):get(1):get(1):get(1)
    w2Pri=modelPri:get(2):get(2):get(1):get(1):get(1):get(1):get(3)
    w1pPri, w1pdxPri = w1Pri:getParameters()
    w2pPri, w2pdxPri = w2Pri:getParameters()
    w1Sec=modelPri:get(2):get(2):get(1):get(1):get(1):get(1):get(1)
    w2Sec=modelPri:get(2):get(2):get(1):get(1):get(1):get(1):get(3)
    w1pSec, w1pdxSec = w1Sec:getParameters()
    w2pSec, w2pdxSec = w2Sec:getParameters()

    print("Pri Attn Prm: "..torch.sum(torch.pow(w1pPri,2)).." / "..torch.sum(torch.pow(w2pPri,2)))
    print("Sec Attn Prm: "..torch.sum(torch.pow(w1pSec,2)).." / "..torch.sum(torch.pow(w2pSec,2)))
    print("Pri Attn Grad: "..torch.sum(torch.pow(w1pdxPri,2)).." / "..torch.sum(torch.pow(w2pdxPri,2)))
    print("Sec Attn Grad: "..torch.sum(torch.pow(w1pdxSec,2)).." / "..torch.sum(torch.pow(w2pdxSec,2)))
    
    lstm1Pri = modelPri:get(1):get(1):get(1):get(6)
    lstm1pPri, lstm1pdxPri = lstm1Pri:getParameters()
    print("Pri LSTM1 : "..torch.sum(torch.pow(lstm1pPri,2)).." / "..torch.sum(torch.pow(lstm1pdxPri,2)))  
    --printtensor("Pri Attn : ",model.attnW2Pri.weight)
    --printtensor("Sec Attn : ",model.attnW2Sec.weight)
    print('end time : '..os.date("%HH-%MM-%SS")..'  time elapsed(sec): '.. printtime(durationSeconds))
    tst_err = test_err(model,tstInpPri,tstInpSec,prm_list["err_test_size"])
    loss_logger:add{['training error pri'] = current_lossPri ,['training error sec'] = current_lossSec,['test err'] = tst_err}
    loss_logger:style{['training error pri'] = '+-' , ['training error sec'] = '+-',['test err'] ='+-' }
    grad_logger:add{['grad weight pri'] = totalGradWeightPri ,['grad weight sec'] = totalGradWeightSec }
    grad_logger:style{['grad weight pri'] = '+-' , ['grad weight sec'] = '+-' }
    
    if i % 10 == 0  then
      loss_logger:plot()
      grad_logger:plot() 
    end
    
    if prm_list["opm_name"] == "rmsprop" then
      local clrPri = statePri.learningRate / (1 + i*prm_list["lr_decay"]) 
      local clrSec = stateSec.learningRate / (1 + i*prm_list["lr_decay"]) 
      statePri.learningRate = clrPri
      stateSec.learningRate =  clrSec
    end
    
    if i % prm_list["dump_frq"] == 0 then
      print(" testing train data ..\n")
      prm_list["train_score"] = test_model_cv(model,trInpPri,trInpSec,prm_list)
      print(" testing test data ..\n")
      prm_list["test_score"] = test_model_cv(model,tstInpPri,tstInpSec,prm_list)
      print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
      score_logger:add{["lr"] = statePri.learningRate, ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
      score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
      score_logger:plot()  
      loss_logger:plot() 
      grad_logger:plot()  
      model_save(model,prm_list,statePri,stateSec)
    end
  
    if i % prm_list["threshold"] == 0 then 
      --statePri.learningRate = statePri.learningRate * prm_list["lr_decay"] 
      --stateSec.learningRate = stateSec.learningRate * prm_list["lr_decay"]
      --prm_list["lr"] = statePri.learningRate
    end
    
  end

  durationSeconds = torch.toc(beginning_time)
  print('time elapsed:'.. printtime(durationSeconds))
  return model
end

corpus1_str = string.sub(corpus_name_pri,1,2)
corpus2_str = string.sub(corpus_name_sec,1,2)
nm = corpus1_str.."-"..corpus2_str

vocab1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".lstm.vocab" -- vocab file  for lang 1
vocab2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".lstm.vocab" -- vocab file  for lang 2
corpus_name1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".lstm.corpus.tch"
corpus_name2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".lstm.corpus.tch"
inputs_name1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".lstm.inputs.tch"
inputs_name2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".lstm.inputs.tch"

prm_names = {"lr","prev_lr","emb_size","hidden_size","lr_decay","threshold","dump_frq","max_epoch","epoch","batch_size","init","max_seq_len","win_size","opm_name","opm","score","num_of_samples"}

scenarios = {}

if config.dtype ~= "morph" then
--for 10000 new 1
--batch_size = 50;init = 1 weight_decay = 1e-4 lr_decay = 1e-4 lr = 0.01
--scenarios[#scenarios+1] = { lr = 0.01;prev_lr = 0;alpha = 0.95;lr_decay = 1e-4;weight_decay = 0; lr_list = nil ;emb_size = 64;hidden_size = 128;num_of_hidden=3;out_size=64;gate_type = 'LSTM';momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=0;batch_size = 50;init = 0.001;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0;fhead="";dropoutProb=0.5}

--scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0.01;alpha = 0.95;lr_decay = 0;weight_decay = 0; lr_list = nil ;emb_size = 64;hidden_size = 128;num_of_hidden=3;out_size=64;gate_type = 'LSTM';momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=100;batch_size = 100;init = 0.001;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0;fhead="";dropoutProb=0.5}

--for 10000 new 1 attn
scenarios[#scenarios+1] = { lr = 0.01;prev_lr = 0;alpha = 0.95;lr_decay = 0;weight_decay = 0; lr_list = nil ;emb_size = 64;hidden_size = 128;num_of_hidden=2;out_size=64;gate_type = 'LSTM';momentum=0.5; threshold = 50 ;dump_frq = 10;max_epoch = 2000;epoch=0;batch_size = 100;init = 0.001;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0;fhead="";dropoutProb=0.5;attnW1Out=30;attnW2Out=1}

else

--for 10000 morph new 1
scenarios[#scenarios+1] = { lr = 0.1;prev_lr = 0;alpha = 0.95;lr_decay = 0;weight_decay = 0; lr_list = nil ;emb_size = 64;hidden_size = 128;num_of_hidden=2;out_size=64;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 5000;epoch=0;batch_size = 100;init = 0.001;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok.morph";testcorpus2 = "turkish.10000.tok.morph";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0;fhead="";dropoutProb=0.5;attnW1Out=30;attnW2Out=10}

end

if corpus_load == false then
  corpus1 = Corpus(corpus_name_pri,data_path)
  corpus2 = Corpus(corpus_name_sec,data_path)
  corpus1:prepare_corpus(nil)
  corpus2:prepare_corpus(nil)
  corpus1.longest_seq = max_sentence_len
  corpus2.longest_seq = max_sentence_len
  trainInp1=corpus1:get_dataLSTM():cuda()
  trainInp2=corpus2:get_dataLSTM():cuda()

  torch.save(corpus_name1,corpus1)
  torch.save(corpus_name2,corpus2)
  torch.save(inputs_name1,trainInp1)
  torch.save(inputs_name2,trainInp2)
  torch.save(vocab1,corpus1.vocab_map)
  torch.save(vocab2,corpus2.vocab_map)
  print("corpus data prepared and saved\n")
else
  print("corpus data will be loaded\n")
  corpus1 = torch.load(corpus_name1)
  corpus2 = torch.load(corpus_name2)
  trainInp1 = torch.load(inputs_name1)
  trainInp2 = torch.load(inputs_name2)
  print("corpus data loaded\n")
end

no_of_sents = #corpus1.sequences
vocab_size1 = corpus1.no_of_words
vocab_size2 = corpus2.no_of_words
seq_len1 = corpus1.longest_seq
seq_len2 = corpus2.longest_seq

print(corpus_name1.." "..corpus1.no_of_words.." words, "..#corpus1.sequences.." sents, "..corpus1.longest_seq.." sent max lenght\n")
print(corpus_name2.." "..corpus2.no_of_words.." words, "..#corpus2.sequences.." sents, "..corpus2.longest_seq.." sent max lenght\n")
print(inputs_name1.." "..inputs_name2.."\n")


function prepheadstr(prm_list)
  str = string.format("%s-%s-%d-%d-lr=%3.7f-lr_decay=%3.7f-w_decay=%3.7f-max_epoch=%d-batch_size=%d-init=%1.4f-max_seq_len=%d-win_size=%d-attn" ,prm_list["fhead"],prm_list["opm_name"],prm_list["emb_size"],prm_list["hidden_size"],prm_list["lr"],prm_list["lr_decay"],prm_list["weight_decay"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])
  return str
end

function model_save(model,prm_list,stPri,stSec)
  if model_save_flag == true then
    mname = model_save_path..nm..'_'..corpus2_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..prm_list["fhead"]
    model:modelSec():clearState()
    torch.save(mname..'.model',model:modelSec())
    torch.save(mname..'.LT',model:getLookupTableSec())
    torch.save(mname..'.sts',stSec)
    print("model saved "..mname.."\n")  
    mname = model_save_path..nm..'_'..corpus1_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..prm_list["fhead"]
    model:modelPri():clearState()
    torch.save(mname..'.model',model:modelPri())
    torch.save(mname..'.LT',model:getLookupTablePri())
    torch.save(mname..'.sts',stPri)
    print("model saved "..mname.."\n")  
  end
end

function model_load(model,prm_list)
  mname = model_load_path..nm..'_'..corpus2_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..prm_list["fhead"]
  print("model to be loaded : "..mname.."\n")  
  model.mdlSec = torch.load(mname..'.model')
  model.ltSec = torch.load(mname..'.LT')
  stsSec = torch.load(mname..'.sts')
  mname = model_load_path..nm..'_'..corpus1_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..prm_list["fhead"]
  print("model to be loaded : "..mname.."\n")  
  model.mdlPri = torch.load(mname..'.model')
  model.ltPri = torch.load(mname..'.LT')
  stsPri = torch.load(mname..'.sts')
  print("model loaded \n")  
  print("Opt states loaded \n")
  return stsPri,stsSec
end

for i=1,#scenarios do
  prm_list = scenarios[i]
  printf("lr %f, lr_decay %f,emb_size %d,hidden_size %d,max_epoch %d, batch_size %d, init %f, max_seq_len %d, win_size %d \n",prm_list["lr"],prm_list["lr_decay"],
    prm_list["emb_size"],prm_list["hidden_size"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])

  if morph == false then
    prm_list["fhead"] = "lstm-avg-drop-attn-"..config.dsize
  else
    prm_list["fhead"] = "lstm-avg-drop-attn-"..config.dsize.."-morph"
  end

  seq_len1 = prm_list["max_seq_len"]
  seq_len2 = prm_list["max_seq_len"]

  loss_logger = optim.Logger(prm_list["fhead"]..'_loss_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["emb_size"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
  grad_logger = optim.Logger(prm_list["fhead"]..'_grad_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["emb_size"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
  score_logger = optim.Logger(prm_list["fhead"]..'_score_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["emb_size"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
 
  testInp1, testInp2 = prepare_testdata(corpus1,corpus2,prm_list)

  modelLSTM = BiLangModelLSTMAvgAttn(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
    
  optStatePri = nil
  optStateSec = nil
  
  if model_load_flag == true then
    optStatePri,optStateSec = model_load(modelLSTM,prm_list)
    print(" testing train data ..\n")
--    prm_list["train_score"] = test_model_cv(modelLSTM,trainInp1,trainInp2,prm_list)
    print(" testing test data ..\n")
--    prm_list["test_score"] = test_model_cv(modelLSTM,testInp1,testInp2,prm_list)
    print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
    score_logger:add{["lr"] = prm_list["prev_lr"], ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
    score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
    score_logger:plot()  
    print(modelLSTM.mdlPri)
    print(" model loaded ..\n")
  else
    print(modelLSTM.mdlPri)
    print(" model created ..\n")
  end
 
  print("model name : BiLangModelLSTMAvgAttn")

  print(" training started ..\n")
  modelLSTM = model_train(modelLSTM,trainInp1,trainInp2,testInp1,testInp2,prm_list,optStatePri,optStateSec)
  print(" testing train data ..\n")
  prm_list["train_score"] = test_model_cv(modelLSTM,trainInp1,trainInp2,prm_list)
  print(" testing test data ..\n")
  prm_list["test_score"] = test_model_cv(modelLSTM,testInp1,testInp2,prm_list)
  print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
  score_logger:add{["lr"] = prm_list["lr"],['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
  score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
  score_logger:plot()  
  model_save(modelLSTM,prm_list,optStatePri,optStateSec)

end

print("end")
