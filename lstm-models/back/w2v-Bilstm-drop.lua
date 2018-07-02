require 'data-lstm-drop'
require 'model-lstm-drop'
require 'fileListTr'
require 'fileListTst'
require 'fileListTstSmp'
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

-- -dtype tok -dsize 10000 -corpus_load 1 -model_save 0 -model_load 0 -max_sntc_len 10
config = {}
config.dtype = "tok" -- data type
config.dsize = "" -- data type
config.corpus_load = 1
config.model_save = 1
config.model_load = 0
config.max_sntc_len = 40

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
  fhead = "BiLstm-Avg-drop-"..config.dsize
else
  corpus_name_pri = "english."..config.dsize..".tok.morph"
  corpus_name_sec = "turkish."..config.dsize..".tok.morph"
  fhead = "BiLstm-Avg-drop-"..config.dsize.."-morph"
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

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

initTotalGPUMem = {} 
initFreeGPUMem = {} 

function initGPUTotalMem()
  print('## Initial Mem on GPU ##') 
  for i=1,cutorch.getDeviceCount() do 
    initFreeGPUMem[i], initTotalGPUMem[i] = cutorch.getMemoryUsage(i) 
    print(i, initFreeGPUMem[i]) 
  end
end

--initGPUTotalMem()

function printGPUMemUsage(step)
  print('## Mem Used on GPU ## '..step) 
  for i=1,cutorch.getDeviceCount() do 
    free, _ = cutorch.getMemoryUsage(i) 
    print(i,free, initFreeGPUMem[i] - free) 
  end
end

function implode(sep,tbl,names)
  local newstr
  newstr = ""
  for i=1,#names do
    newstr = newstr ..names[i].."="..tbl[names[i]] .. sep
  end
  return newstr
end

function test_model(mPri,mSec,inpPri,inpSec,inpPriR,inpSecR,test_size)
  local split = nn.SplitTable(2)
  local inPri = torch.zeros(test_size,inpPri:size()[2]):cuda()
  local inSec = torch.zeros(test_size,inpSec:size()[2]):cuda()
  local inPriR = torch.zeros(test_size,inpPriR:size()[2]):cuda()
  local inSecR = torch.zeros(test_size,inpSecR:size()[2]):cuda()
  cl = inpPri:size()[1]
  math.randomseed( os.time() )
  
  for i=1,test_size do
    ln = math.random(1,cl)
    inPri[i] = inpPri[ln]
    inSec[i] = inpSec[ln]
    inPriR[i] = inpPriR[ln]
    inSecR[i] = inpSecR[ln]
  end
  
  local inputPri = split:forward(inPri)
  local inputSec = split:forward(inSec)
  local inputPriR = split:forward(inPriR)
  local inputSecR = split:forward(inSecR)
  
  local all_rootsPri = mPri:forward({inputPri,inputPriR})
  local all_rootsSec = mSec:forward({inputSec,inputSecR})

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

function test_model_cv(model,insPri,insSec,insPriR,insSecR,prm_list)
  totscore = 0
  --model:modelPri():clearState()
  --model:modelSec():clearState()
  local mPri = model:modelPri()
  local mSec = model:modelSec()
  mPri:evaluate()
  mSec:evaluate()
  
  for i = 1,prm_list["max_cv"] do
    score,nsum = test_model(mPri,mSec,insPri,insSec,insPriR,insSecR,prm_list["test_size"])
    totscore = totscore + score
    print("-- score : "..score.."\n")
  end
  mPri:training()
  mSec:training()
  return totscore/prm_list["max_cv"]
end

function test_err(model,inpPri,inpSec,insPriR,insSecR,test_size) 
  local criterion = model:getCriterion()
  local split = nn.SplitTable(2)
  mPri = model:modelPri()
  mSec = model:modelSec()
  mPri:evaluate()
  mSec:evaluate()
  local inPri = torch.zeros(test_size,inpPri:size()[2]):cuda()
  local inSec = torch.zeros(test_size,inpSec:size()[2]):cuda()
  local inPriR = torch.zeros(test_size,inpPriR:size()[2]):cuda()
  local inSecR = torch.zeros(test_size,inpSecR:size()[2]):cuda()
  cl = inpPri:size()[1]
  math.randomseed( os.time() )
  
  for i=1,test_size do
    ln = math.random(1,cl)
    inPri[i] = inpPri[ln]
    inSec[i] = inpSec[ln]
    inPriR[i] = inpPriR[ln]
    inSecR[i] = inpSecR[ln]   
  end
  
  local inputPri = split:forward(inPri)
  local inputSec = split:forward(inSec)
  local inputPriR = split:forward(inPriR)
  local inputSecR = split:forward(inSecR)
  
  local all_rootsPri = mPri:forward({inputPri,inputPriR})
  local all_rootsSec = mSec:forward({inputSec,inputSecR})

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
  local totaltestInp1r = nil
  local totaltestInp2r = nil
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
      testinputs1,testinputs1r= testcorpus_shapedataBiLSTM(corpus1,testcorpus1)
      testinputs2,testinputs2r= testcorpus_shapedataBiLSTM(corpus2,testcorpus2)
      if totalTestInp1 ~= nil then
        totalTestInp1 = torch.cat(totalTestInp1,testinputs1[{{},{1,prm_list["max_seq_len"]}}],1)
        totalTestInp2 = torch.cat(totalTestInp2,testinputs2[{{},{1,prm_list["max_seq_len"]}}],1)
        totalTestInp1r = torch.cat(totalTestInp1r,testinputs1r[{{},{1,prm_list["max_seq_len"]}}],1)
        totalTestInp2r = torch.cat(totalTestInp2r,testinputs2r[{{},{1,prm_list["max_seq_len"]}}],1)
      else
        totalTestInp1 = testinputs1[{{},{1,prm_list["max_seq_len"]}}]
        totalTestInp2 = testinputs2[{{},{1,prm_list["max_seq_len"]}}]
        totalTestInp1r = testinputs1r[{{},{1,prm_list["max_seq_len"]}}]
        totalTestInp2r = testinputs2r[{{},{1,prm_list["max_seq_len"]}}]
      end
    end
  end
  return totalTestInp1,totalTestInp2,totalTestInp1r,totalTestInp2r
end
        
function model_train(model,trInpPri,trInpSec,trInpPriR,trInpSecR,tstInpPri,tstInpSec,tstInpPriR,tstInpSecR,prm_list,stPri,stSec)
  collectgarbage()
  criterion = model:getCriterion()
  modelPri = model:modelPri()
  modelSec = model:modelSec()
  modelPri:training()
  modelSec:training()
  beginning_time = torch.tic()
  local split = nn.SplitTable(2)
  local paramsPri, gradParamsPri = modelPri:getParameters()
  local xPri2, dl_dxPri2= modelPri:parameters()
  local paramsSec, gradParamsSec = modelSec:getParameters()
  local xSec2, dl_dxSec2 = modelSec:parameters()
  
  local optimStatePriSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"],weightDecay =prm_list["weight_decay"]}
  local optimStateSecSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"],weightDecay =prm_list["weight_decay"]}
  
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

  for i =prm_list["epoch"]+1,prm_list["max_epoch"] do
    be = torch.tic()
    current_lossPri = 0
    current_lossSec = 0
    totalGradWeightPri = 0
    totalGradWeightSec = 0
    local inds = torch.range(1, no_of_sents,prm_list["batch_size"])
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    local k = 0
    
    tt = math.floor(no_of_sents/prm_list["batch_size"])
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
        local inputPriR = split:forward(trInpPriR[{{start,endd},{1,prm_list["max_seq_len"]}}])
        local inputSecR = split:forward(trInpSecR[{{start,endd},{1,prm_list["max_seq_len"]}}])
        
        
        modelPri:zeroGradParameters()
        modelSec:zeroGradParameters()
        modelPri:forget()
        modelSec:forget()

        -- print( target)
        function fevalPri(params)
          local outputPri = modelPri:forward({inputPri,inputPriR})
          local outputSec = modelSec:forward({inputSec,inputSecR})
          local err = criterion:forward( outputPri, outputSec)
          local gradOutputs = criterion:backward(outputPri, outputSec)
          modelPri:backward({inputPri,inputPriR}, gradOutputs)
          totalGradWeightPri = totalGradWeightPri + torch.sum(torch.pow(gradParamsPri,2))
          return err, gradParamsPri
        end
        
        function fevalSec(params)
          local outputPri = modelPri:forward({inputPri,inputPriR})
          local outputSec = modelSec:forward({inputSec,inputSecR})
          local err = criterion:forward( outputSec, outputPri)
          local gradOutputs = criterion:backward(outputSec, outputPri)
          modelSec:backward({inputSec,inputSecR}, gradOutputs)
          totalGradWeightSec = totalGradWeightSec + torch.sum(torch.pow(gradParamsSec,2))
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
    tst_err = test_err(model,tstInpPri,tstInpSec,tstInpPriR,tstInpSecR,prm_list["err_test_size"])
    loss_logger:add{['training error pri'] = current_lossPri ,['training error sec'] = current_lossSec,['test err'] = tst_err}
    loss_logger:style{['training error pri'] = '+-' , ['training error sec'] = '+-',['test err'] ='+-' }
    grad_logger:add{['grad weight pri'] = totalGradWeightPri ,['grad weight sec'] = totalGradWeightSec }
    grad_logger:style{['grad weight pri'] = '+-' , ['grad weight sec'] = '+-' }
    
    if i % 10 == 0  then
      loss_logger:plot()
    end
    
    if prm_list["opm_name"] == "rmsprop" then
      --local clrPri = statePri.learningRate / (1 + i*prm_list["lr_decay"]) 
      --local clrSec = stateSec.learningRate / (1 + i*prm_list["lr_decay"]) 
      --statePri.learningRate = clrPri
      --stateSec.learningRate =  clrSec
    end
    
    if i % prm_list["dump_frq"] == 0 then
      print(" testing train data ..\n")
      prm_list["train_score"] = test_model_cv(model,trInpPri,trInpSec,trInpPriR,trInpSecR,prm_list)
      print(" testing test data ..\n")
      prm_list["test_score"] = test_model_cv(model,tstInpPri,tstInpSec,tstInpPriR,tstInpSecR,prm_list)
      print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
      score_logger:add{["lr"] = statePri.learningRate, ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
      score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
      score_logger:plot()  
      loss_logger:plot()  
      grad_logger:plot()  
      model_save(model,prm_list,statePri,stateSec)
    end
  

    
    if i % prm_list["threshold"] == 0 then 
--      statePri.learningRate = statePri.learningRate * prm_list["lr_decay"] 
--      stateSec.learningRate = stateSec.learningRate * prm_list["lr_decay"]
--      prm_list["lr"] = statePri.learningRate
    end
    
  end

  durationSeconds = torch.toc(beginning_time)
  print('time elapsed:'.. printtime(durationSeconds))
  return model
end

corpus1_str = string.sub(corpus_name_pri,1,2)
corpus2_str = string.sub(corpus_name_sec,1,2)
nm = corpus1_str.."-"..corpus2_str

vocab1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".BiLstm.vocab" -- vocab file  for lang 1
vocab2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".BiLstm.vocab" -- vocab file  for lang 2
corpus_name1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".BiLstm.corpus.tch"
corpus_name2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".BiLstm.corpus.tch"
inputs_name1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".BiLstm.inputs.tch"
inputs_name2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".BiLstm.inputs.tch"
inputs_name1r = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".rv.BiLstm.inputs.tch"
inputs_name2r = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".rv.BiLstm.inputs.tch"

prm_names = {"lr","prev_lr","emb_size","hidden_size","lr_decay","threshold","dump_frq","max_epoch","epoch","batch_size","init","max_seq_len","win_size","opm_name","opm","score","num_of_samples"}

scenarios = {}

if config.dtype ~= "morph" then
--for 10000 BISCAVG 1 
--scenarios[#scenarios+1] = { lr = 0.0001;prev_lr = 0;alpha = 0.95;lr_decay = 1e-6;weight_decay = 1e-4; lr_list = nil ;emb_size = 64;hidden_size = 512;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 200;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}

--for 10000 BIAVG 1 
--scenarios[#scenarios+1] = { lr = 0.0001;prev_lr = 0;alpha = 0.95;lr_decay = 1e-6;weight_decay = 1e-4; lr_list = nil ;emb_size = 128;hidden_size = 1024;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 50;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}

else
  --for 10000 BISCAVG 1 morph 
 -- scenarios[#scenarios+1] = { lr = 0.0001;prev_lr = 0;alpha = 0.95;lr_decay = 1e-6;weight_decay = 1e-4;lr_list = nil ;emb_size = 64; hidden_size = 512;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 200;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}
  
--for 10000 BIAVG 1 morph 
--scenarios[#scenarios+1] = { lr = 0.0001;prev_lr = 0;alpha = 0.95;lr_decay = 1e-6;weight_decay = 1e-4;lr_list = nil ;emb_size = 128; hidden_size = 1024;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 5000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}
 
end


if corpus_load == false then
  corpus1 = Corpus(corpus_name_pri,data_path)
  corpus2 = Corpus(corpus_name_sec,data_path)
  corpus1:prepare_corpus(nil)
  corpus2:prepare_corpus(nil)
  corpus1.longest_seq = max_sentence_len
  corpus2.longest_seq = max_sentence_len
  corpus1:get_dataBiLSTM():cuda()
  corpus2:get_dataBiLSTM():cuda()
  trainInp1=corpus1.f
  trainInp1r=corpus1.b
  trainInp2=corpus2.f
  trainInp2r=corpus2.b
  torch.save(corpus_name1,corpus1)
  torch.save(corpus_name2,corpus2)
  torch.save(inputs_name1,trainInp1)
  torch.save(inputs_name2,trainInp2)
  torch.save(inputs_name1r,trainInp1r)
  torch.save(inputs_name2r,trainInp2r)
  torch.save(vocab1,corpus1.vocab_map)
  torch.save(vocab2,corpus2.vocab_map)
  print("corpus data prepared and saved\n")
else
  print("corpus data will be loaded\n")
  corpus1 = torch.load(corpus_name1)
  corpus2 = torch.load(corpus_name2)
  trainInp1 = torch.load(inputs_name1)
  trainInp1r = torch.load(inputs_name1r)
  trainInp2 = torch.load(inputs_name2)
  trainInp2r = torch.load(inputs_name2r)
  print("corpus data loaded\n")
end

--dumpCorpus(corpus1,"./corpus.english.10000.txt")
--dumpCorpus(corpus2,"./corpus.turkish.10000.txt")

no_of_sents = #corpus1.sequences
vocab_size1 = corpus1.no_of_words
vocab_size2 = corpus2.no_of_words
seq_len1 = corpus1.longest_seq
seq_len2 = corpus2.longest_seq

print(corpus_name1.." "..corpus1.no_of_words.." words, "..#corpus1.sequences.." sents, "..corpus1.longest_seq.." sent max lenght\n")
print(corpus_name2.." "..corpus2.no_of_words.." words, "..#corpus2.sequences.." sents, "..corpus2.longest_seq.." sent max lenght\n")
print(inputs_name1.." "..inputs_name2.."\n")

function prepheadstr(prm_list)
  str = string.format("%s-%s-%d-%d-lr=%3.7f-lr_decay=%3.7f-w_decay=%3.7f-max_epoch=%d-batch_size=%d-init=%1.4f-max_seq_len=%d-win_size=%d" ,fhead,prm_list["opm_name"],prm_list["emb_size"],prm_list["hidden_size"],prm_list["lr"],prm_list["lr_decay"],prm_list["weight_decay"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])
  return str
end

function model_save(model,prm_list,stPri,stSec)
  if model_save_flag == true then
    mname = model_save_path..nm..'_'..corpus2_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..fhead
    model:modelSec():clearState()
    torch.save(mname..'.model',model:modelSec())
    torch.save(mname..'.LT',model:getLookupTableSec())
    torch.save(mname..'.sts',stSec)
    print("model saved "..mname.."\n")  
    mname = model_save_path..nm..'_'..corpus1_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..fhead
    model:modelPri():clearState()
    torch.save(mname..'.model',model:modelPri())
    torch.save(mname..'.LT',model:getLookupTablePri())
    torch.save(mname..'.sts',stPri)
    print("model saved "..mname.."\n")  
  end
end

function model_load(model,prm_list)
  mname = model_load_path..nm..'_'..corpus2_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..fhead
  print("model to be loaded : "..mname.."\n")  
  model.brnnSec = torch.load(mname..'.model')
  model.sharedLookupTableSec = torch.load(mname..'.LT')
  stsSec = torch.load(mname..'.sts')
  mname = model_load_path..nm..'_'..corpus1_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..fhead
  print("model to be loaded : "..mname.."\n")
  model.brnnPri = torch.load(mname..'.model')
  model.sharedLookupTablePri = torch.load(mname..'.LT')
  stsPri = torch.load(mname..'.sts')
  print("model loaded \n")  
  print("Opt states loaded \n")
  return stsPri,stsSec
end

for i=1,#scenarios do
  prm_list = scenarios[i]
  printf("lr %f, lr_decay %f,emb_size %d,hidden_size %d,max_epoch %d, batch_size %d, init %f, max_seq_len %d, win_size %d \n",prm_list["lr"],prm_list["lr_decay"],
    prm_list["emb_size"],prm_list["hidden_size"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])

  seq_len1 = prm_list["max_seq_len"]
  seq_len2 = prm_list["max_seq_len"]

--  trainInp1 = trainInp1[{{},{1,prm_list["max_seq_len"]}}]
--  trainInp2 = trainInp2[{{},{1,prm_list["max_seq_len"]}}]
--  trainInp1r = trainInp1r[{{},{1,prm_list["max_seq_len"]}}]
--  trainInp2r = trainInp2r[{{},{1,prm_list["max_seq_len"]}}]

  loss_logger = optim.Logger(fhead..'_loss_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
  grad_logger = optim.Logger(fhead..'_grad_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
  score_logger = optim.Logger(fhead..'_score_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
 
  testInp1, testInp2,testInp1r, testInp2r = prepare_testdata(corpus1,corpus2,prm_list)

  modelLSTM = BiLangModelBiLSTMAvg(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
  print(" model created ..\n")
  optStatePri = nil
  optStateSec = nil

  
  if model_load_flag == true then
    optStatePri,optStateSec = model_load(modelLSTM,prm_list)
    print(" testing train data ..\n")
    prm_list["train_score"] = test_model_cv(modelLSTM,trainInp1,trainInp2,trainInp1r,trainInp2r,prm_list)
    print(" testing test data ..\n")
    prm_list["test_score"] = test_model_cv(modelLSTM,testInp1,testInp2,testInp1r,testInp2r,prm_list)
    print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
    score_logger:add{["lr"] = prm_list["prev_lr"],['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
    score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
    score_logger:plot()  
  end
  
  print(" training started ..\n")
  modelLSTM = model_train(modelLSTM,trainInp1,trainInp2,trainInp1r,trainInp2r,testInp1,testInp2,testInp1r,testInp2r,prm_list,optStatePri,optStateSec)
  print(" testing train data ..\n")
  prm_list["train_score"] = test_model_cv(modelLSTM,trainInp1,trainInp2,trainInp1r,trainInp2r,prm_list)
  print(" testing test data ..\n")
  prm_list["test_score"] = test_model_cv(modelLSTM,testInp1,testInp2,testInp1r,testInp2r,prm_list)
  print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
  score_logger:add{["lr"] = prm_list["lr"],['train score'] = prm_list["train_score"],['test score'] = prm_list["test_score"] }
  score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
  score_logger:plot()  
  model_save(modelLSTM,prm_list,optStatePri,optStateSec)

end

print("end")
