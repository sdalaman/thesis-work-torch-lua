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

cutorch.setHeapTracking(true)
collectgarbage()

config = {}
config.dtype = "" -- data type
cmd = torch.CmdLine()
cmd:option("-dtype", config.dtype)
params = cmd:parse(arg)
for param, value in pairs(params) do
    config[param] = value
end


path = "/home/saban/work/additive/"
data_path = path.."data/"
model_save_path = path.."model-impl/lstm-models/model-files/"
model_load_path = path.."model-impl/lstm-models/model-files/"
corpus_path = path.."models/corpus/"

print("data type "..config.dtype)
if config.dtype == "morph" then
  morph = true
else
  morph = false
end

dsize = "10000"

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

initGPUTotalMem()

function printGPUMemUsage()
  print('## Mem Used on GPU ##') 
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

function test_model(model,epoch,inpPri,inpSec,inpPriR,inpSecR,test_size)
  mPri = model:modelPri()
  mSec = model:modelSec()

  local split = nn.SplitTable(2)
  inPri = torch.zeros(test_size,inpPri:size()[2]):cuda()
  inSec = torch.zeros(test_size,inpSec:size()[2]):cuda()
  inPriR = torch.zeros(test_size,inpPriR:size()[2]):cuda()
  inSecR = torch.zeros(test_size,inpSecR:size()[2]):cuda()
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
  
  local outputPri = mPri:forward({inputPri,inputPriR})
  local outputSec = mSec:forward({inputSec,inputSecR})
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
  --print("-------Epoch: "..epoch.." - Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  return score,all_rootsPri:size(1)
end

function test_model_cv(model,insPri,insSec,insPriR,insSecR,prm_list)
  totscore = 0
  model.brnnPri:evaluate()
  model.brnnSec:evaluate()
  for i = 1,prm_list["max_cv"] do
    score,nsum = test_model(model,prm_list["epoch"],insPri,insSec,insPriR,insSecR,prm_list["test_size"])
    totscore = totscore + score
    print("-- score : "..score.."\n")
  end
  --print("-------"..prm_list["max_cv"].." Epochs avg: "..totscore/prm_list["max_cv"].."\n")
  model.brnnPri:training()
  model.brnnSec:training()
  return totscore/prm_list["max_cv"]
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
      --testcorpus1r.longest_seq = prm_list["max_seq_len"]
      --testcorpus2r.longest_seq = prm_list["max_seq_len"]
      testinputs1,testinputs1r= testcorpus_shapedataBiLSTM(corpus1,testcorpus1)
      testinputs2,testinputs2r= testcorpus_shapedataBiLSTM(corpus2,testcorpus2)
      
      --print(testcorpusfn1.." "..testcorpus1.no_of_words.." words, "..#testcorpus1.sequences.." sents, "..testcorpus1.longest_seq.." sent max lenght\n")
      --print(testcorpusfn2.." "..testcorpus2.no_of_words.." words, "..#testcorpus2.sequences.." sents, "..testcorpus2.longest_seq.." sent max lenght\n")
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
  return totalTestInp1:cuda(),totalTestInp2:cuda(),totalTestInp1r:cuda(),totalTestInp2r:cuda()
end
        
function model_train(model,trInpPri,trInpSec,trInpPriR,trInpSecR,tstInpPri,tstInpSec,tstInpPriR,tstInpSecR,prm_list,stPri,stSec)
  collectgarbage()
  printGPUMemUsage()
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
  
  local optimStatePriSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"],weightDecay =prm_list["weight_decay"]}
  local optimStateSecSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"],weightDecay =prm_list["weight_decay"]}
  
  local optimStatePriRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"],weightDecay =prm_list["weight_decay"]}
  local optimStateSecRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"],weightDecay =prm_list["weight_decay"]}
  
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

  if prm_list["epoch"] > 0 then
    statePri = stPri
    stateSec = stSec
  end
  
  if statePri == nil or stateSec == nil then
    print("Opt states ERROR \n")
  else
    print("Opt states assigned \n")
    print(statePri)
    print(stateSec)
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
        if((start > trInpPri:size(1)) or (endd > trInpPri:size(1))) then
          k = k + 1
          endd = trInpPri:size(1)
        end
        local inputPri = split:forward(trInpPri[{{start,endd},{}}])
        local inputSec = split:forward(trInpSec[{{start,endd},{}}])
        local inputPriR = split:forward(trInpPriR[{{start,endd},{}}])
        local inputSecR = split:forward(trInpSecR[{{start,endd},{}}])
        
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
          --totalGradWeightPri = totalGradWeightPri + torch.sum(torch.pow(dl_dxPri2[1],2))
          totalGradWeightPri = totalGradWeightPri + torch.sum(torch.pow(gradParamsPri,2))
          return err, gradParamsPri
        end
        
        printGPUMemUsage()
        function fevalSec(params)
          local outputPri = modelPri:forward({inputPri,inputPriR})
          local outputSec = modelSec:forward({inputSec,inputSecR})
          local err = criterion:forward( outputSec, outputPri)
          local gradOutputs = criterion:backward(outputSec, outputPri)
          printGPUMemUsage()
          modelSec:backward({inputSec,inputSecR}, gradOutputs)
          --totalGradWeightSec = totalGradWeightSec + torch.sum(torch.pow(dl_dxSec2[1],2))
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
    loss_logger:add{['lrPri'] = statePri.learningRate,['lrSec'] = stateSec.learningRate,['training error pri'] = current_lossPri ,['training error sec'] = current_lossSec}
    loss_logger:style{['lr']='+-',['training error pri'] = '+-' , ['training error sec'] = '+-' }
    grad_logger:add{['grad weight pri'] = totalGradWeightPri ,['grad weight sec'] = totalGradWeightSec }
    grad_logger:style{['grad weight pri'] = '+-' , ['grad weight sec'] = '+-' }
    
    if i % prm_list["dump_frq"] == 0 then
      print(" testing train data ..\n")
      prm_list["train_score"] = test_model_cv(modelLSTM,trInpPri,trInpSec,trInpPriR,trInpSecR,prm_list)
      print(" testing test data ..\n")
      prm_list["test_score"] = test_model_cv(modelLSTM,tstInpPri,tstInpSec,tstInpPriR,tstInpSecR,prm_list)
      score_logger:add{["lrPri"] = statePri.learningRate,["lrSec"] = stateSec.learningRate, ['train score'] = prm_list["train_score"],['test score'] = prm_list["test_score"] }
      score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['train score'] = '+-',['test score'] = '+-'}
      score_logger:plot()  
      print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
      loss_logger:plot()  
      grad_logger:plot()  
      model_save(model,prm_list,statePri,stateSec)
      print("Opt states \n")
      print(statePri)
      print(stateSec)
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

if morph == false then
  corpus_name_pri = "english."..dsize..".tok"
  corpus_name_sec = "turkish."..dsize..".tok"
  fhead = "BiLstm-avg-drop-"..dsize
else
  corpus_name_pri = "english."..dsize..".tok.morph"
  corpus_name_sec = "turkish."..dsize..".tok.morph"
  fhead = "BiLstm-avg-drop-"..dsize.."-morph"
end

corpus_load = true
model_save_flag = true
model_load_flag = false
max_sentence_len = 60


corpus1_str = string.sub(corpus_name_pri,1,2)
corpus2_str = string.sub(corpus_name_sec,1,2)
nm = corpus1_str.."-"..corpus2_str

vocab1 = corpus_path..corpus_name_pri.."."..nm..".BiLstm.vocab" -- vocab file  for lang 1
vocab2 = corpus_path..corpus_name_sec.."."..nm..".BiLstm.vocab" -- vocab file  for lang 2
corpus_name1 = corpus_path..corpus_name_pri.."."..nm..".BiLstm.corpus.tch"
corpus_name2 = corpus_path..corpus_name_sec.."."..nm..".BiLstm.corpus.tch"
inputs_name1 = corpus_path..corpus_name_pri.."."..nm..".BiLstm.inputs.tch"
inputs_name2 = corpus_path..corpus_name_sec.."."..nm..".BiLstm.inputs.tch"
inputs_name1r = corpus_path..corpus_name_pri.."."..nm..".r.BiLstm.inputs.tch"
inputs_name2r = corpus_path..corpus_name_sec.."."..nm..".r.BiLstm.inputs.tch"

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

prm_names = {"lr","prev_lr","emb_size","hidden_size","lr_decay","threshold","dump_frq","max_epoch","epoch","batch_size","init","max_seq_len","win_size","opm_name","opm","score","num_of_samples"}

scenarios = {}

lr_list_1000 = {}

--for 1000  new  1
--scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 128;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=0;batch_size = 50;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.1000.tok";testcorpus2 = "turkish.1000.tok";test_size = 100;out_file = "test-lstm-1000.txt"; dsize ="1000";max_cv = 10}

--BI for 1000  new  1 ok
--scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 128;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 150;epoch=0;batch_size = 50;init = 1;max_seq_len = 10;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.1000.tok";testcorpus2 = "turkish.1000.tok";test_size = 100;out_file = "test-lstm-1000.txt"; dsize ="1000";max_cv = 10}

-------------------
if config.dtype ~= "morph" then
--for 10000  1 ***
  scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0;alpha = 0.95;lr_decay = 1e-6;weight_decay = 1e-4; lr_list = nil ;emb_size = 128;hidden_size = 512;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 1000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'sgd';opm = optim.sgd; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}
else
  --for 10000  1 morph ***
  scenarios[#scenarios+1] = { lr = 0.01;prev_lr = 0;alpha = 0.95;lr_decay = 1e-4; lr_list = nil ;emb_size = 128;hidden_size = 512;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 1000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'sgd';opm = optim.sgd; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}
end



function prepheadstr(prm_list)
  str = string.format("%s-%s-%d-lr=%3.7f-lr_decay=%3.7f-max_epoch=%d-batch_size=%d-init=%1.4f-max_seq_len=%d-win_size=%d" ,fhead,prm_list["opm_name"],prm_list["hidden_size"],prm_list["lr"],prm_list["lr_decay"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])
  return str
end

function model_save(model,prm_list,stsPri,stsSec)
  if model_save_flag == true then
    mname = model_save_path..nm..'_'..corpus2_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"].."."..fhead
    torch.save(mname..'.model',model:modelSec())
    torch.save(mname..'.LT',model:getLookupTableSec())
    torch.save(mname..'.state',stsPri)
    print("model saved "..mname.."\n")  
    mname = model_save_path..nm..'_'..corpus1_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"].."."..fhead
    torch.save(mname..'.model',model:modelPri())
    torch.save(mname..'.LT',model:getLookupTablePri())
    torch.save(mname..'.state',stsSec)
    print("model saved "..mname.."\n")  
  end
end

function model_load(model,prm_list)
  mname = model_load_path..nm..'_'..corpus2_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"].."."..fhead
  print("model to be loaded : "..mname.."\n")  
  model.brnnSec = torch.load(mname..'.model')
  model.sharedLookupTableSec = torch.load(mname..'.LT')
  stsPri = torch.load(mname..'.state')
  mname = model_load_path..nm..'_'..corpus1_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"].."."..fhead
  print("model to be loaded : "..mname.."\n")
  model.brnnPri = torch.load(mname..'.model')
  model.sharedLookupTablePri = torch.load(mname..'.LT')
  stsSec = torch.load(mname..'.state')
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

  trainInp1 = trainInp1[{{},{1,prm_list["max_seq_len"]}}]
  trainInp2 = trainInp2[{{},{1,prm_list["max_seq_len"]}}]
  trainInp1r = trainInp1r[{{},{1,prm_list["max_seq_len"]}}]
  trainInp2r = trainInp2r[{{},{1,prm_list["max_seq_len"]}}]

  loss_logger = optim.Logger(fhead..'_loss_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"],true,prepheadstr(prm_list))
  grad_logger = optim.Logger(fhead..'_grad_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"],true,prepheadstr(prm_list))
  score_logger = optim.Logger(fhead..'_score_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"],true,prepheadstr(prm_list))
 
  testInp1, testInp2,testInp1r, testInp2r = prepare_testdata(corpus1,corpus2,prm_list)

  modelLSTM = BiLangModelBiLSTMScAvg(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
  corpus1 = nil
  corpus2 = nil
  statePri = nil
  stateSec = nil
  
  if model_load_flag == true then
    statePri,stateSec = model_load(modelLSTM,prm_list)
    print(" testing train data ..\n")
    prm_list["train_score"] = test_model_cv(modelLSTM,trainInp1,trainInp2,trainInp1r,trainInp2r,prm_list)
    print(" testing test data ..\n")
    prm_list["test_score"] = test_model_cv(modelLSTM,testInp1,testInp2,testInp1r,testInp2r,prm_list)
    print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
    score_logger:add{["lrPri"] = prm_list["prev_lr"],["lrSec"] = prm_list["prev_lr"], ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
    score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['train score'] = '+-',['test score'] = '+-'}
    score_logger:plot()  
  end
  
  modelLSTM = model_train(modelLSTM,trainInp1,trainInp2,trainInp1r,trainInp2r,testInp1,testInp2,testInp1r,testInp2r,prm_list,statePri,stateSec)
  print(" testing train data ..\n")
  prm_list["train_score"] = test_model_cv(modelLSTM,trainInp1,trainInp2,trainInp1r,trainInp2r,prm_list)
  print(" testing test data ..\n")
  prm_list["test_score"] = test_model_cv(modelLSTM,testInp1,testInp2,testInp1r,testInp2r,prm_list)
  print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
  score_logger:add{["lrPri"] = prm_list["lr"],["lrSec"] = prm_list["lr"], ['train score'] = prm_list["train_score"],['test score'] = prm_list["test_score"] }
  score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['train score'] = '+-',['test score'] = '+-'}
  score_logger:plot()  
  model_save(modelLSTM,prm_list,statePri,stateSec)

end

print("end")