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

local stringx = require('pl.stringx')
local file = require('pl.file')


path = "/home/saban/work/additive/"
data_path = path.."data/"
model_save_path = path.."model-impl/lstm-models/model-files/multi/"
model_load_path = path.."model-impl/lstm-models/model-files/multi/"
corpus_path = path.."models/corpus/multi/"
data_path = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/corpusfiles/"

--10000 tok
modelEnFile10000Tok = 
"/home/saban/work/additive/model-impl/lstm-models/model-files/10000/En/6-Sc-Avg-Tok-Morph/en-tu_en_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000.model"
vocabEnFile10000Tok =  "/home/saban/work/additive/models/corpus/english.10000.tok.en-tu.60.lstm.vocab"
lookupEnFile10000Tok = "/home/saban/work/additive/model-impl/lstm-models/model-files/10000/En/6-Sc-Avg-Tok-Morph/en-tu_en_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000.LT"

--10000 morph
modelEnFile10000TokMorph = 
"/home/saban/work/additive/model-impl/lstm-models/model-files/10000/En/6-Sc-Avg-Tok-Morph/en-tu_en_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000-morph.model"
vocabEnFile10000TokMorph =  "/home/saban/work/additive/models/corpus/english.10000.tok.morph.en-tu.60.lstm.vocab"
lookupEnFile10000TokMorph = "/home/saban/work/additive/model-impl/lstm-models/model-files/10000/En/6-Sc-Avg-Tok-Morph/en-tu_en_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000-morph.LT"

-- all tok
modelEnFileAllTok = 
"/home/saban/work/additive/model-impl/lstm-models/model-files/En/all/6-Sc-Avg-Tok/en-tu_en_0.001_250.lstm.128.128.60.lstm-ScAvg-drop-all.model"
lookupEnFileAllTok = 
"/home/saban/work/additive/model-impl/lstm-models/model-files/En/all/6-Sc-Avg-Tok/en-tu_en_0.001_250.lstm.128.128.60.lstm-ScAvg-drop-all.LT"
vocabEnFileAllTok =  "/home/saban/work/additive/models/corpus/english.all.tok.en-tu.60.lstm.vocab"

-- all morph
modelEnFileAllTokMorph = 
"/home/saban/work/additive/model-impl/lstm-models/model-files/En/all/6-Sc-Avg-Tok/en-tu_en_0.001_250.lstm.128.128.60.lstm-ScAvg-drop-all-morph.model"
lookupEnFileAllTokMorph = 
"/home/saban/work/additive/model-impl/lstm-models/model-files/En/all/6-Sc-Avg-Tok/en-tu_en_0.001_250.lstm.128.128.60.lstm-ScAvg-drop-all-morph.LT"
vocabEnFileAllTokMorph =  "/home/saban/work/additive/models/corpus/english.all.tok.morph.en-tu.60.lstm.vocab"


cutorch.setHeapTracking(true)
collectgarbage()

-- -corpus1 train.de-en.en.10000 -corpus2 train.de-en.de.10000 -dtype tok -corpus_load 0 -model_save 1 -model_load 0 -max_sntc_len 60 
config = {}
config.dtype = "tok" -- data type
config.corpus_load = 1
config.model_save = 1
config.model_load = 0
config.max_sntc_len = 60
config.corpus1 = ""
config.corpus2 = ""

cmd = torch.CmdLine()
cmd:option("-dtype", config.dtype)
cmd:option("-corpus_load", config.corpus_load)
cmd:option("-model_save", config.model_save)
cmd:option("-model_load", config.model_load)
cmd:option("-max_sntc_len", config.max_sntc_len)
cmd:option("-corpus1", config.corpus1)
cmd:option("-corpus2", config.corpus2)

params = cmd:parse(arg)
for param, value in pairs(params) do
    config[param] = value
end

if config.dtype == "tok" then
  modelEnFile = modelEnFile10000Tok
  lookupEnFile =  lookupEnFile10000Tok
  vocabEnFile =  vocabEnFile10000Tok
end

if config.dtype == "tok.morph" then
  modelEnFile = modelEnFile10000TokMorph
  lookupEnFile =  lookupEnFile10000TokMorph
  vocabEnFile =  vocabEnFile10000TokMorph
end

print("data type "..config.dtype)
print("corpus_load "..config.corpus_load)
print("model_save "..config.model_save)
print("model_load "..config.model_load)
print("max_stnc_len "..config.max_sntc_len)

print("model file : "..modelEnFile.."\n")
print("lookup file : "..lookupEnFile.."\n")
print("vocab file : "..vocabEnFile.."\n")

if false then
  scenariosT={}
  scenariosT[1] = { lr = 0.01;prev_lr = 0;alpha = 0.95;lr_decay = 1e-4;weight_decay = 0; lr_list = nil ;emb_size = 128;hidden_size = 128;num_of_hidden=2;momentum=0.5; threshold = 50 ;dump_frq = 50;plot_frq = 10;max_epoch = 1000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0;fhead="";dropoutProb=0.5}

  prm_list = scenariosT[1]
  modelT = BiLangModelLSTMScAvg(ltEnT.weight:size()[1]-1,10,config.max_sntc_len,config.max_sntc_len,prm_list)  
  modelEn = modelT:modelPri()
  modelEn:cuda()
  ltEnT = torch.load(lookupEnFile)
  ltEnT:cuda()
  ltEn = modelT:getLookupTablePri()
  ltEn.weight:copy(ltEnT.weight)

else

  modelEn = torch.load(modelEnFile):double()
  modelEn:clearState()  -- Clear intermediate module states 
  modelEn:cuda()
  ltEn = torch.load(lookupEnFile)
  ltEn:cuda()

end

--config.emb_size = ltEn:parameters()[1]:size()[2]
vocabEn = torch.load(vocabEnFile)
--vocab_size1 = ltEn:parameters()[1]:size()[1]

corpus_load = config.corpus_load == 1
model_save_flag = config.model_save == 1
model_load_flag = config.model_load == 1
max_sentence_len = config.max_sntc_len
morph = config.dtype == "tok.morph"

if morph == false then
  test_data_pathEn = path.."data/cdlc_en_tr/test/englishTok"
  test_data_pathTr = path.."data/cdlc_en_tr/test/turkishTok"
  train_data_pathEn = path.."data/cdlc_en_tr/train/englishTok"
  train_data_pathTr = path.."data/cdlc_en_tr/train/turkishTok"
  
  train_data_pathEnDe = "/home/saban/work/python/pytorch-works/additive/data/ted/en-de-tok/train"
  train_data_pathDeEn = "/home/saban/work/python/pytorch-works/additive/data/ted/de-en-tok/train"
  test_data_pathEnDe = "/home/saban/work/python/pytorch-works/additive/data/ted/en-de-tok/test"
  test_data_pathDeEn = "/home/saban/work/python/pytorch-works/additive/data/ted/de-en-tok/test"
  
  train_data_pathEnFr = "/home/saban/work/python/pytorch-works/additive/data/ted/en-fr-tok/train"
  train_data_pathFrEn = "/home/saban/work/python/pytorch-works/additive/data/ted/fr-en-tok/train"
  test_data_pathEnFr = "/home/saban/work/python/pytorch-works/additive/data/ted/en-fr-tok/test"
  test_data_pathFrEn = "/home/saban/work/python/pytorch-works/additive/data/ted/fr-en-tok/test"
else
  test_data_pathEn = path.."data/cdlc_en_tr/test/englishMorph"
  test_data_pathTr = path.."data/cdlc_en_tr/test/turkishMorph"
  train_data_pathEn = path.."data/cdlc_en_tr/train/englishMorph"
  train_data_pathTr = path.."data/cdlc_en_tr/train/turkishMorph"
  
  train_data_pathEnDe = "/home/saban/work/python/pytorch-works/additive/data/ted/en-de-morph/train"
  train_data_pathDeEn = "/home/saban/work/python/pytorch-works/additive/data/ted/de-en-morph/train"
  test_data_pathEnDe = "/home/saban/work/python/pytorch-works/additive/data/ted/en-de-morph/test"
  test_data_pathDeEn = "/home/saban/work/python/pytorch-works/additive/data/ted/de-en-morph/test"

  train_data_pathEnFr = "/home/saban/work/python/pytorch-works/additive/data/ted/en-fr-morph/train"
  train_data_pathFrEn = "/home/saban/work/python/pytorch-works/additive/data/ted/fr-en-morph/train"
  test_data_pathEnFr = "/home/saban/work/python/pytorch-works/additive/data/ted/en-fr-morph/test"
  test_data_pathFrEn = "/home/saban/work/python/pytorch-works/additive/data/ted/fr-en-morph/test"
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


function test_model_train(mPri,mSec,inpPri,inpSec,test_size)
  local split = nn.SplitTable(2)
  local inPri = torch.zeros(test_size,inpPri:size()[2]):cuda()
  local inSec = torch.zeros(test_size,inpSec:size()[2]):cuda()
  cl = inpPri:size()[1]
  math.randomseed( os.time() )
  
  for i=1,test_size do
    ln = math.random(1,cl)
    inPri[i]:copy(inpPri[ln])
    inSec[i] = inpSec[ln]
  end
  
  local inputSec = split:forward(inSec)
  
  local all_rootsPri = inPri
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

function test_model_test(mPri,mSec,inpPri,inpSec,test_size)
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

function test_model_cv(modelRef,model,insPri,insSec,prm_list,dtype)
  totscore = 0
  --model:modelPri():clearState()
  --model:modelSec():clearState()
  local mPri = modelRef
  local mSec = model:modelSec()
  mPri:evaluate()
  mSec:evaluate()
  for i = 1,prm_list["max_cv"] do
    if dtype == "Train" then
      score,nsum = test_model_train(mPri,mSec,insPri,insSec,prm_list["test_size"])
    else
      score,nsum = test_model_test(mPri,mSec,insPri,insSec,prm_list["test_size"])
    end
    totscore = totscore + score
    print("-- score : "..score.."\n")
  end
  mPri:training()
  mSec:training()
  return totscore/prm_list["max_cv"]
end

function test_err(modelRef,model,inpPri,inpSec,test_size) 
  local criterion = model:getCriterion()
  local split = nn.SplitTable(2)
  mPri = modelRef
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

function prepare_testdata(vocab1,vocab2,prm_list)
  totalLines = 0
  totscore = 0
  
  --test_data_path1 = test_data_pathEn
  --test_data_path2 = test_data_pathTr
  
  --test_data_path1 = test_data_pathEnDe
  --test_data_path2 = test_data_pathDeEn

  test_data_path1 = test_data_pathEnFr
  test_data_path2 = test_data_pathFrEn

  local totaltestInp1 = nil
  local totaltestInp2 = nil
  --fileList = fileListTstSmpEnDe
  fileList = fileListTstSmpEnFr
  notfound = 0
  for fcnt = 1,#fileList do
    if morph == true then
      testcorpusfn1 = fileList[fcnt]..".tok.morph"
      testcorpusfn2 = fileList[fcnt]..".tok.morph"
    else 
      testcorpusfn1 = fileList[fcnt]..".tok"
      testcorpusfn2 = fileList[fcnt]..".tok"
    end
    sellist = {}
    f1,f1err = io.open(test_data_path1..testcorpusfn1,'r') 
    f2,f2err = io.open(test_data_path2..testcorpusfn2,'r')
    if f1 and f2 then
      f1:close()
      f2:close()
      --print("test corpus data will be prepared\n")
      testcorpus1 = Corpus(testcorpusfn1,test_data_path1)
      testcorpus2 = Corpus(testcorpusfn2,test_data_path2)
      testcorpus1:prepare_corpus(nil,prm_list["test_size"])
      testcorpus2:prepare_corpus(nil,prm_list["test_size"])
      testcorpus1.longest_seq = prm_list["max_seq_len"]
      testcorpus2.longest_seq = prm_list["max_seq_len"]
      testinputs1= testcorpus_shapedataLSTM(vocab1,testcorpus1):cuda()
      testinputs2= testcorpus_shapedataLSTM(vocab2,testcorpus2):cuda()
      if totalTestInp1 ~= nil then
        totalTestInp1 = torch.cat(totalTestInp1,testinputs1[{{},{1,prm_list["max_seq_len"]}}],1)
        totalTestInp2 = torch.cat(totalTestInp2,testinputs2[{{},{1,prm_list["max_seq_len"]}}],1)
      else
        totalTestInp1 = testinputs1[{{},{1,prm_list["max_seq_len"]}}]
        totalTestInp2 = testinputs2[{{},{1,prm_list["max_seq_len"]}}]
      end
    else
      print(testcorpusfn1.." not found \n")
      notfound = notfound + 1
    end
  end
  print(notfound .." files not found \n")
  return totalTestInp1,totalTestInp2
end
   
function model_train(modelRef,model,trInpPri,trInpSec,tstInpPri,tstInpSec,prm_list,stSec)
  collectgarbage()
  criterion = model:getCriterion()
  modelSec = model:modelSec()
  modelSec:training()
  beginning_time = torch.tic()
  local split = nn.SplitTable(2)
  local paramsSec, gradParamsSec = modelSec:getParameters()
  local xSec, dl_dxSec = modelSec:parameters()
  
  local optimStateSecSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"],weightDecay =prm_list["weight_decay"],nesterov=true,dampening=0}
  
  local optimStateSecRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"],weightDecay =prm_list["weight_decay"]}
  
  if prm_list["epoch"] == 0 then
    if prm_list["opm_name"] == "sgd" then
      print("Optimization is SGD \n")
      stateSec = optimStateSecSgd
    end
  
    if prm_list["opm_name"] == "rmsprop" then
      print("Optimization is RmsProp \n")
      stateSec = optimStateSecRmsProp
    end
  else
    stateSec = stSec
    stateSec.learningRate = prm_list["lr"]
  end

  no_of_sents = trInpPri:size(1)
  for i =prm_list["epoch"]+1,prm_list["max_epoch"] do
    current_lossSec = 0
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
        
        local inputSec = split:forward(trInpSec[{{start,endd},{1,prm_list["max_seq_len"]}}])
                
        modelSec:zeroGradParameters()
        modelSec:forget()

        function fevalSec(params)
          local outputPri = trInpPri[{{start,endd},{}}]
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
        
        _,fsSec = prm_list["opm"](fevalSec, paramsSec, stateSec)
        current_lossSec = current_lossSec + fsSec[1] 
    end
    
    prm_list["epoch"] = i
    current_lossSec = current_lossSec / math.floor(no_of_sents/prm_list["batch_size"])
    print("epoch "..i..", lr ".. stateSec.learningRate..", loss (sec) = "..current_lossSec..", grad (sec) = "..totalGradWeightSec)
    durationSeconds = torch.toc(beginning_time)
    print('end time : '..os.date("%HH-%MM-%SS")..'  time elapsed(sec): '.. printtime(durationSeconds))
    tst_err = test_err(modelRef,model,tstInpPri,tstInpSec,prm_list["err_test_size"])
    loss_logger:add{['training error sec'] = current_lossSec,['test err'] = tst_err}
    loss_logger:style{['training error sec'] = '+-',['test err'] ='+-' }
    grad_logger:add{['grad weight sec'] = totalGradWeightSec }
    grad_logger:style{['grad weight sec'] = '+-' }
    
    if i % prm_list["plot_frq"] == 0  then
      loss_logger:plot()
      grad_logger:plot()
    end
    
    if prm_list["opm_name"] == "rmsprop" then
      --local clrPri = statePri.learningRate / (1 + i*prm_list["lr_decay"]) 
      --local clrSec = stateSec.learningRate / (1 + i*prm_list["lr_decay"]) 
      --statePri.learningRate = clrPri
      --stateSec.learningRate =  clrSec
    end
    
    if i % prm_list["dump_frq"] == 0 then
      print(" testing train data ..\n")
      prm_list["train_score"] = test_model_cv(modelRef,model,trInpPri,trInpSec,prm_list,"Train")
      print(" testing test data ..\n")
      prm_list["test_score"] = test_model_cv(modelRef,model,tstInpPri,tstInpSec,prm_list,"Test")
      print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
      score_logger:add{["lr"] = stateSec.learningRate, ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
      score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
      score_logger:plot()  
      loss_logger:plot()  
      grad_logger:plot()  
      model_save(model,prm_list,stateSec)
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

corpus1_str = string.sub(config.corpus1,7)
corpus2_str = string.sub(config.corpus2,7)

--corpus_name1 = corpus_path..corpus1_str.."."..max_sentence_len..".lstm.corpus.tch"
--inputs_name1 = corpus_path..corpus1_str.."."..max_sentence_len..".lstm.inputs.tch"

if morph == true then
  vocab2 = corpus_path..corpus2_str..".morph."..max_sentence_len..".lstm.vocab" -- vocab file  for lang 2
  corpus_name2 = corpus_path..corpus2_str..".morph."..max_sentence_len..".lstm.corpus.tch"
  inputs_name2 = corpus_path..corpus2_str..".morph."..max_sentence_len..".lstm.inputs.tch"
else
  vocab2 = corpus_path..corpus2_str.."."..max_sentence_len..".lstm.vocab" -- vocab file  for lang 2
  corpus_name2 = corpus_path..corpus2_str.."."..max_sentence_len..".lstm.corpus.tch"
  inputs_name2 = corpus_path..corpus2_str.."."..max_sentence_len..".lstm.inputs.tch"
end

prm_names = {"lr","prev_lr","emb_size","hidden_size","lr_decay","threshold","dump_frq","max_epoch","epoch","batch_size","init","max_seq_len","win_size","opm_name","opm","score","num_of_samples"}

scenarios = {}


if config.dtype ~= "tok.morph" then
  scenarios[#scenarios+1] = { lr = 0.01;prev_lr = 0.01;alpha = 0.95;lr_decay = 1e-4;weight_decay = 0; lr_list = nil ;emb_size = 128;hidden_size = 128;num_of_hidden=2;momentum=0.5; threshold = 50 ;dump_frq = 50;plot_frq = 10;max_epoch = 3000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;dropoutProb=0.9;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0;fhead=""}

else

  --for 10000 morph new 1
  scenarios[#scenarios+1] = { lr = 0.01;prev_lr = 0;alpha = 0.95;lr_decay = 1e-4;weight_decay = 0; lr_list = nil ;emb_size = 128;hidden_size = 128;num_of_hidden=2;momentum=0.5; threshold = 50 ;dump_frq = 50;plot_frq = 10;max_epoch = 3000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;dropoutProb=0.9;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok.morph";testcorpus2 = "turkish.10000.tok.morph";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0;fhead=""}

end

function paddingEn(sequence,longest_seq)
  new_sequence = {}
  mlen = math.min(longest_seq,#sequence)
  for i = 1 , mlen do
    new_sequence[i] = sequence[i]
  end
  for i = mlen+1, longest_seq do
    new_sequence[i] = '<pad>'
  end
  return new_sequence
end


function map_dataEn(sentences,longest_seq,vocab)
  local x = torch.zeros(#sentences, longest_seq)
  --local mTable = {}
  local no_of_words = 0
  local unknown = 0
  for i = 1, #sentences do
    local words = stringx.split(sentences[i], '\t') -- tokenized with tab
    no_of_words = no_of_words + #words
    if #words < longest_seq then
      words = paddingEn(words,longest_seq)
    else
      words = table.slice(words, 1, longest_seq, 1)
    end
    for j = 1, #words do 
      if vocab[words[j]] == nil then
          x[i][j] = vocab['_UNK_']
          unknown = unknown + 1
      else
          x[i][j] = vocab[words[j]] 
      end
    end
  end
  print(string.format("Number of words = %d , unknown words = %d", no_of_words,unknown))
  return x
end


function getDataEn(fname,vocab,longest_seq)
  local data = file.read(fname)
  sentencesT = stringx.split(data, '\n')
  sentences = {}
  if sentencesT[#sentencesT] == "" then
    for i=1,#sentencesT-1 do
      table.insert(sentences,sentencesT[i])
    end
  end
  sentencesT = {}
  print(string.format("Loading %s, Number of sentences = %d", fname, #sentences))
  local mapped = map_dataEn(sentences,longest_seq,vocab)
  return mapped
end

function prepareData(model,dataTable,cfg)
  local vectors = torch.zeros(dataTable:size()[1],cfg.emb_size):double():cuda()
  local split = nn.SplitTable(2):cuda()
  noofsents = math.floor(dataTable:size()[1]/cfg.batch_size)*cfg.batch_size
  --print("Epoch size for the first corpus : "..noofsents.."\n")
  for i=1,noofsents/cfg.batch_size do
    b=(i-1)*cfg.batch_size+1
    e=i*cfg.batch_size
    local input = split:forward(dataTable[{{b,e},{}}])
    vectors[{{b,e},{}}] = model:forward(input)
  end
  return vectors
end

config.batch_size = 100
tableEn = getDataEn(data_path..config.corpus1.."."..config.dtype,vocabEn,max_sentence_len)
tableEn = tableEn:cuda()
allEn = prepareData(modelEn,tableEn,config)
ltEn = {}
tableEn = {}

--os.exit()

if corpus_load == false then
  corpus2 = Corpus(config.corpus2.."."..config.dtype,data_path)
  corpus2:prepare_corpus(nil)
  corpus2.longest_seq = max_sentence_len
  trainInp2=corpus2:get_dataLSTM():cuda()

  torch.save(corpus_name2,corpus2)
  torch.save(inputs_name2,trainInp2)
  torch.save(vocab2,corpus2.vocab_map)
  print("corpus data prepared and saved\n")
else
  print("corpus data will be loaded\n")
  corpus2 = torch.load(corpus_name2)
  trainInp2 = torch.load(inputs_name2)
  print("corpus data loaded\n")
end

vocab_size2 = corpus2.no_of_words
seq_len1 = config.max_sntc_len
seq_len2 = config.max_sntc_len

print(corpus_name2.." "..corpus2.no_of_words.." words, "..#corpus2.sequences.." sents, "..corpus2.longest_seq.." sent max lenght\n")
print(inputs_name2.."\n")


function prepheadstr(prm_list)
  str = string.format("%s-%s-%d-%d-lr=%3.7f-lr_decay=%3.7f-w_decay=%3.7f-max_epoch=%d-batch_size=%d-init=%1.4f-max_seq_len=%d-win_size=%d" ,prm_list["fhead"],prm_list["opm_name"],prm_list["emb_size"],prm_list["hidden_size"],prm_list["lr"],prm_list["lr_decay"],prm_list["weight_decay"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])
  return str
end

function model_save(model,prm_list,stSec)
  if model_save_flag == true then
    mname = model_save_path..prm_list["fhead"]..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"]
    model:modelSec():clearState()
    torch.save(mname..'.model',model:modelSec())
    torch.save(mname..'.LT',model:getLookupTableSec())
    torch.save(mname..'.sts',stSec)
    print("model saved "..mname.."\n")  
  end
end

function model_load(model,prm_list)
  mname = model_load_path..prm_list["fhead"]..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["emb_size"]..'.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"]
  print("model to be loaded : "..mname.."\n")  
  model.mdlSec = torch.load(mname..'.model')
  model.ltSec = torch.load(mname..'.LT')
  stsSec = torch.load(mname..'.sts')
  return stsSec
end

for i=1,#scenarios do
  prm_list = scenarios[i]
  printf("lr %f, lr_decay %f,emb_size %d,hidden_size %d,num of hidden %d,max_epoch %d, batch_size %d, init %f, max_seq_len %d, win_size %d \n",prm_list["lr"],prm_list["lr_decay"],
    prm_list["emb_size"],prm_list["hidden_size"],prm_list["num_of_hidden"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])

if morph == false then
    prm_list["fhead"] = "lstm-ScAvg-drop-"..corpus1_str
else
    prm_list["fhead"] = "lstm-ScAvg-drop-"..corpus1_str.."-morph"
end

  seq_len1 = prm_list["max_seq_len"]
  seq_len2 = prm_list["max_seq_len"]

  loss_logger = optim.Logger(prm_list["fhead"]..'_loss_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["emb_size"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
  grad_logger = optim.Logger(prm_list["fhead"]..'_grad_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["emb_size"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
  score_logger = optim.Logger(prm_list["fhead"]..'_score_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["emb_size"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
 
  --testInp1, testInp2 = prepare_testdata(corpus1,corpus2,prm_list)
  
  testInp1, testInp2 = prepare_testdata(vocabEn,corpus2.vocab_map,prm_list)
  
  modelLSTM = BiLangModelLSTMScAvg(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
  
  optStateSec = nil
  
  if model_load_flag == true then
    optStateSec = model_load(modelLSTM,prm_list)
    print(" testing train data ..\n")
    prm_list["train_score"] = test_model_cv(modelEn,modelLSTM,allEn,trainInp2,prm_list,"Train")
    print(" testing test data ..\n")
    prm_list["test_score"] = test_model_cv(modelEn,modelLSTM,testInp1,testInp2,prm_list,"Test")
    print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
    score_logger:add{["lr"] = prm_list["prev_lr"], ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
    score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
    score_logger:plot()  
    print(modelLSTM.mdlSec)
    print(" model loaded ..\n")
  else
    print(modelLSTM.mdlSec)
    print(" model created ..\n")
  end
 
  print("model name : BiLangModelLSTMScAvg")
  
  print(" training started ..\n")
  modelLSTM = model_train(modelEn,modelLSTM,allEn,trainInp2,testInp1,testInp2,prm_list,optStateSec)
  print(" testing train data ..\n")
  prm_list["train_score"] = test_model_cv(modelEn,modelLSTM,trainInp1,trainInp2,prm_list,"Train")
  print(" testing test data ..\n")
  prm_list["test_score"] = test_model_cv(modelEn,modelLSTM,testInp1,testInp2,prm_list,"Test")
  print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
  score_logger:add{["lr"] = prm_list["lr"],['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
  score_logger:style{["lr"] = '+-',['train score'] = '+-',['test score'] = '+-'}
  score_logger:plot()  
  model_save(modelLSTM,prm_list,optStateSec)

end

print("end")
