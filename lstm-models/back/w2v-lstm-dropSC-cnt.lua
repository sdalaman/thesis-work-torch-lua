require 'data-lstm-drop-cnt'
require 'model-lstm-drop'
require 'fileListTr'
require 'fileListTst'
require 'fileListTstSmp'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'

cutorch.setHeapTracking(true)

path = "/home/saban/work/additive/"
data_path = path.."data/"
model_save_path = path.."model-impl/lstm-models/model-files/"
model_load_path = path.."model-impl/lstm-models/model-files/"
corpus_path = path.."models/corpus/"

morph = false

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

function implode(sep,tbl,names)
  local newstr
  newstr = ""
  for i=1,#names do
    newstr = newstr ..names[i].."="..tbl[names[i]] .. sep
  end
  return newstr
end

function approxUnknownWord(inpTable,unkList,wnd_size)
--  for i=1,#unkList do
--    if unkList[i][1] <= prm_list["test_size"] then
--      c = unkList[i][1]  -- row -> col  
--      r = unkList[i][2]  -- col -> row
--      inpTable[r][c] = 0
--    end
--  end
  if wnd_size == 0 then
    return inpTable
  end 
  
  for i=1,#unkList do
    if unkList[i][1] <= prm_list["test_size"] then
      c = unkList[i][1]  -- row -> col  
      r = unkList[i][2]  -- col -> row
      bpos = r - wnd_size
      epos = r + wnd_size
      if bpos < 1 then
        bpos = 1
      end
      if epos > #inpTable then
        epos = #inpTable
      end 
      inpTable[r][c] = 0
      cnt=0
      for j = bpos,epos do
      --for j = 1,#inpTable do
        inpTable[r][c] = inpTable[r][c] + inpTable[j][c]
        cnt = cnt + 1
      end
      --inpTable[r][c] = inpTable[r][c] / (#inpTable-1)
      cnt = cnt + 1
      inpTable[r][c] = inpTable[r][c] / cnt
    end
  end
  return inpTable
end

function test_model(model,mPri,mSec,ltPri,ltSec,epoch,inpPri,inpSec,test_size,dType,tst_wnd_size)
  --print("..data type.."..dType.."..  test window size .."..tst_wnd_size.."..\n")
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
  
  local inputPriTT = ltPri:forward( inputPri)
  local inputSecTT = ltSec:forward( inputSec)
  local inputPriT = inputPriTT
  local inputSecT = inputSecTT

  if dType == 'test' then
    inputPriT = approxUnknownWord(inputPriTT,testUnk1,tst_wnd_size)
    inputSecT = approxUnknownWord(inputSecTT,testUnk2,tst_wnd_size)
  end
  
  local all_rootsPri = mPri:forward( inputPriT)
  local all_rootsSec = mSec:forward( inputSecT)
  
  inPri , inSec = null
  inputPri , inputSec = null
  inputPriT , inputSecT , inputPriTT , inputSecTT = null
  
  --local all_rootsPri = outputPri
  --local all_rootsSec = outputSec

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

function test_model_cv(model,insPri,insSec,prm_list,dType,t_window_size)
  totscore = 0
  model:modelPri():clearState()
  model:modelSec():clearState()
  local mPri = model:modelPri():clone()
  local mSec = model:modelSec():clone()
  mPri:evaluate()
  mSec:evaluate()
  mPri:get(1):get(1):get(1):remove(1)
  mSec:get(1):get(1):get(1):remove(1)
  local ltPri = nn.Sequencer(model:getLookupTablePri())
  local ltSec = nn.Sequencer(model:getLookupTableSec())
  --model.mdlPriSC:evaluate()
  --model.mdlSecSC:evaluate()
  --print("..data type.."..dType.."..  test window size .."..t_window_size.."..\n")
  for i = 1,prm_list["max_cv"] do
    score,nsum = test_model(model,mPri,mSec,ltPri,ltSec,prm_list["epoch"],insPri,insSec,prm_list["test_size"],dType,t_window_size)
    totscore = totscore + score
    print("-- score : "..score.."\n")
  end
  --print("-------"..prm_list["max_cv"].." Epochs avg: "..totscore/prm_list["max_cv"].."\n")
  --model.mdlPriSC:training()
  --model.mdlSecSC:training()
  return totscore/prm_list["max_cv"]
end

function prepare_testdata(corpus1,corpus2,prm_list)
  totalLines = 0
  totscore = 0
  totaltestInp1 = nil
  totaltestInp2 = nil
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
      testinputs1= testcorpus_shapedataLSTM(corpus1,testcorpus1):cuda()
      testinputs2= testcorpus_shapedataLSTM(corpus2,testcorpus2):cuda()

      --print(testcorpusfn1.." "..testcorpus1.no_of_words.." words, "..#testcorpus1.sequences.." sents, "..testcorpus1.longest_seq.." sent max lenght\n")
      --print(testcorpusfn2.." "..testcorpus2.no_of_words.." words, "..#testcorpus2.sequences.." sents, "..testcorpus2.longest_seq.." sent max lenght\n")
      if totalTestInp1 ~= nil then
        totalTestInp1 = torch.cat(totalTestInp1,testinputs1[{{},{1,prm_list["max_seq_len"]}}],1)
        totalTestInp2 = torch.cat(totalTestInp2,testinputs2[{{},{1,prm_list["max_seq_len"]}}],1)
      else
        totalTestInp1 = testinputs1[{{},{1,prm_list["max_seq_len"]}}]
        totalTestInp2 = testinputs2[{{},{1,prm_list["max_seq_len"]}}]
      end
    end
  end
  unk1 = {}
  unk2 = {}
  k = 1
  for i=1,totalTestInp1:size()[1] do
    for j=1,totalTestInp1:size()[2] do
      if totalTestInp1[i][j] == 1 then 
        unk1[k] = {i,j}
        k=k+1
      end
    end
  end
  k = 1
  for i=1,totalTestInp2:size()[1] do
    for j=1,totalTestInp2:size()[2] do
      if totalTestInp2[i][j] == 1 then 
        unk2[k] = {i,j}
        k=k+1
      end
    end
  end
  testcorpus1 , testcorpus2 = null
  return totalTestInp1,totalTestInp2,unk1,unk2
end
        
function model_train(model,trInpPri,trInpSec,tstInpPri,tstInpSec,prm_list,tst_wnd_list)
  lr_list = prm_list["lr_list"]
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
  
  local optimStatePriSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"]}
  local optimStateSecSgd = {learningRate = prm_list["lr"],momentum=prm_list["momentum"],learningRateDecay =prm_list["lr_decay"]}
  
  local optimStatePriRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"]}
  local optimStateSecRmsProp = {learningRate = prm_list["lr"], alpha = prm_list["alpha"]}
  
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
          totalGradWeightPri = totalGradWeightPri + torch.sum(torch.pow(dl_dxPri2[1],2))
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
          totalGradWeightSec = totalGradWeightSec + torch.sum(torch.pow(dl_dxSec2[1],2))
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
    loss_logger:add{['lrPri'] = statePri.learningRate,['lrSec'] = stateSec.learningRate,['training error pri'] = current_lossPri ,['training error sec'] = current_lossSec}
    loss_logger:style{['lr']='+-',['training error pri'] = '+-' , ['training error sec'] = '+-' }
    grad_logger:add{['grad weight pri'] = totalGradWeightPri ,['grad weight sec'] = totalGradWeightSec }
    grad_logger:style{['grad weight pri'] = '+-' , ['grad weight sec'] = '+-' }
    
    if i % prm_list["dump_frq"] == 0 then
      print(" testing train data ..\n")
      prm_list["train_score"] = test_model_cv(modelLSTM,trInpPri,trInpSec,prm_list,'train',0)
      print(" testing test data ..\n")
      for t = 1,#tst_wnd_list do
        test_window_size = tst_wnd_list[t]
        print(" test window size .."..test_window_size.."\n")
        prm_list["test_score"] = test_model_cv(modelLSTM,tstInpPri,tstInpSec,prm_list,'test',test_window_size)
        print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
        score_logger:add{["lrPri"] = prm_list["prev_lr"],["lrSec"] = prm_list["prev_lr"], ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
        score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['train score'] = '+-',['test score'] = '+-'}
        score_logger:plot()  
        loss_logger:plot()  
        grad_logger:plot()  
      end
      model_save(model,prm_list)
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

if morph == false then
  corpus_name_pri = "english.1000.tok"
  corpus_name_sec = "turkish.1000.tok"
  fhead = "lstm-SC-CNT-avg-drop-1000"
else
  corpus_name_pri = "english.1000.tok.morph"
  corpus_name_sec = "turkish.1000.tok.morph"
  fhead = "lstm-SC-CNT-avg-drop-1000-morph"
end

corpus_load = true
model_save_flag = true
model_load_flag = false
max_sentence_len = 60
test_window_size = 2

corpus1_str = string.sub(corpus_name_pri,1,2)
corpus2_str = string.sub(corpus_name_sec,1,2)
nm = corpus1_str.."-"..corpus2_str

vocab1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".lstm.CNT.vocab" -- vocab file  for lang 1
vocab2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".lstm.CNT.vocab" -- vocab file  for lang 2
corpus_name1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".lstm.corpus.CNT.tch"
corpus_name2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".lstm.corpus.CNT.tch"
inputs_name1 = corpus_path..corpus_name_pri.."."..nm.."."..max_sentence_len..".lstm.inputs.CNT.tch"
inputs_name2 = corpus_path..corpus_name_sec.."."..nm.."."..max_sentence_len..".lstm.inputs.CNT.tch"

prm_names = {"lr","prev_lr","emb_size","hidden_size","lr_decay","threshold","dump_frq","max_epoch","epoch","batch_size","init","max_seq_len","win_size","opm_name","opm","score","num_of_samples"}

scenarios = {}

lr_list_1000 = {}

--for 1000  new  1 ***
scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 128;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.1000.tok";testcorpus2 = "turkish.1000.tok";test_size = 100;out_file = "test-lstm-1000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}

--for 10000  1 ****
--scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 512;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}

--for 10000 morph 1 ***
--scenarios[#scenarios+1] = { lr = 0.001;prev_lr = 0;alpha = 0.95;lr_decay = 1; lr_list = nil ;emb_size = 64;hidden_size = 512;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 2000;epoch=0;batch_size = 100;init = 1;max_seq_len = max_sentence_len;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok.morph";testcorpus2 = "turkish.10000.tok.morph";test_size = 100;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0}


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
  str = string.format("%s-%s-%d-lr=%3.7f-lr_decay=%3.7f-max_epoch=%d-batch_size=%d-init=%1.4f-max_seq_len=%d-win_size=%d" ,fhead,prm_list["opm_name"],prm_list["hidden_size"],prm_list["lr"],prm_list["lr_decay"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])
  return str
end

function model_save(model,prm_list)
  if model_save_flag == true then
    mname = model_save_path..nm..'_'..corpus2_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..fhead
    model:modelSec():clearState()
    torch.save(mname..'.model',model:modelSec())
    torch.save(mname..'.LT',model:getLookupTableSec())
    print("model saved "..mname.."\n")  
    mname = model_save_path..nm..'_'..corpus1_str..'_'..prm_list["lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..fhead
    model:modelPri():clearState()
    torch.save(mname..'.model',model:modelPri())
    torch.save(mname..'.LT',model:getLookupTablePri())
    print("model saved "..mname.."\n")  
  end
end

function model_load(model,prm_list)
  mname = model_load_path..nm..'_'..corpus2_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..fhead
  print("model to be loaded : "..mname.."\n")  
  model.mdlSecSc = torch.load(mname..'.model')
  model.ltSecSc = torch.load(mname..'.LT')
  mname = model_load_path..nm..'_'..corpus1_str..'_'..prm_list["prev_lr"]..'_'..prm_list["epoch"]..'.lstm.'..prm_list["hidden_size"]..'.'..prm_list["max_seq_len"].."."..fhead
  print("model to be loaded : "..mname.."\n")  
  model.mdlPriSc = torch.load(mname..'.model')
  model.ltPriSc = torch.load(mname..'.LT')
  print("model loaded \n")  
end

for i=1,#scenarios do
  prm_list = scenarios[i]
  printf("lr %f, lr_decay %f,emb_size %d,hidden_size %d,max_epoch %d, batch_size %d, init %f, max_seq_len %d, win_size %d \n",prm_list["lr"],prm_list["lr_decay"],
    prm_list["emb_size"],prm_list["hidden_size"],prm_list["max_epoch"],prm_list["batch_size"],prm_list["init"],prm_list["max_seq_len"],prm_list["win_size"])

  seq_len1 = prm_list["max_seq_len"]
  seq_len2 = prm_list["max_seq_len"]

  loss_logger = optim.Logger(fhead..'_loss_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
  grad_logger = optim.Logger(fhead..'_grad_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
  score_logger = optim.Logger(fhead..'_score_log_'..i..'-'..prm_list["lr"]..'-'..prm_list["hidden_size"]..'-'..prm_list["max_seq_len"],true,prepheadstr(prm_list))
 
  testInp1, testInp2, testUnk1,testUnk2 = prepare_testdata(corpus1,corpus2,prm_list)

  modelLSTM = BiLangModelLSTMScAvg(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
  tst_wnd_list = {0,2}
  print(" model created ..\n")
  
  if model_load_flag == true then
    model_load(modelLSTM,prm_list)
    print(" testing train data ..\n")
    prm_list["train_score"] = test_model_cv(modelLSTM,trainInp1,trainInp2,prm_list,'train',0)
    print(" testing test data ..\n")
    for t = 1,#tst_wnd_list do
      test_window_size = tst_wnd_list[t]
      print(" test window size .."..test_window_size.."\n")
      prm_list["test_score"] = test_model_cv(modelLSTM,testInp1,testInp2,prm_list,'test',test_window_size)
      print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
      score_logger:add{["lrPri"] = prm_list["prev_lr"],["lrSec"] = prm_list["prev_lr"], ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
      score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['train score'] = '+-',['test score'] = '+-'}
      score_logger:plot()  
    end
  end
  
  print(" training started ..\n")
  modelLSTM = model_train(modelLSTM,trainInp1,trainInp2,testInp1,testInp2,prm_list,tst_wnd_list)
  print(" testing train data ..\n")
  prm_list["train_score"] = test_model_cv(modelLSTM,trainInp1,trainInp2,prm_list,'train',0)
  print(" testing test data ..\n")
  for t = 1,#tst_wnd_list do
      test_window_size = tst_wnd_list[t]
      print(" test window size .."..test_window_size.."\n")
      prm_list["test_score"] = test_model_cv(modelLSTM,testInp1,testInp2,prm_list,'test',test_window_size)
      print("Train Data Score : "..prm_list["train_score"].." - Test Data Score : "..prm_list["test_score"].."\n")
      score_logger:add{["lrPri"] = prm_list["prev_lr"],["lrSec"] = prm_list["prev_lr"], ['train score'] = prm_list["train_score"], ['test score'] = prm_list["test_score"]  }
      score_logger:style{["lrPri"] = '+-',["lrSec"] = '+-',['train score'] = '+-',['test score'] = '+-'}
      score_logger:plot()  
  end
  model_save(modelLSTM,prm_list)

end

print("end")
