math.randomseed(1)

require 'nn'
require 'cunn'
require 'rnn'
--require 'lfs'
require 'utils'
require 'optim'
require 'io'
require 'os'
require 'paths'

model_path = "/home/saban/work/additive/models/"
corpus_path = "/home/saban/work/additive/models/corpus/"

local stringx = require('pl.stringx')
local file = require('pl.file')
--local data_path = "./data/cdlc_en_tr/"

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   return err
end

function padding(sequence,longest_seq)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , longest_seq - #sequence do
    new_sequence[i] = '<pad>'
  end
  j = 1
  for i = (longest_seq - #sequence)+1, longest_seq do
    new_sequence[i] = sequence[j]
    j = j + 1
  end
  return new_sequence
end

function map_data(data,longest_seq,vocab)
  --local vocab = torch.load(vocab_file)
  x = torch.Tensor(#data,longest_seq)
  for idx,item in ipairs(data) do
    all_words = stringx.split(item, sep)
    sample = torch.Tensor(longest_seq)
    all_words = padding(all_words,longest_seq)
    for k,word in ipairs(all_words) do
      if(vocab[word] ~= nil) then
        sample[k] = vocab[word]
      else
        sample[k] = 0
      end
    end
    x[idx] = sample  
  end
  return x
end

function getData(data_path,class,vocab,dtype)
  local pos_data = {}
  local neg_data = {}
  local longest_seq = 0
  local pcnt = 0
  local ncnt = 0
  for f in lfs.dir(data_path..'/'..class..'/positive') do
    local text = file.read(data_path..'/'..class..'/positive/'..f)
    if (text ~= nil) then
      pcnt = pcnt + 1
      no_words = #stringx.split(text,sep)
      --no_words = table.getn(stringx.split(text,sep))
      if longest_seq < no_words then 
        longest_seq = no_words
      end
      table.insert(pos_data,text) 
    end
  end  
  for f in lfs.dir(data_path..'/'..class..'/negative') do
    local text = file.read(data_path..'/'..class..'/negative/'..f)
    if (text ~= nil) then
      ncnt = ncnt + 1
      no_words = #stringx.split(text,sep)
      --no_words = table.getn(stringx.split(text,sep))
      if longest_seq < no_words then 
        longest_seq = no_words
      end
      table.insert(neg_data,text)
      if pcnt == ncnt and dtype == "train" then
        break
      end
    end
  end
  local pos_mapped = map_data(pos_data,longest_seq,vocab)
  local neg_mapped = map_data(neg_data,longest_seq,vocab)
  return pos_mapped, neg_mapped,pcnt,ncnt
end

function padding2(sequence,longest_seq)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , #sequence do
    new_sequence[i] = sequence[i]
  end
  j = 1
  for i = #sequence+1, longest_seq do
    new_sequence[i] = '<pad>'
    j = j + 1
  end
  return new_sequence
end


function map_data2(data,longest_seq,vocab)
  docstable = {}
  for idx,item in ipairs(data) do
    lines = stringx.split(item, '\n')
    doctensor = torch.Tensor(#lines,longest_seq)
    for lcnt = 1,#lines do
      line_words = stringx.split(lines[lcnt], sep)
      sample = torch.Tensor(longest_seq)
      line_words = padding2(line_words,longest_seq)
      for k,word in ipairs(line_words) do
        if(vocab[word] ~= nil) then
          sample[k] = vocab[word]
        else
          sample[k] = 0
        end
      end
      doctensor[lcnt] = sample  
    end
    table.insert(docstable,doctensor)
  end
  return docstable
end


function getData2(data_path,class,vocab,dtype)
  local pos_data = {}
  local neg_data = {}
  local longest_seq = 0
  local pcnt = 0
  local ncnt = 0
  for f in lfs.dir(data_path..'/'..class..'/positive') do
    local text = file.read(data_path..'/'..class..'/positive/'..f)
    if (text ~= nil) then
      lines = stringx.split(text, '\n')
      pcnt = pcnt + 1
      for lcnt = 1,#lines do
        nm_words = #stringx.split(lines[lcnt],sep)
        if longest_seq < nm_words then 
          longest_seq = nm_words
        end
      end
      table.insert(pos_data,text) 
    end
  end  
  for f in lfs.dir(data_path..'/'..class..'/negative') do
    local text = file.read(data_path..'/'..class..'/negative/'..f)
    if (text ~= nil) then
      ncnt = ncnt + 1
      lines = stringx.split(text, '\n')
      for lcnt = 1,#lines do
        nm_words = #stringx.split(lines[lcnt],sep)
        if longest_seq < nm_words then 
          longest_seq = nm_words
        end
      end
      table.insert(neg_data,text)
      if pcnt == ncnt and dtype == "train" then
        break
      end
    end
  end
  local pos_mapped = map_data2(pos_data,longest_seq,vocab)
  local neg_mapped = map_data2(neg_data,longest_seq,vocab)
  return pos_mapped, neg_mapped,pcnt,ncnt
end



sep = '\t'
--local classes = {'art','arts','biology'}
local classes = {'art','arts','biology','business','creativity','culture','design','economics','education','entertainment','global','health','politics','science','technology'}


--inp_size = 64
--max_epoch = 300
--lr = 0.01
--lr_decay = 1
--threshold = 5
--folds = 10
--init = 0.001
--test_percentage = 0.25
--train_percentage = 1


-- Default configuration
config = {}
config.corpus1 = "" -- corpus file  for lang 1
config.corpus2 = "" -- corpus file  for lang 2
config.train_data_path1 = "" -- train files  for lang 1
config.train_data_path2 = "" -- train files  for lang 2
config.test_data_path1 = "" -- test files  for lang 1
config.test_data_path2 = "" -- test files  for lang 2
config.model1 = "" -- model file  for lang 1
config.model2 = "" -- model file  for lang 2
config.vocab1 = "" -- vocab file  for lang 1
config.vocab2 = "" -- vocab file  for lang 2
config.inp_size = 64
config.lr = 0.01
config.lr_decay = 1
config.threshold = 5
config.max_epoch = 300
config.folds = 10
config.init = 0.001
config.test_per = 0.25
config.train_per = 1
config.train_test = 0  -- test with train data
config.test_test = 0   -- test with test data
config.modeltype = "" -- model file  
config.dsize = "" -- data size 
config.batch_size = 10

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus1", config.corpus1)
cmd:option("-corpus2", config.corpus2)
cmd:option("-train_data_path1", config.train_data_path1)
cmd:option("-train_data_path2", config.train_data_path2)
cmd:option("-test_data_path1", config.test_data_path1)
cmd:option("-test_data_path2", config.test_data_path2)
cmd:option("-lr", config.lr)
cmd:option("-lr_decay", config.lr_decay)
cmd:option("-max_epoch", config.max_epoch)
cmd:option("-batch_size", config.batch_size)
cmd:option("-inp_size", config.inp_size)
cmd:option("-threshold", config.threshold)
cmd:option("-folds", config.folds)
cmd:option("-init", config.init)
cmd:option("-test_per", config.test_per)
cmd:option("-train_per", config.train_per)
cmd:option("-train_test", config.train_test)
cmd:option("-test_test", config.test_test)
cmd:option("-modeltype", config.modeltype)
cmd:option("-dsize", config.dsize)
params = cmd:parse(arg)


---corpus1 english.1000.tok -corpus2 turkish.1000.tok -train_data_path1 "../data/cdlc_en_tr/train/englishTok" -train_data_path2    "../data/cdlc_en_tr/train/turkishTok" -test_data_path1 "../data/cdlc_en_tr/test/englishTok" -test_data_path2 "../data/cdlc_en_tr/test/turkishTok" -lr 0.01 -lr_decay 1 -max_epoch 300 -inp_size 64 -threshold 5 -folds 10 -init 0.001 -test_per 1 -train_per 1 -train_test 1 -test_test 1 -modeltype additive -dsize 1000 -batch_size 10

for param, value in pairs(params) do
    config[param] = value
end

model_path = model_path..config.modeltype.."/"..config.dsize.."/"

print("Model type for word vectors..."..config.modeltype)

nm = string.sub(config.corpus1,1,2).."-"..string.sub(config.corpus2,1,2)
modelPriFileName =  model_path..config.corpus1.."."..nm..".model"
modelSecFileName =  model_path..config.corpus2.."."..nm..".model"
vocabPriFileName = corpus_path..config.corpus1.."."..nm..".vocab"
vocabSecFileName = corpus_path..config.corpus2.."."..nm..".vocab"

for i,j in pairs(config) do
    --print(i..": "..j)
end

local modelPri = torch.load(modelPriFileName):double()
local modelSec = torch.load(modelSecFileName):double()
modelPri:clearState()  -- Clear intermediate module states 
modelSec:clearState()  -- Clear intermediate module states 
modelPri:cuda()
modelSec:cuda()


local vocabPri = torch.load(vocabPriFileName)
local vocabSec = torch.load(vocabSecFileName)

trainDataTest = config.train_test == 1
testDataTest = config.test_test == 1

file_timestamp = os.date("%Y_%m_%d_%X")
output_file = io.open("f1scorevslr_"..nm.."-"..config.modeltype.."-"..config.dsize.."-"..file_timestamp.."-newtotal-full.csv", "w")
output_file:write("Program args : \n")
for param, value in pairs(params) do
  output_file:write(param.." : "..value..", ")
end
output_file:write("\n")
output_file:flush()

--if trainDataTest == true then
--  output_file:write("Data Sets for test ( train-test) : "..config.train_data_path1.."/"..config.train_data_path2.."\n")
--else
--  output_file:write("Train Data Set not used for testing \n")
--end

--if testDataTest == true then
--  output_file:write("Test Data Sets (test-test) : "..config.test_data_path1.."/"..config.test_data_path2.."\n\n")
--else
--  output_file:write("Test Data Set not used for testing \n")
--end

if trainDataTest == true then
  f1_score_avgTr = 0
  score_avgTr = 0
  mcc_acc_avgTr = 0
end

if testDataTest == true then
  f1_score_avgTst = 0
  score_avgTst = 0
  mcc_acc_avgTst = 0
end

if trainDataTest == true then
  score_loggerTr = {}
end

if testDataTest == true then
  score_loggerTst = {}
end

lcnt = 0
for _,class in ipairs(classes) do
  lcnt = lcnt + 1
  if trainDataTest == true then
    --score_loggerTr[lcnt] = optim.Logger(class..'-train-score-log-'..nm..file_timestamp..".log",false,"train")
  end
  
  if testDataTest == true then
    --score_loggerTst[lcnt] = optim.Logger(class..'-test-score-log-'..nm..file_timestamp..".log",false,"test")
  end
end

fdata = "full"

lcnt = 0
for _,class in ipairs(classes) do
  print("\n--------------------\n")
  print("\nClass "..class)
  print("..."..os.date("%Y_%m_%d_%X"))
  lcnt = lcnt + 1
  --score_logger[lcnt] = optim.Logger(class..'-score-log-'..nm,true,"aaa")
  
  local positivePriTableT, negativePriTableT,pc2,nc2 = getData2(config.train_data_path1,class,vocabPri,fdata)
  print("positive train example : "..pc2.." - negative train example : "..nc2)
  positivePriTable = {}
  negativePriTable = {}
  for i = 1,math.floor(#positivePriTableT*config.train_per) do
    table.insert(positivePriTable,positivePriTableT[i])
  end
  for i = 1,math.floor(#negativePriTableT*config.train_per) do
    table.insert(negativePriTable,negativePriTableT[i])
  end
  positivePriTableT = {}
  negativePriTableT = {}
  
  local positivePri2 = torch.zeros(#positivePriTable,config.inp_size):double():cuda()
  local negativePri2 = torch.zeros(#negativePriTable,config.inp_size):double():cuda()
  for i = 1,#positivePriTable do
    local t = modelPri:forward(positivePriTable[i]:cuda())
    positivePri2[i]:copy(torch.sum(t,1))
  end
  for i = 1,#negativePriTable do
    local t = modelPri:forward(negativePriTable[i]:cuda())
    negativePri2[i]:copy(torch.sum(t,1))
  end
  local allPri2 = nn.JoinTable(1):forward{positivePri2, negativePri2}
  local targetsPri2 = nn.JoinTable(1):forward{torch.Tensor(#positivePriTable):fill(1), torch.Tensor(#negativePriTable):fill(0)}
  local allPri  = allPri2:cuda()
  local targetsPri = targetsPri2:cuda()

  
  --[[
  local positivePri, negativePri,pc,nc = getData(config.train_data_path1,class,vocabPri,fdata)
  print("positive train example : "..pc.." - negative train example : "..nc)
  positivePri = positivePri[{{1,math.floor(positivePri:size(1)*config.train_per)},{}}]
  negativePri = negativePri[{{1,math.floor(negativePri:size(1)*config.train_per)},{}}]
  --local split = nn.SplitTable(2)
  local all_rawPri = nn.JoinTable(1):forward{positivePri, negativePri}
  local targetsPri = nn.JoinTable(1):forward{torch.Tensor(positivePri:size(1)):fill(1), torch.Tensor(negativePri:size(1)):fill(0)}
  all_rawPri:cuda()
  targetsPri:cuda()
  local allPri = modelPri:forward(all_rawPri)
  ]]--
  
  local mlpClassifier = nn.Sequential()
  mlpClassifier:add(nn.Linear(config.inp_size, 1))
  mlpClassifier:add(nn.Sigmoid())
  criterion=nn.BCECriterion():cuda()
  mlpClassifier:cuda()
  
  mlpClassifier:getParameters():uniform(-1*config.init,config.init)
  
  if trainDataTest == true then
    scoreTr = 0
    precision_accTr = 0
    recall_accTr = 0
    f1_scoreTr = 0
    mcc_accTr = 0
  end
  
  if testDataTest == true then
    scoreTst = 0
    precision_accTst = 0
    recall_accTst = 0
    f1_scoreTst = 0
    mcc_accTst = 0
  end
  
  
  train_size = math.floor(allPri:size(1)*config.train_per)
  print("Train size/all : "..train_size.."/"..allPri:size(1))
  loss_logger = optim.Logger('cdlc-'..class.."-"..config.modeltype.."-"..config.dsize..'-loss-log',true,class.."-"..config.modeltype.."-"..config.dsize)  
  
  for fold = 1, config.folds do
    local errors = {}
    print("fold begin..."..os.date("%Y_%m_%d_%X"))
    for epoch = 1, config.max_epoch do
      --print("class : "..class..", fold : "..fold..", epoch "..epoch)
      --errors = {}
      -- loop across all the samples
      k=0
      local inds = torch.range(1, math.floor(train_size),config.batch_size)
      local shuffle = torch.totable(torch.randperm(inds:size(1)))

      --local shuffle = torch.totable(torch.randperm(allPri:size(1)))
      for i = 1, inds:size(1) do
        local start = inds[shuffle[i]]
        local endd = inds[shuffle[i]]+config.batch_size-1
        --print("start "..start.."- end "..endd)
        if((start > train_size) or (endd > train_size)) then
          k = k + 1
          endd = train_size
        end
        local x = allPri[{{start,endd},{}}]
        local y = targetsPri[{{start,endd}}]:cuda()
        
        --local x = allPri[shuffle[i]]
        --y = torch.Tensor(1):fill(targetPri):cuda()
        
        err = gradUpdate(mlpClassifier, x, y, criterion, config.lr)
        table.insert(errors, err)
      end
      if epoch % config.threshold == 0 then 
        config.lr = config.lr / config.lr_decay 
        loss_logger:add{['training error mean'] = torch.mean( torch.Tensor( errors))}
        loss_logger:style{['training error mean'] = '+-'}
      end
      if epoch % 100 == 0 then 
        --loss_logger:plot()  
      end
    end -- end for mlp train loop
    print(class.." fold "..fold.." model trained ")
    print("fold end..."..os.date("%Y_%m_%d_%X"))
    
    if trainDataTest == true then
    -- TRAIN DATA ACC 
    -- get L2 data
    --positiveTrSec, negativeTrSec = getData(config.train_data_path2,class,vocab2FileName)
      local positiveTrSecF, negativeTrSecF,pc,nc = getData(config.train_data_path2,class,vocabSec,fdata)
      print("positive train example (test) : "..pc.." - negative train example (test) : "..nc)
      local positiveTrSec = positiveTrSecF[{{1,math.floor(positiveTrSecF:size(1)*config.test_per)},{}}]
      local negativeTrSec = negativeTrSecF[{{1,math.floor(negativeTrSecF:size(1)*config.test_per)},{}}]
      local all_rawTrSec = nn.JoinTable(1):forward{positiveTrSec, negativeTrSec}
      local targetsTrSec = nn.JoinTable(1):forward{torch.Tensor(positiveTrSec:size(1)):fill(1), torch.Tensor(negativeTrSec:size(1)):fill(0)}
      all_rawTrSec:cuda()
      targetsTrSec:cuda()
      local allTrSec = modelSec:forward(all_rawTrSec)
    
      correctTr = 0
      predicted_positivesTr = 0
      true_positivesTr = 0
      true_negativesTr = 0
      false_positivesTr = 0
      false_negativesTr = 0
      all_positivesTr = positiveTrSec:size(1)
      print("Train Test Size "..allTrSec:size(1))
      for i = 1, allTrSec:size(1) do
        x = allTrSec[i]
        pred = mlpClassifier:forward(x)
        if pred[1] < 0.5 then
          output = 0 
        else 
          output = 1 
          predicted_positivesTr = predicted_positivesTr + 1
        end
        if output == targetsTrSec[i] then
          correctTr = correctTr + 1 
          if targetsTrSec[i] == 1 then true_positivesTr = true_positivesTr + 1 end
          if targetsTrSec[i] == 0 then true_negativesTr = true_negativesTr + 1 end
        else
          if targetsTrSec[i] == 1 then false_positivesTr = false_positivesTr + 1 end
          if targetsTrSec[i] == 0 then false_negativesTr = false_negativesTr + 1 end
        end
      end  -- End for train-test loop
    
      if not(predicted_positivesTr == 0 or all_positivesTr == 0) then 
        precisionTr = true_positivesTr / predicted_positivesTr
        recallTr = true_positivesTr / all_positivesTr
        precision_accTr = precision_accTr + precisionTr
        recall_accTr = recall_accTr + recallTr
        
        tmp = (true_positivesTr + false_positivesTr)*(true_positivesTr + false_negativesTr)*(true_negativesTr + false_positivesTr)*(true_negativesTr + false_negativesTr)
        mcc = (true_positivesTr * true_negativesTr - false_positivesTr * false_negativesTr)/math.sqrt(tmp)
        mcc_accTr = mcc_accTr + mcc
        print(class.." Train Fold "..fold.." MCC-Score: "..mcc)
        if (precisionTr+recallTr) ~=  0 then
          f1_scoreTr = f1_scoreTr + (2 * precisionTr * recallTr / (precisionTr+recallTr))
        else
          print(class.." Train Fold "..fold.." F1-Score: 0 ")
        end
        scoreTr = scoreTr + correctTr / allTrSec:size(1)
      else 
        fold = fold - 1
      end
    --score_loggerTr[lcnt]:add{['score '] = correctTr / allTrSec:size(1),['precision '] = precisionTr,['recall '] = recallTr,['f1_score '] = (2 * precisionTr * recallTr / (precisionTr+recallTr)) }
      collectgarbage()
    end
  
    if testDataTest == true then
      -- TEST DATA ACC 
      -- get L2 data
      --positiveTstSec, negativeTstSec = getData(config.test_data_path2,class,vocab2FileName)
      local positiveTstSecF, negativeTstSecF,pc,nc = getData(config.test_data_path2,class,vocabSec,"test")
      print("positive test example : "..pc.." - negative test example : "..nc)
      local positiveTstSec = positiveTstSecF[{{1,math.floor(positiveTstSecF:size(1)*config.test_per)},{}}]
      local negativeTstSec = negativeTstSecF[{{1,math.floor(negativeTstSecF:size(1)*config.test_per)},{}}]
      local all_rawTstSec = nn.JoinTable(1):forward{positiveTstSec, negativeTstSec}
      local targetsTstSec = nn.JoinTable(1):forward{torch.Tensor(positiveTstSec:size(1)):fill(1), torch.Tensor(negativeTstSec:size(1)):fill(0)}
      all_rawTstSec:cuda()
      targetsTstSec:cuda()
      local allTstSec = modelSec:forward(all_rawTstSec)
    
      correctTst = 0
      predicted_positivesTst = 0
      true_positivesTst = 0
      true_negativesTst = 0
      false_positivesTst = 0
      false_negativesTst = 0
      all_positivesTst = positiveTstSec:size(1)
      print("Test Test Size "..allTstSec:size(1))
      for i = 1, allTstSec:size(1) do
        x = allTstSec[i]
        pred = mlpClassifier:forward(x)
        if pred[1] < 0.5 then
          output = 0 
        else 
          output = 1 
          predicted_positivesTst = predicted_positivesTst + 1
        end
        if output == targetsTstSec[i] then
          correctTst = correctTst + 1 
          if targetsTstSec[i] == 1 then true_positivesTst = true_positivesTst + 1 end
          if targetsTstSec[i] == 0 then true_negativesTst = true_negativesTst + 1 end
        else
          if targetsTstSec[i] == 1 then false_positivesTst = false_positivesTst + 1 end
          if targetsTstSec[i] == 0 then false_negativesTst = false_negativesTst + 1 end
        end
      end  -- End for train-test loop
    
      if not(predicted_positivesTst == 0 or all_positivesTst == 0) then 
        precisionTst = true_positivesTst / predicted_positivesTst
        recallTst = true_positivesTst / all_positivesTst
        precision_accTst = precision_accTst + precisionTst
        recall_accTst = recall_accTst + recallTst
        
        tmp = (true_positivesTst + false_positivesTst)*(true_positivesTst + false_negativesTst)*(true_negativesTst + false_positivesTst)*(true_negativesTst + false_negativesTst)
        mcc = (true_positivesTst * true_negativesTst - false_positivesTst * false_negativesTst)/math.sqrt(tmp)
        mcc_accTst = mcc_accTst + mcc
        
        if (precisionTst+recallTst) ~= 0 then
          f1_scoreTst = f1_scoreTst + (2 * precisionTst * recallTst / (precisionTst+recallTst))
        else
          print(class.." Test Fold "..fold.." F1-Score: 0 ")
        end
        scoreTst = scoreTst + correctTst / allTstSec:size(1)
      else 
        fold = fold - 1
      end
      --print("Test F1-Score: "..fold.."  ".. (2 * precisionTst * recallTst / (precisionTst+recallTst)))
      --score_loggerTst[lcnt]:add{['score '] = correctTst / allTstSec:size(1),['precision '] = precisionTst,['recall '] = recallTst,['f1_score '] = (2 * precisionTst * recallTst / (precisionTst+recallTst)) }
      collectgarbage()
    end
    print("\n")
  end -- End for fold loop

  if trainDataTest == true then
    print("Class: "..class)
    print("Train Score: " .. (scoreTr / config.folds) * 100 .. "%")
    print("Train Precision: " .. precision_accTr / config.folds)
    print("Train Recall: " .. recall_accTr / config.folds)
    print("Train F1-Score: " .. f1_scoreTr / config.folds)
    print("Train MCC-Score: " .. mcc_accTr / config.folds)
  end
  
  if testDataTest == true then
    print("Class: "..class)
    print("Test Score: " .. (scoreTst / config.folds) * 100 .. "%")
    print("Test Precision: " .. precision_accTst / config.folds)
    print("Test Recall: " .. recall_accTst / config.folds)
    print("Test F1-Score: " .. f1_scoreTst / config.folds)
    print("Test MCC-Score: " .. mcc_accTst / config.folds)
  end

  if trainDataTest == true then
    output_file:write("Class: "..class)
    output_file:write("\n")
    output_file:write("Train Score: " .. (scoreTr / config.folds) * 100 .. "%")
    output_file:write("\n")
    output_file:write("Train Precision: " .. precision_accTr / config.folds)
    output_file:write("\n")
    output_file:write("Train Recall: " .. recall_accTr / config.folds)
    output_file:write("\n")
    output_file:write("Train F1-Score: " .. f1_scoreTr / config.folds)
    output_file:write("\n")
    output_file:write("Train MCC-Score: " .. mcc_accTr / config.folds)
    output_file:write("\n")
    output_file:write("\n")
  end

  if testDataTest == true then
    output_file:write("Class: "..class)
    output_file:write("\n")
    output_file:write("Test Score: " .. (scoreTst / config.folds) * 100 .. "%")
    output_file:write("\n")
    output_file:write("Test Precision: " .. precision_accTst / config.folds)
    output_file:write("\n")
    output_file:write("Test Recall: " .. recall_accTst / config.folds)
    output_file:write("\n")
    output_file:write("Test F1-Score: " .. f1_scoreTst / config.folds)
    output_file:write("\n")
    output_file:write("Train MCC-Score: " .. mcc_accTst / config.folds)
    output_file:write("\n")
    output_file:write("\n")
  end

  output_file:flush()
  
  if trainDataTest == true then
    f1_score_avgTr = f1_score_avgTr + f1_scoreTr / config.folds
    score_avgTr = score_avgTr +  (scoreTr / config.folds) * 100
    mcc_acc_avgTr = mcc_acc_avgTr + mcc_accTr / config.folds
  end

  if testDataTest == true then
    f1_score_avgTst = f1_score_avgTst + f1_scoreTst / config.folds
    score_avgTst = score_avgTst +  (scoreTst / config.folds) * 100
    mcc_acc_avgTst = mcc_acc_avgTst + mcc_accTst / config.folds
  end
  collectgarbage()
end -- End for class loop


if trainDataTest == true then
  print("Average Train F1-Score: " .. f1_score_avgTr / #classes)
  print("Average Train MCC-Score: " .. mcc_acc_avgTr / #classes)  
  print("Average Train Acc: " .. score_avgTr / #classes)
  output_file:write("Average Train F1-Score: " .. f1_score_avgTr / #classes)
  output_file:write("\n")
  output_file:write("Average Train MCC-Score: " .. mcc_acc_avgTr / #classes)
  output_file:write("\n")
  output_file:write("Average Train Acc: " .. score_avgTr / #classes)
  output_file:write("\n")
end

if testDataTest == true then
  print("Average Test F1-Score: " .. f1_score_avgTst / #classes)
  print("Average Test MCC-Score: " .. mcc_acc_avgTst / #classes)  
  print("Average Test Acc: " .. score_avgTst / #classes)
  output_file:write("Average Test F1-Score: " .. f1_score_avgTst / #classes)
  output_file:write("\n")
  output_file:write("Average Test MCC-Score: " .. mcc_acc_avgTst / #classes)
  output_file:write("\n")
  output_file:write("Average Test Acc: " .. score_avgTst / #classes)
  output_file:write("\n")
end

io.close(output_file)


