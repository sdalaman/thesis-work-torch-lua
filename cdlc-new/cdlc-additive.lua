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


log_flag = false

path = "/home/saban/work/additive/"
path2 = "/home/saban/work/python/pytorch-works/additive/data/ted/"
model_pathPri = path.."models/additive"
model_pathSec = model_pathPri
corpus_path = path.."models/corpus/"

test_data_pathEnTrTok = path.."data/cdlc_en_tr/test/englishTok"
test_data_pathTrEnTok = path.."data/cdlc_en_tr/test/turkishTok"
test_data_pathEnTrMorph = path.."data/cdlc_en_tr/test/englishMorph"
test_data_pathTrEnMorph = path.."data/cdlc_en_tr/test/turkishMorph"
train_data_pathEnTrTok = path.."data/cdlc_en_tr/train/englishTok"
train_data_pathTrEnTok = path.."data/cdlc_en_tr/train/turkishTok"
train_data_pathEnTrMorph = path.."data/cdlc_en_tr/train/englishMorph"
train_data_pathTrEnMorph = path.."data/cdlc_en_tr/train/turkishMorph"

test_data_pathEnDeTok = path2.."en-de-tok/test"
test_data_pathDeEnTok = path2.."de-en-tok/test"
test_data_pathEnDeMorph = path2.."en-de-morph/test"
test_data_pathDeEnMorph = path2.."de-en-morph/test"
train_data_pathEnDeTok = path2.."en-de-tok/train"
train_data_pathDeEnTok = path2.."de-en-tok/train"
train_data_pathEnDeMorph = path2.."en-de-morph/train"
train_data_pathDeEnMorph = path2.."de-en-morph/train"

test_data_pathEnFrTok = path2.."en-fr-tok/test"
test_data_pathFrEnTok = path2.."fr-en-tok/test"
test_data_pathEnFrMorph = path2.."en-fr-morph/test"
test_data_pathFrEnMorph = path2.."fr-en-morph/test"
train_data_pathEnFrTok = path2.."en-fr-tok/train"
train_data_pathFrEnTok = path2.."fr-en-tok/train"
train_data_pathEnFrMorph = path2.."en-fr-morph/train"
train_data_pathFrEnMorph = path2.."fr-en-morph/train"


modelDeTok="1/train.de-en.tok.de.0.1.175.additive.tok.64"
modelEnTok="1/english.all.tok.en-tu"
modelFrTok="1/train.en-fr.tok.fr.0.1.300.additive.tok.64"
modelTrTok="1/turkish.all.tok.en-tu"

modelDeMorph="1/train.de-en.de.0.01.1450.additive.tok.morph.256"
modelEnMorph="1/english.all.tok.morph.en-tu.0.01.3200.256.additive-morph"
modelFrMorph="1/train.en-fr.fr.0.01.950.additive.tok.morph.256"
modelTrMorph="1/turkish.all.tok.morph.en-tu.0.01.3200.256.additive-morph"

vocabEnTok="english.all.tok.en-tu"
vocabTrTok="turkish.all.tok.en-tu"
vocabEnMorph="english.all.tok.morph.en-tu"
vocabTrMorph="turkish.all.tok.morph.en-tu"

vocabDeTok="multi/train.de-en.de.tok"
vocabFrTok="multi/train.en-fr.fr.tok"
vocabDeMorph="multi/train.de-en.de.tok.morph"
vocabFrMorph="multi/train.en-fr.fr.tok.morph"

local stringx = require('pl.stringx')
local file = require('pl.file')

function gradUpdate(mlp, x, y, criterion, cfg, paramsPri, gradParamsPri,epoch)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   --local paramsPri, gradParamsPri = mlp:getParameters()
   mlp:backward(x, gradCriterion)
   if cfg.coefL1 ~= 0 or cfg.coefL2 ~= 0 then
    local norm,sign= torch.norm,torch.sign
    k1 = cfg.coefL1 * norm(paramsPri,1)
    k2 = cfg.coefL1 * norm(paramsPri,2)^2/2
    err = err + k1
    err = err + k2
    --print("epoch : "..epoch.."  err : "..err.." - k1 : "..k1.." k2 : "..k2)
    gradParamsPri:add( sign(paramsPri):mul(cfg.coefL1) + paramsPri:clone():mul(cfg.coefL2) )
   end
   mlp:updateParameters(cfg.lr)
   return err
end

function paddingOld(sequence,longest_seq)
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

function padding(sequence,longest_seq)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , longest_seq - #sequence do
    new_sequence[i] = '<pad>'
  end
  j = 1
  for i = (longest_seq - math.min(longest_seq,#sequence))+1, longest_seq do
    new_sequence[i] = sequence[j]
    j = j + 1
  end
  return new_sequence
end

function map_data(data,longest_seq,vocab)
  docstable = {}
  tcnt = 0
  for _,item in pairs(data) do
    lines = stringx.split(item[2], '\n')
    cnt = item[1]
    doctensor = torch.Tensor(#lines,cnt)
    tcnt = tcnt + 1
    for lcnt = 1,#lines do
      line_words = stringx.split(lines[lcnt], sep)
      sample = torch.Tensor(cnt)
      line_words = padding(line_words,cnt)
      for k,word in ipairs(line_words) do
        if(vocab[word] ~= nil) then
          sample[k] = vocab[word]
        else
          if vocab['_UNK'] ~= nil then
            sample[k] = vocab['_UNK']
          else
            sample[k] = 0
          end
        end
      end
      doctensor[lcnt] = sample  
    end
    table.insert(docstable,doctensor)
  end
  return docstable
end

function getData(data_path,class,vocab,datatype,fnmax)
  local pos_data = {}
  local neg_data = {}
  local longest_seq = 0
  local pcnt = 0
  local ncnt = 0
  local fcnt = 0
  for f in lfs.dir(data_path..'/'..class..'/positive') do
    local text = file.read(data_path..'/'..class..'/positive/'..f)
    if (text ~= nil) then
      lines = stringx.split(text, '\n')
      pcnt = pcnt + 1
      local ln= {}
      for lcnt = 1,#lines do
        nm_words = #stringx.split(lines[lcnt],sep)
        table.insert(ln,nm_words) 
        if longest_seq < nm_words then 
          longest_seq = nm_words
        end
      end
      --pos_data[math.max(unpack(ln))] = text
      pos_data[pcnt] = {math.max(unpack(ln)),text}
    end
    fcnt = fcnt + 1
    if fcnt == fnmax then
      break
    end
  end  
  fcnt = 0
  for f in lfs.dir(data_path..'/'..class..'/negative') do
    local text = file.read(data_path..'/'..class..'/negative/'..f)
    if (text ~= nil) then
      ncnt = ncnt + 1
      local ln= {}
      lines = stringx.split(text, '\n')
      for lcnt = 1,#lines do
        nm_words = #stringx.split(lines[lcnt],sep)
        table.insert(ln,nm_words) 
        if longest_seq < nm_words then 
          longest_seq = nm_words
        end
      end
      --neg_data[math.max(unpack(ln))] = text
      neg_data[ncnt] = {math.max(unpack(ln)),text}
      --if pcnt == ncnt and datatype == "train" then
      --  break
      --end
    end
    fcnt = fcnt + 1
    if fcnt == fnmax then
      break
    end
  end
  local pos_mapped = map_data(pos_data,longest_seq,vocab)
  local neg_mapped = map_data(neg_data,longest_seq,vocab)
  return pos_mapped,neg_mapped,pcnt,ncnt
end

sep = '\t'
--local classes = {'arts'}
local classes = {'art','arts','biology','business','creativity','culture','design','economics','education','entertainment','global','health','politics','science','technology'}


-- Default configuration
config = {}
config.train_data_pathPri = "" -- train files  for lang 1
config.train_data_pathSec = "" -- train files  for lang 2
config.test_data_pathPri = "" -- test files  for lang 1
config.test_data_pathSec = "" -- test files  for lang 2
config.modelPri = "" -- model file  for lang 1
config.modelSec = "" -- model file  for lang 2
config.vocabPri = "" -- vocab file  for lang 1
config.vocabSec = "" -- vocab file  for lang 2
config.emb_size = 0
config.lr = 0.01
config.lr_decay = 1
config.threshold = 5
config.graphshow = 100
config.max_epoch = 300
config.max_loop = 5
config.folds = 0
config.init = 0.001
config.test_per = 0.25
config.train_per = 1
config.train_test = 0  -- test with train data
config.test_test_pri = 0   -- test with pri test data
config.test_test_sec = 0   -- test with sec test data
config.trmax = 10   -- max train doc
config.tstmax = 10  -- max test doc
config.out_file = "" 
config.dtype = "tok" -- 
config.dsize = "all" -- 
config.batch_size = 10
config.langPri = "En"
config.langSec= "Tr"
config.coefL1 = 0
config.coefL2= 0


-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-lr", config.lr)
cmd:option("-lr_decay", config.lr_decay)
cmd:option("-max_epoch", config.max_epoch)
cmd:option("-max_loop", config.max_loop)
cmd:option("-batch_size", config.batch_size)
cmd:option("-emb_size", config.emb_size)
cmd:option("-threshold", config.threshold)
cmd:option("-graphshow", config.graphshow)
cmd:option("-folds", config.folds)
cmd:option("-init", config.init)
cmd:option("-test_per", config.test_per)
cmd:option("-train_per", config.train_per)
cmd:option("-train_test", config.train_test)
cmd:option("-test_test_pri", config.test_test_pri)
cmd:option("-test_test_sec", config.test_test_sec)
cmd:option("-out_file", config.out_file)
cmd:option("-trmax", config.trmax)
cmd:option("-tstmax", config.tstmax)
cmd:option("-dtype", config.dtype)
cmd:option("-langPri", config.langPri)
cmd:option("-langSec", config.langSec)
cmd:option("-coefL1", config.coefL1)
cmd:option("-coefL2", config.coefL2)
params = cmd:parse(arg)


--th cdlc-additive.lua -vocab1 english.all.tok.morph.en-tu -vocab2 multi/train.en-fr.fr.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 300 -threshold 5 -graphshow 100 -folds 10 -init 0.001 -test_per 1 -train_per 1 -train_test 0 -test_test_pri 1 -test_test_sec 1 -dtype morph -batch_size 10 -max_loop 2 -trmax 100 -tstmax 100 -model1  3/english.all.tok.morph.en-tu.0.01.3200.256.additive-morph -model2  1/train.en-fr.fr.0.01.950.additive.tok.morph.256 -lang1 En -lang2 Fr 


--th cdlc-additive.lua -lr 0.01 -lr_decay 1 -max_epoch 300 -threshold 5 -graphshow 100 -folds 10 -init 0.001 -test_per 1 -train_per 1 -train_test 0 -test_test_pri 1 -test_test_sec 1 -dtype tok -batch_size 10 -max_loop 2 -trmax 1000 -tstmax 100 -langPri En -langSec Tr 


for param, value in pairs(params) do
    config[param] = value
end

if config.dtype == "morph" then
  model_pathPri = model_pathPri.."-morph"
  model_pathSec = model_pathSec.."-morph"
end

if config.dtype == "morph" then
  
  if config.langPri == "En" then
    config.train_data_pathPri = train_data_pathEnTrMorph -- 
    config.test_data_pathPri = test_data_pathEnTrMorph -- 
    config.modelPri = modelEnMorph
    config.vocabPri = vocabEnMorph
  end
  if config.langPri == "Tr" then
    config.train_data_pathPri = train_data_pathTrEnMorph -- 
    config.test_data_pathPri = test_data_pathTrEnMorph -- 
    config.modelPri = modelTrMorph
    config.vocabPri = vocabTrMorph
  end
  if config.langPri == "De" then
    config.train_data_pathPri = train_data_pathDeEnMorph -- 
    config.test_data_pathPri = test_data_pathDeEnMorph -- 
    config.modelPri = modelDeMorph
    config.vocabPri = vocabDeMorph
  end
  if config.langPri == "Fr" then
    config.train_data_pathPri = train_data_pathFrEnMorph -- 
    config.test_data_pathPri = test_data_pathFrEnMorph -- 
    config.modelPri = modelFrMorph
    config.vocabPri = vocabFrMorph
  end

  if config.langSec == "En" then
    config.train_data_pathSec = train_data_pathEnTrMorph -- 
    config.test_data_pathSec = test_data_pathEnTrMorph -- 
    config.modelSec = modelEnMorph
    config.vocabSec = vocabEnMorph
  end
  if config.langSec == "Tr" then
    config.train_data_pathSec = train_data_pathTrEnMorph -- 
    config.test_data_pathSec = test_data_pathTrEnMorph -- 
    config.modelSec = modelTrMorph
    config.vocabSec = vocabTrMorph
  end
  if config.langSec == "De" then
    config.train_data_pathSec = train_data_pathDeEnMorph -- 
    config.test_data_pathSec = test_data_pathDeEnMorph -- 
    config.modelSec = modelDeMorph
    config.vocabSec = vocabDeMorph
  end
  if config.langSec == "Fr" then
    config.train_data_pathSec = train_data_pathFrEnMorph -- 
    config.test_data_pathSec = test_data_pathFrEnMorph -- 
    config.modelSec = modelFrMorph
    config.vocabSec = vocabFrMorph
  end
  
else
  
  if config.langPri == "En" then
    config.train_data_pathPri = train_data_pathEnTrTok -- 
    config.test_data_pathPri = test_data_pathEnTrTok -- 
    config.modelPri = modelEnTok
    config.vocabPri = vocabEnTok
  end
  if config.langPri == "Tr" then
    config.train_data_pathPri = train_data_pathTrEnTok -- 
    config.test_data_pathPri = test_data_pathTrEnTok -- 
    config.modelPri = modelTrTok
    config.vocabPri = vocabTrTok
  end
  if config.langPri == "De" then
    config.train_data_pathPri = train_data_pathDeEnTok -- 
    config.test_data_pathPri = test_data_pathDeEnTok -- 
    config.modelPri = modelDeTok
    config.vocabPri = vocabDeTok
  end
  if config.langPri == "Fr" then
    config.train_data_pathPri = train_data_pathFrEnTok -- 
    config.test_data_pathPri = test_data_pathFrEnTok -- 
    config.modelPri = modelFrTok
    config.vocabPri = vocabFrTok
  end

  if config.langSec == "En" then
    config.train_data_pathSec = train_data_pathEnTrTok -- 
    config.test_data_pathSec = test_data_pathEnTrTok -- 
    config.modelSec = modelEnTok
    config.vocabSec = vocabEnTok
  end
  if config.langSec == "Tr" then
    config.train_data_pathSec = train_data_pathTrEnTok -- 
    config.test_data_pathSec = test_data_pathTrEnTok -- 
    config.modelSec = modelTrTok
    config.vocabSec = vocabTrTok
  end
  if config.langSec == "De" then
    config.train_data_pathSec = train_data_pathDeEnTok -- 
    config.test_data_pathSec = test_data_pathDeEnTok -- 
    config.modelSec = modelDeTok
    config.vocabSec = vocabDeTok
  end
  if config.langSec == "Fr" then
    config.train_data_pathSec = train_data_pathFrEnTok -- 
    config.test_data_pathSec = test_data_pathFrEnTok -- 
    config.modelSec = modelFrTok
    config.vocabSec = vocabFrTok
  end
end

lang_pair = config.langPri.."-"..config.langSec
model_pathPri = model_pathPri.."/"..config.dsize.."/"..config.langPri.."/"
model_pathSec = model_pathSec.."/"..config.dsize.."/"..config.langSec.."/"

config.modelPri = model_pathPri..config.modelPri..".model" --model file  for lang 1
config.modelSec = model_pathSec..config.modelSec..".model" -- model file  for lang 2
config.vocabPri = corpus_path..config.vocabPri..".vocab" -- vocab file  for lang 1
config.vocabSec = corpus_path..config.vocabSec..".vocab" -- vocab file  for lang 2

for i,j in pairs(config) do
    --print(i..": "..j)
end

local modelPri = torch.load(config.modelPri):double()
local modelSec = torch.load(config.modelSec):double()
modelPri:clearState()  -- Clear intermediate module states 
modelSec:clearState()  -- Clear intermediate module states 
modelPri:cuda()
modelSec:cuda()

--config.emb_size = modelPri:get(2):get(1):get(1):parameters():size()[2]
lp = modelPri:findModules('nn.LookupTableMaskZero')
config.emb_size = lp[1]:parameters()[1]:size()[2]

local vocabPri = torch.load(config.vocabPri)
local vocabSec = torch.load(config.vocabSec)

trainDataSec = config.train_test == 1
tstDataPri = config.test_test_pri == 1
tstDataSec = config.test_test_sec == 1


file_timestamp = os.date("%Y_%m_%d_%X")
--output_file = io.open("f1scorevslr_"..nm.."-"..config.dtype.."-"..config.dsize.."-"..file_timestamp.."-newtotal-full.csv", "w")
config.out_file = "Additive".."-"..config.dtype.."-"..lang_pair
output_file = io.open(config.out_file.."-"..file_timestamp..".csv", "w")
output_file:write("Program args : \n")
summary_file = io.open("summary-additive"..".csv", "a+")
for param, value in pairs(params) do
  output_file:write(param.." : "..value..", ")
end
output_file:write("\n")
output_file:flush()

if trainDataSec == true then
  f1_score_avgTrainSec = 0
  score_avgTrainSec = 0
  mcc_acc_avgTrainSec = 0
end

if tstDataPri == true then
  f1_score_avgTstPri = 0
  score_avgTstPri = 0
  mcc_acc_avgTstPri = 0
end

if tstDataSec == true then
  f1_score_avgTstSec = 0
  score_avgTstSec = 0
  mcc_acc_avgTstSec = 0
end

dataTypeTrainPri = "traindatapri"
dataTypeTrainSec = "traindatasec"
dataTypeTestPri  = "testdatapri"
dataTypeTestSec  = "testdatasec"

function prepareData(model,className,vocab,dataType,cfg)
  if dataType == dataTypeTrainPri then
    fnmax = cfg.trmax
    data_path = cfg.train_data_pathPri
  end
  if dataType == dataTypeTrainSec then
    fnmax = cfg.trmax
    data_path = cfg.train_data_pathSec
  end
  if dataType == dataTypeTestPri then
    fnmax = cfg.tstmax
    data_path = cfg.test_data_pathPri
  end
  if dataType == dataTypeTestSec then
    fnmax = cfg.tstmax
    data_path = cfg.test_data_pathSec
  end
  
  model:evaluate()
  positiveTableT, negativeTableT,pc,nc = getData(data_path,className,vocab,dataType,fnmax)
  local positiveTable = {}
  local negativeTable = {}
  for i = 1,math.floor(#positiveTableT*cfg.train_per) do
    table.insert(positiveTable,positiveTableT[i])
  end
  for i = 1,math.floor(#negativeTableT*cfg.train_per) do
    table.insert(negativeTable,negativeTableT[i])
  end
  positiveTableT = {}
  negativeTableT = {}
  
  local positive = torch.zeros(#positiveTable,cfg.emb_size):double():cuda()
  local negative = torch.zeros(#negativeTable,cfg.emb_size):double():cuda()
  for i = 1,#positiveTable do
    local t = model:forward(positiveTable[i]:cuda())
    positive[i]:copy(torch.sum(t,1))
  end
  for i = 1,#negativeTable do
    local t = model:forward(negativeTable[i]:cuda())
    negative[i]:copy(torch.sum(t,1))
  end
  if dataType == dataTypeTrainPri then
    diff = negative:size()[1] - positive:size()[1]
    local posT = negative:clone():fill(0)
    math.randomseed( os.time() )
    for i=1,positive:size()[1] do
      posT[i]:copy(positive[i])
    end
    for i=1,diff do
      cnt = positive:size()[1]+i
      ln = math.random(1,positive:size()[1])
      posT[cnt]:copy(positive[ln])
    end
    positive = posT
  end
  local inputs = nn.JoinTable(1):forward{positive, negative}
  local targets = nn.JoinTable(1):forward{torch.Tensor(positive:size()[1]):fill(1), torch.Tensor(negative:size()[1]):fill(0)}
  return inputs,targets,pc,nc
end

class_loop_cnt = 0
--score_loggerTstPriAll = optim.Logger('cdlc-additive-test-all-pri-'..config.dtype.."-"..config.dsize.."-"..lang_pair..'-score-log',true,"test-all-pri-"..config.dtype.."-"..config.dsize.."-"..lang_pair)  
--score_loggerTstSecAll = optim.Logger('cdlc-additive-test-all-sec-'..config.dtype.."-"..config.dsize.."-"..lang_pair..'-score-log',true,"test-all-sec-"..config.dtype.."-"..config.dsize.."-"..lang_pair)  

f1_table_tst_pri = {}
f1_table_tst_sec = {}
mcc_table_tst_pri = {}
mcc_table_tst_sec = {}
score_table_tst_pri = {}
score_table_tst_sec = {}

for _,class in ipairs(classes) do
  print("\n--------------------\n")
  print("\nClass "..class.." "..lang_pair)
  print("..."..os.date("%Y_%m_%d_%X"))
  
  allPri,targetsPri,pc,nc = prepareData(modelPri,class,vocabPri,dataTypeTrainPri,config)
  print("positive train example : "..pc.." - negative train example : "..nc)

  allPri:cuda()
  targetsPri:cuda()
  
  --score_loggerTrSec = optim.Logger('./log/cdlc-additive-train-sec-'..class..'-'..config.dtype.."-"..config.dsize..'-score-log',true,class.."-".."train-sec-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  --score_loggerTstPri = optim.Logger('./log/cdlc-additive-test-pri-'..class..'-'..config.dtype.."-"..config.dsize..'-score-log',true,class.."-".."test-pri-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  score_loggerTstSec = optim.Logger('./log/cdlc-additive-test-sec-'..class..'-'..config.dtype.."-"..config.dsize..'-score-log',true,class.."-".."test-sec-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  


  --local mlpClassifier = nn.Sequential()
  --mlpClassifier:add(nn.Linear(config.emb_size, 1))
  --mlpClassifier:add(nn.Sigmoid())
  --criterion=nn.BCECriterion():cuda()
  --mlpClassifier:cuda()
  
  local mlpClassifier = nn.Sequential()
  mlpClassifier:add(nn.Linear(config.emb_size, config.emb_size))
  mlpClassifier:add(nn.Tanh())
  mlpClassifier:add(nn.Linear(config.emb_size, 1))
  mlpClassifier:add(nn.Sigmoid())
  criterion=nn.BCECriterion():cuda()
  mlpClassifier:cuda()

  
  mlpClassifier:getParameters():uniform(-1*config.init,config.init)
  local paramsPri, gradParamsPri = mlpClassifier:getParameters()

  
  if trainDataSec == true then
    scoreTrainSec = 0
    precision_accTrainSec = 0
    recall_accTrainSec = 0
    f1_scoreTrainSec = 0
    mcc_accTrainSec = 0
  end
  
  if tstDataPri == true then
    scoreTstPri = 0
    precision_accTstPri = 0
    recall_accTstPri = 0
    f1_scoreTstPri = 0
    mcc_accTstPri = 0
  end
  
  if tstDataSec == true then
    scoreTstSec = 0
    precision_accTstSec = 0
    recall_accTstSec = 0
    f1_scoreTstSec = 0
    mcc_accTstSec = 0
  end
  
  trainDataLoaded = false
  testDataLoadedPri = false
  testDataLoadedSec = false
  
  train_size = math.floor(allPri:size(1)*config.train_per)
  print("Train size/all : "..train_size.."/"..allPri:size(1))
  loss_logger_train = optim.Logger('./log/cdlc-additive-'..class.."-"..config.dtype.."-"..config.dsize..'-train-loss-log',true,class.."-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  loss_logger_test = optim.Logger('./log/cdlc-additive-'..class.."-"..config.dtype.."-"..config.dsize..'-test-loss-log',true,class.."-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  
  loopcnt = 1
  totalfoldcnt = 0
  local errorsTrain = {}
  local errorsTest = {}
  tempTable = {}
  while true do
  for fold = 1, config.folds do
    totalfoldcnt = totalfoldcnt + 1
    print("fold begin..."..os.date("%Y_%m_%d_%X"))
    mlpClassifier:training()
    for epoch = 1, config.max_epoch do
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
        local x = allPri[{{start,endd},{}}]:cuda()
        local y = targetsPri[{{start,endd}}]:cuda()
        err = gradUpdate(mlpClassifier, x, y, criterion,config,paramsPri, gradParamsPri,epoch)
        table.insert(errorsTrain, err)
      end
      if epoch % config.threshold == 0 then 
        config.lr = config.lr * config.lr_decay 
        --print("New learning rate : "..config.lr)
      end
      if epoch % config.graphshow == 0 then 
        if log_flag == true then
          loss_logger_train:plot()
        end
      end
      loss_logger_train:add{['training error mean'] = torch.mean( torch.Tensor( errorsTrain))}
      loss_logger_train:style{['training error mean'] = '+-'}
    end -- end for mlp train loop
    if log_flag == true then
          loss_logger_train:plot()
    end
    print(class.." fold :"..fold.." - fold loop cnt : "..loopcnt.." model trained ")
    print("fold end..."..os.date("%Y_%m_%d_%X"))
    
    if trainDataSec == true then
    -- SECONDARY TRAIN DATA ACC 
      if trainDataLoaded == false then
        allTrSec,targetsTrSec,pc,nc = prepareData(modelSec,class,vocabSec,dataTypeTrainSec,config)
        allTrSec:cuda()
        targetsTrSec:cuda()
        print("positive train example (test) : "..pc.." - negative train example (test) : "..nc)
        trainDataLoaded = true
      else
        print("train examples already loaded ")
      end
    
      correctTrSec = 0
      predicted_positivesTrSec = 0
      true_positivesTrSec = 0
      true_negativesTrSec = 0
      false_positivesTrSec = 0
      false_negativesTrSec = 0
      all_positivesTrSec = 0
      mlpClassifier:evaluate()
      print("Train Test Size "..allTrSec:size(1))
      for i = 1, allTrSec:size(1) do
        if targetsTrSec[i] == 1 then all_positivesTrSec = all_positivesTrSec + 1 end
        x = allTrSec[i]:cuda()
        pred = mlpClassifier:forward(x)
        if pred[1] < 0.5 then
          output = 0 
        else 
          output = 1 
          predicted_positivesTrSec = predicted_positivesTrSec + 1
        end
        if output == targetsTrSec[i] then
          correctTrSec = correctTrSec + 1 
          if targetsTrSec[i] == 1 then true_positivesTrSec = true_positivesTrSec + 1 end
          if targetsTrSec[i] == 0 then true_negativesTrSec = true_negativesTrSec + 1 end
        else
          if output == 1 then false_positivesTrSec = false_positivesTrSec + 1 end
          if output == 0 then false_negativesTrSec = false_negativesTrSec + 1 end
        end
      end  -- End for train-test loop
    
      print("TP:"..true_positivesTrSec..", FP:"..false_positivesTrSec..", TN:"..true_negativesTrSec..", FN:"..false_negativesTrSec)
      if not(predicted_positivesTrSec == 0 or all_positivesTrSec == 0) then 
        precisionTrSec = true_positivesTrSec / predicted_positivesTrSec
        recallTrSec = true_positivesTrSec / all_positivesTrSec
        precision_accTrSec = precision_accTrSec + precisionTrSec
        recall_accTrSec = recall_accTrSec + recallTrSec
        
        tmp = (true_positivesTrSec + false_positivesTrSec)*(true_positivesTrSec + false_negativesTrSec)*(true_negativesTrSec + false_positivesTrSec)*(true_negativesTrSec + false_negativesTrSec)
        mcc = (true_positivesTrSec * true_negativesTrSec - false_positivesTrSec * false_negativesTrSec)/math.sqrt(tmp)
        mcc_accTrSec = mcc_accTrSec + mcc
        f1_score = 0
        if (precisionTr+recallTrSec) ~=  0 then
          f1_score = (2 * precisionTrSec * recallTrSec / (precisionTrSec + recallTrSec))
          f1_scoreTrSec = f1_scoreTrSec + f1_score
        else
          print(class.." Train Fold : "..fold..", loop : "..loopcnt.." F1-Score: 0 ")
        end
        scoreTrSec = scoreTrSec + correctTrSec / allTrSec:size(1)
        
        score_loggerTrSec:add{['f1-score'] = f1_score,['score'] = correctTr / allTrSec:size(1),['mcc-score'] = mcc}
        score_loggerTrSec:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
        print("fr score tr sec "..f1_score)
      else 
        print("predicted_positivesTrSec :"..predicted_positivesTrSec.." - all_positivesTrSec :"..all_positivesTrSec)         
        fold = fold - 1
      end
      collectgarbage()
    end
  
    if tstDataPri == true then
      -- PRIMARY TEST DATA ACC 
      if testDataLoadedPri == false then
        allTstPri,targetsTstPri,pc,nc = prepareData(modelPri,class,vocabPri,dataTypeTestPri,config)
        allTstPri:cuda()
        targetsTstPri:cuda()
        print("positive primary test example : "..pc.." - negative primary test example : "..nc)
        testDataLoadedPri = true
      else
        print("primary test examples already loaded ")
      end
    
      correctTstPri = 0
      predicted_positivesTstPri = 0
      true_positivesTstPri = 0
      true_negativesTstPri = 0
      false_positivesTstPri = 0
      false_negativesTstPri = 0
      all_positivesTstPri = 0
      mlpClassifier:evaluate()
      print("Test Test Size "..allTstPri:size(1))
      for i = 1, allTstPri:size(1) do
        if targetsTstPri[i] == 1 then all_positivesTstPri = all_positivesTstPri + 1 end
        x = allTstPri[i]:cuda()
        pred = mlpClassifier:forward(x)
        if pred[1] < 0.5 then
          output = 0 
        else 
          output = 1 
          predicted_positivesTstPri = predicted_positivesTstPri + 1
        end
        if output == targetsTstPri[i] then
          correctTstPri = correctTstPri + 1 
          if targetsTstPri[i] == 1 then true_positivesTstPri = true_positivesTstPri + 1 end
          if targetsTstPri[i] == 0 then true_negativesTstPri = true_negativesTstPri + 1 end
        else
          if output == 1 then false_positivesTstPri = false_positivesTstPri + 1 end
          if output == 0 then false_negativesTstPri = false_negativesTstPri + 1 end
        end
      end  -- End for primary test loop
    
      print("TP:"..true_positivesTstPri..", FP:"..false_positivesTstPri..", TN:"..true_negativesTstPri..", FN:"..false_negativesTstPri)
      if not(predicted_positivesTstPri == 0 or all_positivesTstPri == 0) then 
        precisionTstPri = true_positivesTstPri / predicted_positivesTstPri
        recallTstPri = true_positivesTstPri / all_positivesTstPri
        precision_accTstPri = precision_accTstPri + precisionTstPri
        recall_accTstPri = recall_accTstPri + recallTstPri
        
        tmp = (true_positivesTstPri + false_positivesTstPri)*(true_positivesTstPri + false_negativesTstPri)*(true_negativesTstPri + false_positivesTstPri)*(true_negativesTstPri + false_negativesTstPri)
        mcc = (true_positivesTstPri * true_negativesTstPri - false_positivesTstPri * false_negativesTstPri)/math.sqrt(tmp)
        mcc_accTstPri = mcc_accTstPri + mcc
        
        f1_score = 0
        if (precisionTstPri+recallTstPri) ~= 0 then
          f1_score = (2 * precisionTstPri * recallTstPri / (precisionTstPri+recallTstPri))
          f1_scoreTstPri = f1_scoreTstPri + f1_score
        else
          print(class.." Test Fold : "..fold..", loop : "..loopcnt.." F1-Score: 0 ")
        end
        scoreTstPri = scoreTstPri + correctTstPri / allTstPri:size(1)
        
        --score_loggerTstPri:add{['f1-score'] = f1_score,['score'] = correctTstPri / allTstPri:size(1),['mcc-score'] = mcc}
        --score_loggerTstPri:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
        print("fr score tst pri "..f1_score)
      else 
        print("predicted_positivesTstPri :"..predicted_positivesTstPri.." - all_positivesTstPri :"..all_positivesTstPri)         
        fold = fold - 1
      end
      collectgarbage()
    end
    
    if tstDataSec == true then
      -- SECONDARY TEST DATA ACC 
      if testDataLoadedSec == false then
        allTstSec,targetsTstSec,pc,nc = prepareData(modelSec,class,vocabSec,dataTypeTestSec,config)
        allTstSec:cuda()
        targetsTstSec:cuda()
        print("positive secondary test example : "..pc.." - negative secondary test example : "..nc)
        testDataLoadedSec = true
      else
        print("secondary test examples already loaded ")
      end
    
      correctTstSec = 0
      predicted_positivesTstSec = 0
      true_positivesTstSec = 0
      true_negativesTstSec = 0
      false_positivesTstSec = 0
      false_negativesTstSec = 0
      all_positivesTstSec = 0
      local err
      mlpClassifier:evaluate()
      print("Test Test Size "..allTstSec:size(1))
      for i = 1, allTstSec:size(1) do
        if targetsTstSec[i] == 1 then all_positivesTstSec = all_positivesTstSec + 1 end
        x = allTstSec[i]:cuda()
        pred = mlpClassifier:forward(x)
        y= torch.Tensor({targetsTstSec[i]}):cuda()
        err = criterion:forward(pred,y)
        table.insert(errorsTest, err)
        loss_logger_test:add{['test error mean'] = torch.mean( torch.Tensor( errorsTest))}
        loss_logger_test:style{['test error mean'] = '+-'}
        if pred[1] < 0.5 then
          output = 0 
        else 
          output = 1 
          predicted_positivesTstSec = predicted_positivesTstSec + 1
        end
        if output == targetsTstSec[i] then
          correctTstSec = correctTstSec + 1 
          if targetsTstSec[i] == 1 then true_positivesTstSec = true_positivesTstSec + 1 end
          if targetsTstSec[i] == 0 then true_negativesTstSec = true_negativesTstSec + 1 end
        else
          if output == 1 then false_positivesTstSec = false_positivesTstSec + 1 end
          if output == 0 then false_negativesTstSec = false_negativesTstSec + 1 end
        end
      end  -- End for secondary test loop
      if log_flag == true then
          loss_logger_test:plot()
        end
      print("TP:"..true_positivesTstSec..", FP:"..false_positivesTstSec..", TN:"..true_negativesTstSec..", FN:"..false_negativesTstSec)
      if not(predicted_positivesTstSec == 0 or all_positivesTstSec == 0) then 
        precisionTstSec = true_positivesTstSec / predicted_positivesTstSec
        recallTstSec = true_positivesTstSec / all_positivesTstSec
        precision_accTstSec = precision_accTstSec + precisionTstSec
        recall_accTstSec = recall_accTstSec + recallTstSec
        
        tmp = (true_positivesTstSec + false_positivesTstSec)*(true_positivesTstSec + false_negativesTstSec)*(true_negativesTstSec + false_positivesTstSec)*(true_negativesTstSec + false_negativesTstSec)
        mcc = (true_positivesTstSec * true_negativesTstSec - false_positivesTstSec * false_negativesTstSec)/math.sqrt(tmp)
        mcc_accTstSec = mcc_accTstSec + mcc
        
        f1_score = 0
        if (precisionTstSec+recallTstSec) ~= 0 then
          f1_score = (2 * precisionTstSec * recallTstSec / (precisionTstSec+recallTstSec))
          f1_scoreTstSec = f1_scoreTstSec + f1_score
        else
          print(class.." Test Fold : "..fold..", loop : "..loopcnt.." F1-Score: 0 ")
        end
        print("score tst sec "..scoreTstSec.." "..correctTstSec / allTstSec:size(1).."\n")
        scoreTstSec = scoreTstSec + correctTstSec / allTstSec:size(1)
        
        score_loggerTstSec:add{['f1-score'] = f1_score,['score'] = correctTstSec / allTstSec:size(1),['mcc-score'] = mcc}
        score_loggerTstSec:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
        --score_loggerTstSec:plot()
        print("fr score tst sec "..f1_score)
      else 
        print("predicted_positivesTstSec :"..predicted_positivesTstSec.." - all_positivesTstSec :"..all_positivesTstSec)         
        fold = fold - 1
      end
      tempTable[#tempTable+1] = correctTstSec / allTstSec:size(1)
      collectgarbage()
    end
    
    print("\n")
  end -- End for fold loop
  
  loopcnt = loopcnt + 1
  if f1_scoreTstSec > 0 then 
    break -- exit while
  end
  
  if loopcnt > config.max_loop then 
    print("Max loop reached : "..loopcnt)
    break -- exit while
  end
  
  end -- while for loop 

  tt = 0
for i=1, #tempTable do
  print( tempTable[i].." ")
  tt = tt + tempTable[i]
end
print( "toplam :  "..tt)

print("fold cnt "..totalfoldcnt.."\n")
  
  if trainDataSec == true then
    print("Class: "..class.." "..lang_pair)
    print("Secondary Train Score: " .. (scoreTrSec / totalfoldcnt) * 100 .. "%")
    print("Secondary Train Precision: " .. precision_accTrSec / totalfoldcnt)
    print("Secondary Train Recall: " .. recall_accTrSec / totalfoldcnt)
    print("Secondary Train F1-Score: " .. f1_scoreTrSec / totalfoldcnt)
    print("Secondary Train MCC-Score: " .. mcc_accTrSec / totalfoldcnts)
  end
  
  if tstDataPri == true then
    print("Class: "..class.." "..lang_pair)
    print("Primary Test Score: " .. (scoreTstPri / totalfoldcnt) * 100 .. "%")
    print("Primary Test Precision: " .. precision_accTstPri / totalfoldcnt)
    print("Primary Test Recall: " .. recall_accTstPri / totalfoldcnt)
    print("Primary Test F1-Score: " .. f1_scoreTstPri / totalfoldcnt)
    print("Primary Test MCC-Score: " .. mcc_accTstPri / totalfoldcnt)
  end
  
  if tstDataSec == true then
    print("Class: "..class.." "..lang_pair)
    print("Secondary Test Score: " .. (scoreTstSec / totalfoldcnt) * 100 .. "%")
    print("Secondary Test Precision: " .. precision_accTstSec / totalfoldcnt)
    print("Secondary Test Recall: " .. recall_accTstSec / totalfoldcnt)
    print("Secondary Test F1-Score: " .. f1_scoreTstSec / totalfoldcnt)
    print("Secondary Test MCC-Score: " .. mcc_accTstSec / totalfoldcnt)
  end

  if trainDataSec == true then
    output_file:write("Class: "..class.." "..lang_pair)
    output_file:write("\n")
    output_file:write("Secondary Train Score: " .. (scoreTrSec / totalfoldcnt) * 100 .. "%")
    output_file:write("\n")
    output_file:write("Secondary Train Precision: " .. precision_accTrSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Secondary Train Recall: " .. recall_accTrSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Secondary Train F1-Score: " .. f1_scoreTrSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Secondary Train MCC-Score: " .. mcc_accTrSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("\n")
  end

  if tstDataPri == true then
    output_file:write("Class: "..class.." "..lang_pair)
    output_file:write("\n")
    output_file:write("Primary Test Score: " .. (scoreTstPri / totalfoldcnt) * 100 .. "%")
    output_file:write("\n")
    output_file:write("Primary Test Precision: " .. precision_accTstPri / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Primary Test Recall: " .. recall_accTstPri / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Primary Test F1-Score: " .. f1_scoreTstPri / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Primary Test MCC-Score: " .. mcc_accTstPri / totalfoldcnt)
    output_file:write("\n")
    output_file:write("\n")
  end
  
  if tstDataSec == true then
    output_file:write("Class: "..class.." "..lang_pair)
    output_file:write("\n")
    output_file:write("Secondary Test Score: " .. (scoreTstSec / totalfoldcnt) * 100 .. "%")
    output_file:write("\n")
    output_file:write("Secondary Test Precision: " .. precision_accTstSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Secondary Test Recall: " .. recall_accTstSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Secondary Test F1-Score: " .. f1_scoreTstSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("Secondary Test MCC-Score: " .. mcc_accTstSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("\n")
  end

  output_file:flush()
  
  if trainDataSec == true then
    f1_score_avgTrSec = f1_score_avgTrSec + f1_scoreTrSec / totalfoldcnt
    score_avgTrSec = score_avgTrSec +  (scoreTrSec / totalfoldcnt) * 100
    mcc_acc_avgTrSec = mcc_acc_avgTrSec + mcc_accTrSec / totalfoldcnt
  end

  if tstDataPri == true then
    f1_score_avgTstPri = f1_score_avgTstPri + f1_scoreTstPri / totalfoldcnt
    score_avgTstPri = score_avgTstPri +  (scoreTstPri / totalfoldcnt) * 100
    mcc_acc_avgTstPri = mcc_acc_avgTstPri + mcc_accTstPri / totalfoldcnt
  end

  if tstDataSec == true then
    f1_score_avgTstSec = f1_score_avgTstSec + f1_scoreTstSec / totalfoldcnt
    score_avgTstSec = score_avgTstSec +  (scoreTstSec / totalfoldcnt) * 100
    mcc_acc_avgTstSec = mcc_acc_avgTstSec + mcc_accTstSec / totalfoldcnt
  end
  class_loop_cnt = class_loop_cnt + 1
  --score_loggerTstPriAll:add{['f1-score'] = f1_score_avgTstPri / class_loop_cnt,['score'] = score_avgTstPri / class_loop_cnt,['mcc-score'] = mcc_acc_avgTstPri / class_loop_cnt}
  --score_loggerTstPriAll:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
  --score_loggerTstPriAll:plot()
  --score_loggerTstSecAll:add{['f1-score'] = f1_score_avgTstSec / class_loop_cnt,['score'] = score_avgTstSec / class_loop_cnt,['mcc-score'] = mcc_acc_avgTstSec / class_loop_cnt}
  --score_loggerTstSecAll:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
  --score_loggerTstSecAll:plot()

  collectgarbage()
end -- End for class loop

--summary_file:write("\n-----------------------------------\n")
for param, value in pairs(params) do
  --summary_file:write(param.." : "..value..", ")
end
--summary_file:write(config.folds.." : "..value..", ")
--summary_file:write("\n")
--summary_file:flush()

if trainDataSec == true then
  f1_avg=f1_score_avgTrSec / #classes
  mcc_avg=mcc_acc_avgTrSec / #classes
  score_avg=score_avgTrSec / #classes
  print("Average Secondary Train F1-Score: " .. f1_avg)
  print("Average Secondary Train MCC-Score: " .. mcc_avg)  
  print("Average Secondary Train Acc: " .. score_avg)
  output_file:write("Average Secondary Train F1-Score: " .. f1_avg)
  output_file:write("\n")
  output_file:write("Average Secondary Train MCC-Score: " .. mcc_avg)
  output_file:write("\n")
  output_file:write("Average Secondary Train Acc: " .. score_avg)
  output_file:write("\n")
  summary_file:write("Train,Sec,"..config.dtype..","..config.folds..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
end

if tstDataPri == true then
  f1_avg=f1_score_avgTstPri / #classes
  mcc_avg=mcc_acc_avgTstPri / #classes
  score_avg=score_avgTstPri / #classes
  print("Average Primary Test F1-Score: " .. f1_avg)
  print("Average Primary Test MCC-Score: " .. mcc_avg)  
  print("Average Primary Test Acc: " .. score_avg)
  output_file:write("Average Primary Test F1-Score: " .. f1_avg)
  output_file:write("\n")
  output_file:write("Average Primary Test MCC-Score: " .. mcc_avg)
  output_file:write("\n")
  output_file:write("Average Primary Test Acc: " .. score_avg)
  output_file:write("\n")
  summary_file:write("Test,Pri,"..config.dtype..","..config.folds..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
end

if tstDataSec == true then
  f1_avg=f1_score_avgTstSec / #classes
  mcc_avg=mcc_acc_avgTstSec / #classes
  score_avg=score_avgTstSec / #classes
  print("Average Secondary Test F1-Score: " .. f1_avg)
  print("Average Secondary Test MCC-Score: " .. mcc_avg)  
  print("Average Secondary Test Acc: " .. score_avg)
  output_file:write("Average Secondary Test F1-Score: " .. f1_avg)
  output_file:write("\n")
  output_file:write("Average Secondary Test MCC-Score: " .. mcc_avg)
  output_file:write("\n")
  output_file:write("Average Secondary Test Acc: " .. score_avg)
  output_file:write("\n")
  summary_file:write("Test,Sec,"..config.dtype..","..config.folds..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
  --t1 = {} 
  --t2 = {}
  --t3 = {}
  --t1[config.langSec]=f1_avg
  --t2[config.langPri]=t1
  --t3[config.dtype]=f1_avg
end


io.close(output_file)
io.close(summary_file)


