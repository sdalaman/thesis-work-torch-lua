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
model_pathPri = path.."models/tanh"
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

modelDeTok="1/train.de-en.de.0.01.850.tanh.tok.64"
modelEnTok="1/english.all.tok.en-tu.0.01.1150.tanh.64.2"
modelFrTok="1/train.en-fr.fr.0.0001.450.tanh.tok.64"
modelTrTok="1/turkish.all.tok.en-tu.0.01.1150.tanh.64.2"

modelDeMorph="1/train.de-en.de.0.001.1850.tanh.tok.morph.64"
modelEnMorph="1/english.all.tok.morph.en-tu.0.001.750.tanh.64.2"
modelFrMorph="1/train.en-fr.fr.0.001.1900.tanh.tok.morph.64"
modelTrMorph="1/turkish.all.tok.morph.en-tu.0.001.750.tanh.64.2"

corpusEnTok="english.all.tok.en-tu.new"
corpusTrTok="turkish.all.tok.en-tu.new"
corpusEnMorph="english.all.tok.morph.en-tu.new"
corpusTrMorph="turkish.all.tok.morph.en-tu.new"

corpusDeTok="multi/train.de-en.de.tok.tanh"
corpusFrTok="multi/train.en-fr.fr.tok.tanh"
corpusDeMorph="multi/train.de-en.de.tok.morph.tanh"
corpusFrMorph="multi/train.en-fr.fr.tok.morph.tanh"

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


function map_data(data,vocab)
  docstable = {}
  for idx,item in pairs(data) do
    lines = stringx.split(item[2], '\n')
    doctensor = torch.Tensor(#lines,item[1])
    for lcnt = 1,#lines do
      line_words = stringx.split(lines[lcnt], sep)
      sample = torch.Tensor(item[1])
      line_words = padding(line_words,item[1])
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
  local pcnt = 0
  local ncnt = 0
  for f in lfs.dir(data_path..'/'..class..'/positive') do
    local text = file.read(data_path..'/'..class..'/positive/'..f)
    if (text ~= nil) then
      lines = stringx.split(text, '\n')
      pcnt = pcnt + 1
      local ln= {}
      for lcnt = 1,#lines do
        nm_words = #stringx.split(lines[lcnt],sep)
        table.insert(ln,nm_words) 
      end
      table.insert(pos_data, {math.max(unpack(ln)),text})
    end
    if pcnt == fnmax then
      break
    end
  end  
  for f in lfs.dir(data_path..'/'..class..'/negative') do
    local text = file.read(data_path..'/'..class..'/negative/'..f)
    if (text ~= nil) then
      ncnt = ncnt + 1
      local ln= {}
      lines = stringx.split(text, '\n')
      for lcnt = 1,#lines do
        nm_words = #stringx.split(lines[lcnt],sep)
        table.insert(ln,nm_words) 
      end
      table.insert(neg_data, {math.max(unpack(ln)),text})
      --if pcnt == ncnt and dtype == "train" then
        --break
      --end
    end
    if ncnt == fnmax then
      break
    end
  end
  local pos_mapped = map_data(pos_data,vocab)
  local neg_mapped = map_data(neg_data,vocab)
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
config.corpusPri = "" -- corpus file  for lang 1
config.corpusSec = "" -- corpus file  for lang 2
config.lookuptablePri = "" -- lookup table file  for lang 1
config.lookuptableSec = "" -- lookup table file  for lang 2
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
config.dtype = "" -- model file  
config.dsize = "all" -- data size 
config.batch_size = 10
config.langPri = "En"
config.langSec= "Tr"
config.win_size=2
config.seq_lenPri = 0
config.seq_lenSec = 0
config.mdlImp = "Tanh-1"     -- Tanh-1,Tanh-2,Add



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
cmd:option("-dsize", config.dsize)
cmd:option("-langPri", config.langPri)
cmd:option("-langSec", config.langSec)
cmd:option("-win_size", config.win_size)
cmd:option("-mdlImp", config.mdlImp)  
params = cmd:parse(arg)


--th cdlc-tanh.lua -vocab1 english.all.tok.en-tu -vocab2 turkish.all.tok.en-tu -lr 0.01 -lr_decay 1 -max_epoch 300 -threshold 5 -graphshow 100 -folds 10 -init 0.001 -test_per 1 -train_per 1 -train_test 0 -test_test_pri 1 -test_test_sec 1 -dtype tok -batch_size 10 -max_loop 2 -trmax 100 -tstmax 100 -lookuptable1  1/english.all.tok.en-tu.tanh -lookuptable2  1/turkish.all.tok.en-tu.tanh -lang1 En -lang2 Tr 

--th cdlc-tanh.lua -vocab1 english.all.tok.morph.en-tu -vocab2 turkish.all.tok.morph.en-tu -lr 0.01 -lr_decay 1 -max_epoch 300 -threshold 5 -graphshow 100 -folds 10 -init 0.001 -test_per 1 -train_per 1 -train_test 0 -test_test_pri 1 -test_test_sec 1 -dtype morph -batch_size 10 -max_loop 2 -trmax 100 -tstmax 100 -lookuptable1  2/turkish.all.tok.morph.en-tu.0.01.1550.tanh.256 -lookuptable2  2/turkish.all.tok.morph.en-tu.0.01.1550.tanh.256 -lang1 En -lang2 Tr 

--th cdlc-tanh.lua -lr 0.01 -lr_decay 1 -max_epoch 300 -threshold 5 -graphshow 100 -folds 10 -init 0.001 -test_per 1 -train_per 1 -train_test 0 -test_test_pri 1 -test_test_sec 1 -dtype tok -batch_size 10 -max_loop 2 -trmax 100 -tstmax 100 -langPri En -langSec Tr 



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
    config.corpusPri = corpusEnMorph
  end
  if config.langPri == "Tr" then
    config.train_data_pathPri = train_data_pathTrEnMorph -- 
    config.test_data_pathPri = test_data_pathTrEnMorph -- 
    config.modelPri = modelTrMorph
    config.corpusPri = corpusTrMorph
  end
  if config.langPri == "De" then
    config.train_data_pathPri = train_data_pathDeEnMorph -- 
    config.test_data_pathPri = test_data_pathDeEnMorph -- 
    config.modelPri = modelDeMorph
    config.corpusPri = corpusDeMorph
  end
  if config.langPri == "Fr" then
    config.train_data_pathPri = train_data_pathFrEnMorph -- 
    config.test_data_pathPri = test_data_pathFrEnMorph -- 
    config.modelPri = modelFrMorph
    config.corpusPri = corpusFrMorph
  end

  if config.langSec == "En" then
    config.train_data_pathSec = train_data_pathEnTrMorph -- 
    config.test_data_pathSec = test_data_pathEnTrMorph -- 
    config.modelSec = modelEnMorph
    config.corpusSec = corpusEnMorph
  end
  if config.langSec == "Tr" then
    config.train_data_pathSec = train_data_pathTrEnMorph -- 
    config.test_data_pathSec = test_data_pathTrEnMorph -- 
    config.modelSec = modelTrMorph
    config.corpusSec = corpusTrMorph
  end
  if config.langSec == "De" then
    config.train_data_pathSec = train_data_pathDeEnMorph -- 
    config.test_data_pathSec = test_data_pathDeEnMorph -- 
    config.modelSec = modelDeMorph
    config.corpusSec = corpusDeMorph
  end
  if config.langSec == "Fr" then
    config.train_data_pathSec = train_data_pathFrEnMorph -- 
    config.test_data_pathSec = test_data_pathFrEnMorph -- 
    config.modelSec = modelFrMorph
    config.corpusSec = corpusFrMorph
  end
  
else
  
  if config.langPri == "En" then
    config.train_data_pathPri = train_data_pathEnTrTok -- 
    config.test_data_pathPri = test_data_pathEnTrTok -- 
    config.modelPri = modelEnTok
    config.corpusPri = corpusEnTok
  end
  if config.langPri == "Tr" then
    config.train_data_pathPri = train_data_pathTrEnTok -- 
    config.test_data_pathPri = test_data_pathTrEnTok -- 
    config.modelPri = modelTrTok
    config.corpusPri = corpusTrTok
  end
  if config.langPri == "De" then
    config.train_data_pathPri = train_data_pathDeEnTok -- 
    config.test_data_pathPri = test_data_pathDeEnTok -- 
    config.modelPri = modelDeTok
    config.corpusPri = corpusDeTok
  end
  if config.langPri == "Fr" then
    config.train_data_pathPri = train_data_pathFrEnTok -- 
    config.test_data_pathPri = test_data_pathFrEnTok -- 
    config.modelPri = modelFrTok
    config.corpusPri = corpusFrTok
  end

  if config.langSec == "En" then
    config.train_data_pathSec = train_data_pathEnTrTok -- 
    config.test_data_pathSec = test_data_pathEnTrTok -- 
    config.modelSec = modelEnTok
    config.corpusSec = corpusEnTok
  end
  if config.langSec == "Tr" then
    config.train_data_pathSec = train_data_pathTrEnTok -- 
    config.test_data_pathSec = test_data_pathTrEnTok -- 
    config.modelSec = modelTrTok
    config.corpusSec = corpusTrTok
  end
  if config.langSec == "De" then
    config.train_data_pathSec = train_data_pathDeEnTok -- 
    config.test_data_pathSec = test_data_pathDeEnTok -- 
    config.modelSec = modelDeTok
    config.corpusSec = corpusDeTok
  end
  if config.langSec == "Fr" then
    config.train_data_pathSec = train_data_pathFrEnTok -- 
    config.test_data_pathSec = test_data_pathFrEnTok -- 
    config.modelSec = modelFrTok
    config.corpusSec = corpusFrTok
  end
end

config.lookuptablePri = config.modelPri
config.lookuptableSec = config.modelSec

lang_pair = config.langPri.."-"..config.langSec
model_pathPri = model_pathPri.."/"..config.dsize.."/"..config.langPri.."/"
model_pathSec = model_pathSec.."/"..config.dsize.."/"..config.langSec.."/"

config.modelPri = model_pathPri..config.modelPri..".model" --model file  for lang 1
config.modelSec = model_pathSec..config.modelSec..".model" -- model file  for lang 2
config.corpusPri = corpus_path..config.corpusPri..".corpus.tch" -- corpus file  for lang 1
config.corpusSec = corpus_path..config.corpusSec..".corpus.tch" -- corpus file  for lang 2

config.lookuptablePri = model_pathPri..config.lookuptablePri..".LT" -- lookup table file  for lang 1
config.lookuptableSec = model_pathSec..config.lookuptableSec..".LT" -- lookup table file  for lang 2


print("model data will be loaded\n")
print(config.lookuptablePri.." "..config.lookuptableSec.."\n")
--biLnModelFirst.modelPri = torch.load(modelPriFileName)
--biLnModelFirst.modelSec = torch.load(modelSecFileName)
ltPri = torch.load(config.lookuptablePri)
ltSec = torch.load(config.lookuptableSec)
--ltPri.weight:copy(ltPriTemp.weight)
--ltSec.weight:copy(ltSecTemp.weight)

--biLnModel.modelPri:clearState()  -- Clear intermediate module states 
--biLnModel.modelSec:clearState()  -- Clear intermediate module states 
ltPri:cuda()
ltSec:cuda()

--lp = modelPri:findModules('nn.LookupTableMaskZero')
config.emb_size = ltPri:parameters()[1]:size()[2]

local Corpus = torch.class("Corpus")

local corpusPri = torch.load(config.corpusPri)
local corpusSec = torch.load(config.corpusSec)
local vocabPri = corpusPri.vocab_map
local vocabSec = corpusSec.vocab_map


trainDataSec = config.train_test == 1
tstDataPri = config.test_test_pri == 1
tstDataSec = config.test_test_sec == 1

file_timestamp = os.date("%Y_%m_%d_%X")
--output_file = io.open("f1scorevslr_"..nm.."-"..config.dtype.."-"..config.dsize.."-"..file_timestamp.."-newtotal-full.csv", "w")
config.out_file = "Tanh".."-"..config.dtype.."-"..lang_pair
output_file = io.open(config.out_file.."-"..file_timestamp..".csv", "w")
output_file:write("Program args : \n")
summary_file = io.open("summary-tanh"..".csv", "a+")
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


function createTempModelTanh(lt,seq_len,config)
  local vocab_size = lt.weight:size()[1] -1
  local ltM = nn.LookupTableMaskZero(vocab_size,config.emb_size) -- MaskZero
  ltM.weight:copy(lt.weight)
  local model = nn.Sequential()
  model:add(nn.Sequencer(ltM))
  model:add( nn.SplitTable(2))
  mod = nn.ConcatTable()
  for i = 1, seq_len-config.win_size+1, 1 do -- seq length
    rec = nn.NarrowTable(i,config.win_size)
    mod:add(rec)
  end
  model:add(mod)
  add_tan = nn:Sequential()
  add_tan:add(nn.Sequencer(nn.CAddTable()))
  add_tan:add(nn.Sequencer(nn.Tanh()))
  add_tan:add(nn.CAddTable()) -- add linear layer
  model:add(add_tan)
  model:cuda()
  return model
end

function createTempModelAdd(lt,seq_len,config)
  local vocab_size = lt.weight:size()[1] -1
  local ltM = nn.LookupTableMaskZero(vocab_size,config.emb_size) -- MaskZero
  ltM.weight:copy(lt.weight)
  local model = nn.Sequential()
  model:add( nn.SplitTable(2))
  model:add(nn.Sequencer(ltM))
  model:add( nn.CAddTable())
  model:cuda()
  return model
end

function prepareData(lt,className,vocab,dataType,cfg)
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
  
  positiveTableT, negativeTableT,pc,nc = getData(data_path,className,vocab,dataType,fnmax)
  --modelLn = createTempModel(lt.weight:size(1)-1,lt,seq_len,config)
  --modelLn = BiLangModel(ltPri.weight:size(1)-1,ltSec.weight:size(1)-1,config)
  --modelLn.ltPri.weight:copy(ltPri.weight)
  --modelLn.ltSec.weight:copy(ltSec.weight)
  --if modelLeg == "Pri" then
--    model = modelLn:getAdditivePri()
  --else
--    model = modelLn:getAdditiveSec()
--  end
  
  print("model data will be loaded\n")

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
  local modelLn =  {}
  for i = 1,#positiveTable do
    if cfg.mdlImp == "Tanh-1" then 
      modelLn = createTempModelTanh(lt,positiveTable[i]:size()[2],cfg)
    end
    if cfg.mdlImp == "Add" then 
      modelLn = createTempModelAdd(lt,positiveTable[i]:size()[2],cfg)
    end
    local t = modelLn:forward(positiveTable[i]:cuda())
    positive[i]:copy(torch.sum(t,1))
  end
  for i = 1,#negativeTable do
    if cfg.mdlImp == "Tanh-1" then 
      modelLn = createTempModelTanh(lt,negativeTable[i]:size()[2],cfg)
    end
    if cfg.mdlImp == "Add" then 
      modelLn = createTempModelAdd(lt,negativeTable[i]:size()[2],cfg)
    end
    local t = modelLn:forward(negativeTable[i]:cuda())
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
--score_loggerF1All = optim.Logger('cdlc-tanh-all-'..config.dtype.."-"..config.dsize.."-"..lang_pair..'-f1-log',true,"all-f1-"..config.dtype.."-"..config.dsize.."-"..lang_pair)  
--score_loggerAccAll = optim.Logger('cdlc-tanh-all-'..config.dtype.."-"..config.dsize.."-"..lang_pair..'-acc-log',true,"all-acc-"..config.dtype.."-"..config.dsize.."-"..lang_pair)  
--score_loggerMccAll = optim.Logger('cdlc-tanh-all-'..config.dtype.."-"..config.dsize.."-"..lang_pair..'-mcc-log',true,"all-mcc-"..config.dtype.."-"..config.dsize.."-"..lang_pair) 

for _,class in ipairs(classes) do
  print("\n--------------------\n")
  print("\nClass "..class.." "..lang_pair)
  print("..."..os.date("%Y_%m_%d_%X"))
  allPri,targetsPri,pc,nc = prepareData(ltPri,class,vocabPri,dataTypeTrainPri,config)
  print("positive train example : "..pc.." - negative train example : "..nc)

  allPri:cuda()
  targetsPri:cuda()

  --score_loggerTrSec = optim.Logger('./log/cdlc-tanh-train-sec-'..class..'-'..config.dtype.."-"..config.dsize.."-"..config.out_file..'-score-log',true,class.."-".."train-sec-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  --score_loggerTstPri = optim.Logger('./log/cdlc-tanh-test-pri-'..class..'-'..config.dtype.."-"..config.dsize.."-"..config.out_file..'-score-log',true,class.."-".."test-pri-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  score_loggerTstSec = optim.Logger('./log/cdlc-tanh-test-sec-'..class..'-'..config.dtype.."-"..config.dsize.."-"..config.out_file..'-score-log',true,class.."-".."test-sec-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  

  local mlpClassifier = nn.Sequential()
  mlpClassifier:add(nn.Linear(config.emb_size, config.emb_size))
  mlpClassifier:add(nn.Tanh())
  mlpClassifier:add(nn.Linear(config.emb_size, 1))
  mlpClassifier:add(nn.Sigmoid())
  criterion=nn.BCECriterion():cuda()
  mlpClassifier:cuda()
  
  --local mlpClassifier = nn.Sequential()
  --mlpClassifier:add(nn.Linear(config.emb_size, 1))
  --mlpClassifier:add(nn.Sigmoid())
  --criterion=nn.BCECriterion():cuda()
  --mlpClassifier:cuda()

  mlpClassifier:getParameters():uniform(-1*config.init,config.init)
  
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
  loss_logger_train = optim.Logger('./log/cdlc-tanh-'..class.."-"..config.dtype.."-"..config.dsize.."-"..config.out_file..'-train-loss-log',true,class.."-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  loss_logger_test = optim.Logger('./log/cdlc-tanh-'..class.."-"..config.dtype.."-"..config.dsize..'-test-loss-log',true,class.."-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  

  loopcnt = 1
  totalfoldcnt = 0
  local errorsTrain = {}  
  local errorsTest = {}  
  while true do
  for fold = 1, config.folds do
    totalfoldcnt = totalfoldcnt + 1
    mlpClassifier:training()
    print("fold begin..."..os.date("%Y_%m_%d_%X"))
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
        err = gradUpdate(mlpClassifier, x, y, criterion, config.lr)
        table.insert(errorsTrain, err)
      end
      if epoch % config.threshold == 0 then 
        config.lr = config.lr / config.lr_decay 
      end
      if epoch % config.graphshow == 0 then 
        if log_flag == true then
          loss_logger_train:plot()
        end
      end
      local mn = torch.mean( torch.Tensor( errorsTrain))
      loss_logger_train:add{['training error mean'] = mn}
      loss_logger_train:style{['training error mean'] = '+-'}
    end -- end for mlp train loop
    if log_flag == true then
          loss_logger_train:plot()
    end
    print(class.." fold :"..fold.." - fold loop cnt : "..loopcnt.." model trained ")
    print("fold end..."..os.date("%Y_%m_%d_%X"))
    
    mlpClassifier:evaluate()
    if trainDataSec == true then
    -- SECONDARY TRAIN DATA ACC 
      if trainDataLoaded == false then
        allTrSec,targetsTrSec,pc,nc = prepareData(ltSec,class,vocabSec,dataTypeTrainSec,config)
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
        mcc = 0
        if tmp == 0 then
          print("mcc is Zero : "..tmp)
          mcc = 0
        else
          mcc = (true_positivesTrSec * true_negativesTrSec - false_positivesTrSec * false_negativesTrSec)/math.sqrt(tmp)
        end
        mcc_accTrSec = mcc_accTrSec + mcc
        f1_score = 0
        if (precisionTrSec+recallTrSec) ~=  0 then
          f1_score = (2 * precisionTrSec * recallTrSec / (precisionTrSec + recallTrSec))
          f1_scoreTrSec = f1_scoreTrSec + f1_score
        else
          print(class.." Train Fold "..fold.." F1-Score: 0 ")
        end
        scoreTrSec = scoreTrSec + correctTrSec / allTrSec:size(1)
        print("fr score tr sec "..f1_score)
        --score_loggerTrSec:add{['f1-score'] = f1_score,['score'] = correctTr / allTrSec:size(1),['mcc-score'] = mcc}
        --score_loggerTrSec:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
        
      else 
        print("predicted_positivesTr :"..predicted_positivesTrSec.." - all_positivesTr :"..all_positivesTrSec)         
        fold = fold - 1
      end
      collectgarbage()
    end
  
    if tstDataPri == true then
      -- PRIMARY TEST DATA ACC 
      if testDataLoadedPri == false then
        allTstPri,targetsTstPri,pc,nc = prepareData(ltPri,class,vocabPri,dataTypeTestPri,config)
        allTstPri:cuda()
        targetsTstPri:cuda()
        print("positive test example : "..pc.." - negative test example : "..nc)
        testDataLoadedPri = true
      else
        print("primary test data examples already loaded ")
      end
    
      correctTstPri = 0
      predicted_positivesTstPri = 0
      true_positivesTstPri = 0
      true_negativesTstPri = 0
      false_positivesTstPri = 0
      false_negativesTstPri = 0
      all_positivesTstPri = 0
      mlpClassifier:evaluate()
      print("Primary Test Size "..allTstPri:size(1))
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
        mcc = 0
        if tmp == 0 then
          print("mcc is Zero : "..tmp)
          mcc = 0
        else
          mcc = (true_positivesTstPri * true_negativesTstPri - false_positivesTstPri * false_negativesTstPri)/math.sqrt(tmp)
        end
        
        mcc_accTstPri = mcc_accTstPri + mcc
        
        f1_score = 0
        if (precisionTstPri+recallTstPri) ~= 0 then
          f1_score = (2 * precisionTstPri * recallTstPri / (precisionTstPri+recallTstPri))
          f1_scoreTstPri = f1_scoreTstPri + f1_score
        else
          print(class.." Test Fold "..fold.." F1-Score: 0 ")
        end
        scoreTstPri = scoreTstPri + correctTstPri / allTstPri:size(1)
        print("fr score tst pri "..f1_score)
        --score_loggerTstPri:add{['f1-score'] = f1_score,['score'] = correctTstPri / allTstPri:size(1),['mcc-score'] = mcc}
        --score_loggerTstPri:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
        
      else 
        print("predicted_positivesTstPri :"..predicted_positivesTstPri.." - all_positivesTstPri :"..all_positivesTstPri)         
        fold = fold - 1
      end
      collectgarbage()
    end
  
  
    if tstDataSec == true then
      -- SECONDARY TEST DATA ACC 
      if testDataLoadedSec == false then
        allTstSec,targetsTstSec,pc,nc = prepareData(ltSec,class,vocabSec,dataTypeTestSec,config)
        allTstSec:cuda()
        targetsTstSec:cuda()
        print("positive test example : "..pc.." - negative test example : "..nc)
        testDataLoadedSec = true
      else
        print("secondary test data examples already loaded ")
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
      print("Secondary Test Size "..allTstSec:size(1))
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
        mcc = 0
        if tmp == 0 then
          print("mcc is Zero : "..tmp)
          mcc = 0
        else
          mcc = (true_positivesTstSec * true_negativesTstSec - false_positivesTstSec * false_negativesTstSec)/math.sqrt(tmp)
        end
        
        mcc_accTstSec = mcc_accTstSec + mcc
        
        f1_score = 0
        if (precisionTstSec+recallTstSec) ~= 0 then
          f1_score = (2 * precisionTstSec * recallTstSec / (precisionTstSec+recallTstSec))
          f1_scoreTstSec = f1_scoreTstSec + f1_score
        else
          print(class.." Test Fold "..fold.." F1-Score: 0 ")
        end
        scoreTstSec = scoreTstSec + correctTstSec / allTstSec:size(1)
        print("fr score tst sec "..f1_score)
        score_loggerTstSec:add{['f1-score'] = f1_score,['score'] = correctTstSec / allTstSec:size(1),['mcc-score'] = mcc}
        score_loggerTstSec:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
        --score_loggerTstSec:plot()
      else 
        print("predicted_positivesTstSec :"..predicted_positivesTstSec.." - all_positivesTstSec :"..all_positivesTstSec)         
        fold = fold - 1
      end
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

  print("fold cnt "..totalfoldcnt.."\n")

  if trainDataSec == true then
    print("Class: "..class.." "..lang_pair)
    print("Secondary Train Score: " .. (scoreTrSec / totalfoldcnt) * 100 .. "%")
    print("Secondary Train Precision: " .. precision_accTrSec / totalfoldcnt)
    print("Secondary Train Recall: " .. recall_accTrSec / totalfoldcnt)
    print("Secondary Train F1-Score: " .. f1_scoreTrSec / totalfoldcnt)
    print("Secondary Train MCC-Score: " .. mcc_accTrSec / totalfoldcnt)
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
  --score_loggerF1All:add{['f1-Train'] = f1_score_avgTrSec / class_loop_cnt,['f1-TstPri'] = f1_score_avgTstPri / class_loop_cnt,['f1-TstSec'] = f1_score_avgTstSec / class_loop_cnt}
  --score_loggerF1All:style{['f1-Train'] = '+-',['f1-TstPri'] = '+-',['f1-TstSec'] = '+-'}
  --score_loggerF1All:plot()
  
  --score_loggerAccAll:add{['acc-Train'] = score_avgTrSec / class_loop_cnt,['acc-TstPri'] = score_avgTstPri / class_loop_cnt,['acc-TstSec'] = score_avgTstSec / class_loop_cnt}
  --score_loggerAccAll:style{['acc-Train'] = '+-',['acc-TstPri'] = '+-',['acc-TstSec'] = '+-'}
  --score_loggerAccAll:plot()
  
  --score_loggerMccAll:add{['mcc-Train'] = mcc_acc_avgTrSec / class_loop_cnt,['mcc-TstPri'] = mcc_acc_avgTstPri / class_loop_cnt,['mcc-TstSec'] = mcc_acc_avgTstSec / class_loop_cnt}
  --score_loggerMccAll:style{['mcc-Train'] = '+-',['mcc-TstPri'] = '+-',['mcc-TstSec'] = '+-'}
  --score_loggerMccAll:plot()
  
  collectgarbage()
end -- End for class loop

--summary_file:write("\n-----------------------------------\n")
for param, value in pairs(params) do
  ----summary_file:write(param.." : "..value..", ")
end
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
  summary_file:write("Train,Sec,"..config.dtype..","..config.folds..","..config.win_size..","..config.mdlImp..","..config.tstmax..","..config.langPri..","..config.langSec..", "..f1_avg..","..mcc_avg..","..score_avg.."\n")
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
  summary_file:write("Test,Pri,"..config.dtype..","..config.folds..","..config.win_size..","..config.mdlImp..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
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
  summary_file:write("Test,Sec,"..config.dtype..","..config.folds..","..config.win_size..","..config.mdlImp..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
end

io.close(output_file)
io.close(summary_file)


