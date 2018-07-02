math.randomseed(1)

require 'nn'
require 'cunn'
require 'rnn'
require 'lfs'
require 'utils'
require 'optim'
require 'io'
require 'os'
require 'paths'

log_flag = false

local stringx = require('pl.stringx')
local file = require('pl.file')
--local data_path = "./data_tok/cdlc_en_tr/"

test_doc_data_path="./samples/"
path = "/home/saban/work/additive/"
path2 = "/home/saban/work/python/pytorch-works/additive/data/ted/"
model_pathPri = path.."models/lstm"
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


vocabEnTokLstm="english.10000.tok.en-tu.60.lstm"
vocabTrTokLstm="turkish.10000.tok.en-tu.60.lstm"
vocabEnMorphLstm="english.10000.tok.morph.en-tu.60.lstm"
vocabTrMorphLstm="turkish.10000.tok.morph.en-tu.60.lstm"

vocabEnTokBiLstm="english.10000.tok.en-tu.60.BiLstm"
vocabTrTokBiLstm="turkish.10000.tok.en-tu.60.BiLstm"
vocabEnMorphBiLstm="english.10000.tok.morph.en-tu.60.BiLstm"
vocabTrMorphBiLstm="turkish.10000.tok.morph.en-tu.60.BiLstm"

vocabDeTokLstm="multi/de-en.de.10000.60.lstm"
vocabFrTokLstm="multi/en-fr.fr.10000.60.lstm"
vocabDeMorphLstm="multi/de-en.de.10000.morph.60.lstm"
vocabFrMorphLstm="multi/en-fr.fr.10000.morph.60.lstm"

vocabDeTokBiLstm="multi/de-en.de.10000.60.Bilstm"
vocabFrTokBiLstm="multi/en-fr.fr.10000.60.Bilstm"
vocabDeMorphBiLstm="multi/de-en.de.10000.morph.60.Bilstm"
vocabFrMorphBiLstm="multi/en-fr.fr.10000.morph.60.Bilstm"

modelDeTokBiLstm ="1-Bilstm-Sc-Avg(64-128)-Tok/BiLstmSc-Avg-drop-de-en.en.10000_0.001_1250.lstm.64.128.60"
modelDeTokLstm   ="1-Sc-Avg-Tok/lstm-ScAvg-drop-de-en.en.10000_0.01_50.lstm.128.128.60"
modelEnTokBiLstm ="1-Bilstm-Sc-Avg(64-128)-Tok/en-tu_en_0.001_750.lstm.64.128.60.BiLstmSc-Avg-drop-10000"
modelEnTokLstm   ="1-Sc-Avg-Tok/en-tu_en_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000"
modelFrTokBiLstm ="1-Bilstm-Sc-Avg(64-128)-Tok/BiLstmSc-Avg-drop-en-fr.en.10000_0.001_1050.lstm.64.128.60"
modelFrTokLstm   ="1-Sc-Avg-Tok/lstm-ScAvg-drop-en-fr.en.10000_0.01_50.lstm.128.128.60"
modelTrTokBiLstm ="1-Bilstm-Sc-Avg(64-128)-Tok/en-tu_tu_0.001_750.lstm.64.128.60.BiLstmSc-Avg-drop-10000"
modelTrTokLstm   ="1-Sc-Avg-Tok/en-tu_tu_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000"


modelDeMorphBiLstm ="1-Bilstm-Sc-Avg(64-128)-Morph/BiLstmSc-Avg-drop-de-en.en.10000-morph_0.0001_1600.lstm.64.128.60"
modelDeMorphLstm   ="1-Sc-Avg-Morph/lstm-ScAvg-drop-de-en.en.10000-morph_0.01_50.lstm.128.128.60"
modelEnMorphBiLstm ="1-Bilstm-Sc-Avg(64-128)-Morph/en-tu_en_1e-05_1400.lstm.64.128.60.BiLstmSc-Avg-drop-10000-morph"
modelEnMorphLstm   ="1-Sc-Avg-Morph/en-tu_en_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000-morph"
modelFrMorphBiLstm ="1-Bilstm-Sc-Avg(64-128)-Morph/BiLstmSc-Avg-drop-en-fr.en.10000-morph_0.0001_1250.lstm.64.128.60"
modelFrMorphLstm   ="1-Sc-Avg-Morph/lstm-ScAvg-drop-en-fr.en.10000-morph_0.01_50.lstm.128.128.60"
modelTrMorphBiLstm ="1-Bilstm-Sc-Avg(64-128)-Morph/en-tu_tu_1e-05_1400.lstm.64.128.60.BiLstmSc-Avg-drop-10000-morph"
modelTrMorphLstm   ="1-Sc-Avg-Morph/en-tu_tu_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000-morph"

function shuffle_data(x, y)
  local new_x = x
  local new_y = y
  local rand = torch.totable(torch.randperm(x:size(1)))
  for i = 1 , x:size(1) do
    new_x[i] = x[rand[i]]
    new_y[i] = y[rand[i]]
  end
  return new_x, new_y
end

norm,sign= torch.norm,torch.sign

function gradUpdateOld(mlp, x, y, criterion, learningRate,dl_dx)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   return err,torch.sum(torch.pow(dl_dx[1],2))
end

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   if true then -- Method 1
    local paramsTns, gradParamsTns = mlp:getParameters()
    local norm,sign= torch.norm,torch.sign
    err = err + config.coefL1 * norm(paramsTns,1)
    err = err + config.coefL2 * norm(paramsTns,2)^2/2
    addGrad = sign(paramsTns):mul(config.coefL1) + paramsTns:clone():mul(config.coefL2) 
    gradParamsTns:add(addGrad)
    mlp:updateParameters(learningRate)
   end
   if false then -- Method 2
    local parameters, _ = network:parameters()
    local penalty = 0
    for i=1, table.getn(parameters) do
      penalty = penalty + config.coefL1 * parameters[i]:norm(1) + config.coefL2 * parameters[i]:norm(2) ^ 2
      local update = torch.clamp(parameters[i], -config.coefL1, config.coefL1)
      update:add(parameters[i]:mul(-config.coefL2))
      parameters[i]:csub(update)
    end
    err = err + penalty
   end
   local _, dl_dx= mlp:parameters()
   return err,torch.sum(torch.pow(dl_dx[1],2))
end

function paddingOld(sequence,longest_seq)
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

function paddingOld(sequence,longest_seq)
  new_sequence = {}
  mn = math.min(longest_seq,#sequence)
  for i = 1 , mn do
    new_sequence[i] = sequence[i]
  end
  for i = mn+1, longest_seq do
    new_sequence[i] = '<pad>'
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

function paddingLSTM(sequence,longest_seq)
  -- if extracting word embeddings! 
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


function map_data(data,vocab,seq_len)
  docstable = {}
  for idx,item in ipairs(data) do
    lines = stringx.split(item[2], '\n')
    doctensor = torch.Tensor(#lines,seq_len)
    for lcnt = 1,#lines do
      line_words = stringx.split(lines[lcnt], sep)
      sample = torch.Tensor(seq_len)
      line_words = paddingLSTM(line_words,seq_len)
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

function getData(data_path,class,vocab,dtype,fnmax,seq_len)
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
      pos_data[pcnt] = {math.max(unpack(ln)),text}
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
      neg_data[ncnt] = {math.max(unpack(ln)),text}
      if pcnt == ncnt and dtype == "train" then
      --  break
      end
    end
    if ncnt == fnmax then
      break
    end
  end
  local pos_mapped = map_data(pos_data,vocab,seq_len)
  local neg_mapped = map_data(neg_data,vocab,seq_len)
  return pos_mapped,neg_mapped,pcnt,ncnt
end

sep = '\t'
--local classes = {'arts'}
local classes = {'art','arts','biology','business','creativity','culture','design','economics','education','entertainment','global','health','politics','science','technology'}

--local classes = {'science','technology'}

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
config.hidden_size = 0
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
config.test_test_pri = 0   -- test with test data
config.test_test_sec = 0   -- test with test data
config.trmax = 10   -- max train doc
config.tstmax = 10  -- max test doc
config.seq_len = 60
config.out_file = "" 
config.inp_file_tag = "" 
config.dtype = "" -- model file  
config.dsize = "" -- data size 
config.datasave = 0 
config.batch_size = 10
config.coefL1 = 0
config.coefL2 = 0
config.langPri = "En"
config.langSec= "Tr"
config.lstmtype = "Lstm"

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-lr", config.lr)
cmd:option("-coefL1", config.coefL1)
cmd:option("-coefL2", config.coefL2)
cmd:option("-lr_decay", config.lr_decay)
cmd:option("-max_epoch", config.max_epoch)
cmd:option("-max_loop", config.max_loop)
cmd:option("-batch_size", config.batch_size)
cmd:option("-hidden_size", config.hidden_size)
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
cmd:option("-seq_len", config.seq_len)
cmd:option("-trmax", config.trmax)
cmd:option("-tstmax", config.tstmax)
cmd:option("-dtype", config.dtype)
cmd:option("-dsize", config.dsize)
cmd:option("-datasave", config.datasave)
cmd:option("-langPri", config.langPri)
cmd:option("-langSec", config.langSec)
cmd:option("-lstmtype", config.lstmtype)
params = cmd:parse(arg)


-- th cdlc-lstm.lua -lr 1 -lr_decay 1 -datasave 1 -max_epoch 300 -threshold 5 -graphshow 100 -folds 10 -init 0.00001 -lstmtype Lstm -test_per 1 -train_per 1 -train_test 0 -test_test_pri 1 -test_test_sec 1 -dtype tok -dsize 10000 -batch_size 10 -seq_len 60 -max_loop 2 -trmax 100 -tstmax 100 -coefL1 0 -coefL2 0 -langPri En -langSec Tr 


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
    if config.lstmtype == "Lstm" then
      config.modelPri = modelEnMorphLstm
      config.vocabPri = vocabEnMorphLstm
    else
      config.modelPri = modelEnMorphBiLstm
      config.vocabPri = vocabEnMorphBiLstm
    end
  end
  if config.langPri == "Tr" then
    config.train_data_pathPri = train_data_pathTrEnMorph -- 
    config.test_data_pathPri = test_data_pathTrEnMorph -- 
    if config.lstmtype == "Lstm" then
      config.modelPri = modelTrMorphLstm
      config.vocabPri = vocabTrMorphLstm
    else
      config.modelPri = modelTrMorphBiLstm
      config.vocabPri = vocabTrMorphBiLstm
    end
  end
  if config.langPri == "De" then
    config.train_data_pathPri = train_data_pathDeEnMorph -- 
    config.test_data_pathPri = test_data_pathDeEnMorph -- 
    if config.lstmtype == "Lstm" then
      config.modelPri = modelDeMorphLstm
      config.vocabPri = vocabDeMorphLstm
    else
      config.modelPri = modelDeMorphBiLstm
      config.vocabPri = vocabDeMorphBiLstm
    end
  end
  if config.langPri == "Fr" then
    config.train_data_pathPri = train_data_pathFrEnMorph -- 
    config.test_data_pathPri = test_data_pathFrEnMorph -- 
    if config.lstmtype == "Lstm" then
      config.modelPri = modelFrMorphLstm
      config.vocabPri = vocabFrMorphLstm
    else
      config.modelPri = modelFrMorphBiLstm
      config.vocabPri = vocabFrMorphBiLstm
    end
  end

  if config.langSec == "En" then
    config.train_data_pathSec = train_data_pathEnTrMorph -- 
    config.test_data_pathSec = test_data_pathEnTrMorph -- 
    if config.lstmtype == "Lstm" then
      config.modelSec = modelEnMorphLstm
      config.vocabSec = vocabEnMorphLstm
    else
      config.modelSec = modelEnMorphBiLstm
      config.vocabSec = vocabEnMorphBiLstm
    end
  end
  if config.langSec == "Tr" then
    config.train_data_pathSec = train_data_pathTrEnMorph -- 
    config.test_data_pathSec = test_data_pathTrEnMorph -- 
    if config.lstmtype == "Lstm" then
      config.modelSec = modelTrMorphLstm
      config.vocabSec = vocabTrMorphLstm
    else
      config.modelSec = modelTrMorphBiLstm
      config.vocabSec = vocabTrMorphBiLstm
    end
  end
  if config.langSec == "De" then
    config.train_data_pathSec = train_data_pathDeEnMorph -- 
    config.test_data_pathSec = test_data_pathDeEnMorph -- 
    if config.lstmtype == "Lstm" then
      config.modelSec = modelDeMorphLstm
      config.vocabSec = vocabDeMorphLstm
    else
      config.modelSec = modelDeMorphBiLstm
      config.vocabSec = vocabDeMorphBiLstm
    end
  end
  if config.langSec == "Fr" then
    config.train_data_pathSec = train_data_pathFrEnMorph -- 
    config.test_data_pathSec = test_data_pathFrEnMorph -- 
    if config.lstmtype == "Lstm" then
      config.modelSec = modelFrMorphLstm
      config.vocabSec = vocabFrMorphLstm
    else
      config.modelSec = modelFrMorphBiLstm
      config.vocabSec = vocabFrMorphBiLstm
    end
  end
  
else
  
  if config.langPri == "En" then
    config.train_data_pathPri = train_data_pathEnTrTok
    config.test_data_pathPri = test_data_pathEnTrTok -- 
    if config.lstmtype == "Lstm" then
      config.modelPri = modelEnTokLstm
      config.vocabPri = vocabEnTokLstm
    else
      config.modelPri = modelEnTokBiLstm
      config.vocabPri = vocabEnTokBiLstm
    end
  end
  if config.langPri == "Tr" then
    config.train_data_pathPri = train_data_pathTrEnTok -- 
    config.test_data_pathPri = test_data_pathTrEnTok -- 
    if config.lstmtype == "Lstm" then
      config.modelPri = modelTrTokLstm
      config.vocabPri = vocabTrTokLstm
    else
      config.modelPri = modelTrTokBiLstm
      config.vocabPri = vocabTrTokBiLstm
    end
  end
  if config.langPri == "De" then
    config.train_data_pathPri = train_data_pathDeEnTok -- 
    config.test_data_pathPri = test_data_pathDeEnTok -- 
    if config.lstmtype == "Lstm" then
      config.modelPri = modelDeTokLstm
      config.vocabPri = vocabDeTokLstm
    else
      config.modelPri = modelDeTokBiLstm
      config.vocabPri = vocabDeTokBiLstm
    end
  end
  if config.langPri == "Fr" then
    config.train_data_pathPri = train_data_pathFrEnTok -- 
    config.test_data_pathPri = test_data_pathFrEnTok -- 
    if config.lstmtype == "Lstm" then
      config.modelPri = modelFrTokLstm
      config.vocabPri = vocabFrTokLstm
    else
      config.modelPri = modelFrTokBiLstm
      config.vocabPri = vocabFrTokBiLstm
    end
  end

  if config.langSec == "En" then
    config.train_data_pathSec = train_data_pathEnTrTok -- 
    config.test_data_pathSec = test_data_pathEnTrTok -- 
    if config.lstmtype == "Lstm" then
      config.modelSec = modelEnTokLstm
      config.vocabSec = vocabEnTokLstm
    else
      config.modelSec = modelEnTokBiLstm
      config.vocabSec = vocabEnTokBiLstm
    end
  end
  if config.langSec == "Tr" then
    config.train_data_pathSec = train_data_pathTrEnTok -- 
    config.test_data_pathSec = test_data_pathTrEnTok -- 
    if config.lstmtype == "Lstm" then
      config.modelSec = modelTrTokLstm
      config.vocabSec = vocabTrTokLstm
    else
      config.modelSec = modelTrTokBiLstm
      config.vocabSec = vocabTrTokBiLstm
    end
  end
  if config.langSec == "De" then
    config.train_data_pathSec = train_data_pathDeEnTok -- 
    config.test_data_pathSec = test_data_pathDeEnTok -- 
    if config.lstmtype == "Lstm" then
      config.modelSec = modelDeTokLstm
      config.vocabSec = vocabDeTokLstm
    else
      config.modelSec = modelDeTokBiLstm
      config.vocabSec = vocabDeTokBiLstm
    end
  end
  if config.langSec == "Fr" then
    config.train_data_pathSec = train_data_pathFrEnTok -- 
    config.test_data_pathSec = test_data_pathFrEnTok -- 
    if config.lstmtype == "Lstm" then
      config.modelSec = modelFrTokLstm
      config.vocabSec = vocabFrTokLstm
    else
      config.modelSec = modelFrTokBiLstm
      config.vocabSec = vocabFrTokBiLstm
    end
  end
end

lang_pair = config.langPri.."-"..config.langSec
model_pathPri = model_pathPri.."/"..config.dsize.."/"..config.langPri.."/"
model_pathSec = model_pathSec.."/"..config.dsize.."/"..config.langSec.."/"

lookupPri = model_pathPri..config.modelPri..".LT" 
config.modelPri = model_pathPri..config.modelPri..".model" --model file  for lang 1
config.modelSec = model_pathSec..config.modelSec..".model" -- model file  for lang 2
config.vocabPri = corpus_path..config.vocabPri..".vocab" -- vocab file  for lang 1
config.vocabSec = corpus_path..config.vocabSec..".vocab" -- vocab file  for lang 2

for i,j in pairs(config) do
    --print(i..": "..j)
end

ltPri = torch.load(lookupPri)
--ltSec = torch.load(config.lookuptable2)
--ltPri:cuda()
--ltSec:cuda()


config.emb_size = ltPri:parameters()[1]:size()[2]
ltPri = {}

if config.lstmtype == "Lstm" then
    config.vector_size= config.emb_size
  else
    config.vector_size= 2*config.emb_size
end

print("Word vector dimension : "..config.vector_size)
trainDataPri = config.train_test == 1
trainDataSec = config.train_test == 1
tstDataPri = config.test_test_pri == 1
tstDataSec = config.test_test_sec == 1

file_timestamp = os.date("%Y_%m_%d_%X")
--output_file = io.open("f1scorevslr_"..nm.."-"..config.dtype.."-"..config.dsize.."-"..file_timestamp.."-newtotal-full.csv", "w")
config.out_file = config.lstmtype.."-"..config.dtype.."-"..lang_pair
output_file = io.open(config.out_file.."-"..file_timestamp..".csv", "w")
output_file:write("Program args : \n")
summary_file = io.open("summary-lstm"..".csv", "a+")
for param, value in pairs(params) do
  output_file:write(param.." : "..value..", ")
end
output_file:write("\n")
output_file:flush()

if trainDataPri == true then
  f1_score_avgTrPri = 0
  score_avgTrPri = 0
  mcc_acc_avgTrPri = 0
  fall_out_avgTrPri = 0
end

if trainDataSec == true then
  f1_score_avgTrSec = 0
  score_avgTrSec = 0
  mcc_acc_avgTrSec = 0
  fall_out_avgTrSec = 0
end

if tstDataPri == true then
  f1_score_avgTstPri = 0
  score_avgTstPri = 0
  mcc_acc_avgTstPri = 0
  fall_out_avgTstPri = 0
end

if tstDataSec == true then
  f1_score_avgTstSec = 0
  score_avgTstSec = 0
  mcc_acc_avgTstSec = 0
  fall_out_avgTstSec = 0
end

fdata = "full"

function normalize(x)
  return x
  --local ret = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
  --local ret = x / torch.norm(x)
  --return ret
end

function minmax(x)
  local mn = torch.min(x,1)
  local mx = torch.max(x,1)
  local ret = torch.cat(mn,mx)
  return ret
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
  --print("data path "..data_path)
  positiveTableT, negativeTableT,pc,nc = getData(data_path,className,vocab,dataType,fnmax,cfg.seq_len)
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
  
  local split = nn.SplitTable(2):cuda()
  local npositive = torch.zeros(#positiveTable,cfg.vector_size):double():cuda()
  --local targets = nn.JoinTable(1):forward{torch.Tensor(#positiveTable):fill(1), torch.Tensor(#negativeTable):fill(0)}
  for i = 1,#positiveTable do
    local input = split:forward(positiveTable[i]:cuda())
    local t = model:forward(input)
    local s = torch.sum(t,1)
    npositive[i]:copy(normalize(s))
  end
  positiveTable = {}
  local nnegative = torch.zeros(#negativeTable,cfg.vector_size):double():cuda()
  for i = 1,#negativeTable do
    local input = split:forward(negativeTable[i]:cuda())
    local t = model:forward(input)
    local s = torch.sum(t,1)
    nnegative[i]:copy(normalize(s))
  end
  negativeTable = {}
  
  if dataType == dataTypeTrainPri then
    diff = nnegative:size()[1] - npositive:size()[1]
    local posT = nnegative:clone():fill(0)
    math.randomseed( os.time() )
    for i=1,npositive:size()[1] do
      posT[i]:copy(npositive[i])
    end
    for i=1,diff do
      cnt = npositive:size()[1]+i
      ln = math.random(1,npositive:size()[1])
      posT[cnt]:copy(npositive[ln])
    end
    npositive = posT
  end
  local inputs = nn.JoinTable(1):forward{npositive, nnegative}
  local targets = nn.JoinTable(1):forward{torch.Tensor(npositive:size()[1]):fill(1), torch.Tensor(nnegative:size()[1]):fill(0)}
  return inputs,targets,pc,nc
end

function prepareDataFile(config,class,fdata)
  local modelPri = torch.load(config.modelPri):double()
  local modelSec = torch.load(config.modelSec):double()
  modelPri:clearState()  -- Clear intermediate module states 
  modelSec:clearState()  -- Clear intermediate module states 
  modelPri:cuda()
  modelSec:cuda()
  modelPri:evaluate()
  modelSec:evaluate()
  collectgarbage()

  local vocabPri = torch.load(config.vocabPri)
  local vocabSec = torch.load(config.vocabSec)
  
  --print("Lang pri "..config.langPri)
  --print("Lang sec "..config.langSec)
  --print("model pri "..config.modelPri)
  --print("model sec "..config.modelSec)
  --print("vocab pri "..config.vocabPri)
  --print("vocab sec "..config.vocabSec)
  
 
  doc_samples_tables_pri = test_doc_data_path..class.."."..config.dtype.."."..config.dsize.."."..config.lstmtype.."."..config.langPri
  doc_samples_tables_sec = test_doc_data_path..class.."."..config.dtype.."."..config.dsize.."."..config.lstmtype.."."..config.langSec
  
  if config.langPri == "En" and (config.langSec == "De" or config.langSec == "Fr") then
    doc_samples_tables_pri = doc_samples_tables_pri.."Multi"
  end
  if config.langSec == "En" and (config.langPri == "De" or config.langPri == "Fr") then
    doc_samples_tables_sec = doc_samples_tables_sec.."Multi"
  end
    
  print("Lang "..config.langPri)
  all,targets,pc,nc = prepareData(modelPri,class,vocabPri,dataTypeTrainPri,config)
  torch.save(doc_samples_tables_pri..".train.pri.input.pch",all)
  torch.save(doc_samples_tables_pri..".train.pri.targets.pch",targets)
  --    os.exit()
  all = {}
  print("positive primary train example : "..pc.." - negative primary train example : "..nc)
  
  print("Lang "..config.langSec)
  all,targets,pc,nc = prepareData(modelSec,class,vocabSec,dataTypeTrainSec,config)
  torch.save(doc_samples_tables_sec..".train.sec.input.pch",all)
  torch.save(doc_samples_tables_sec..".train.sec.targets.pch",targets)
  --os.exit()
  all = {}
  print("positive secondary train example : "..pc.." - negative example : "..nc)
  
  print("Lang "..config.langPri)
  all,targets,pc,nc = prepareData(modelPri,class,vocabPri,dataTypeTestPri,config)
  torch.save(doc_samples_tables_pri..".test.pri.input.pch",all)
  torch.save(doc_samples_tables_pri..".test.pri.targets.pch",targets)
  --os.exit()
  all = {}
  print("positive primary test example : "..pc.." - negative primary test example : "..nc)
  
  print("Lang "..config.langSec)
  all,targets,pc,nc = prepareData(modelSec,class,vocabSec,dataTypeTestSec,config)
  torch.save(doc_samples_tables_sec..".test.sec.input.pch",all)
  torch.save(doc_samples_tables_sec..".test.sec.targets.pch",targets)
  --os.exit()
  all = {}
  print("positive secondary test example : "..pc.." - negative secondary test example : "..nc)
  collectgarbage()
end

function shuffleTensor(x,y)
  ind = torch.randperm(x:size()[1])
  local xt = x:clone()
  local yt = y:clone()
  for i=1,x:size()[1] do
    xt[i] = x[ind[i]]
    yt[i] = y[ind[i]]
  end
  return xt,yt
end

lr = config.lr

class_loop_cnt = 0
--score_loggerTstPriAll = optim.Logger('./log/cdlc-lstm-test-all-pri-'..config.dtype.."-"..config.dsize.."-"..lang_pair..'-score-log',true,"test-all-pri-"..config.dtype.."-"..config.dsize.."-"..lang_pair)  
--score_loggerTstSecAll = optim.Logger('./log/cdlc-lstm-test-all-sec-'..config.dtype.."-"..config.dsize.."-"..lang_pair..'-score-log',true,"test-all-sec-"..config.dtype.."-"..config.dsize.."-"..lang_pair)  


if config.datasave == 1 then
  for _,class in ipairs(classes) do
    print("\n--------------------\n")
    print("\nClass "..class.." "..lang_pair)
    print("..."..os.date("%Y_%m_%d_%X"))
    prepareDataFile(config,class,fdata)
    print(class.." doc data vectors saved")
  end
  os.exit()
end

function selectvectors(x,y,pc,nc)
  local xt = torch.zeros(pc+nc,x:size()[2])
  local yt = torch.zeros(pc+nc)
  local p = 1
  local n = 1
  local lcnt = 1
  for i=1,x:size()[1] do
    if y[i] == 1 then
      if p <= pc then
        xt[lcnt] = x[i]
        yt[lcnt] = y[i]
        p = p + 1
        lcnt = lcnt + 1
      end
    else
      if n <= nc then
        xt[lcnt] = x[i]
        yt[lcnt] = y[i]
        n = n + 1
        lcnt = lcnt + 1
      end
    end
  end
  return xt,yt
end

for _,class in ipairs(classes) do
  config.lr = lr
  print("\n--------------------\n")
  print("\nClass "..class.." "..lang_pair)
  print("..."..os.date("%Y_%m_%d_%X"))
  pc = 0
  nc = 0
  
  doc_samples_tables_pri = test_doc_data_path..class.."."..config.dtype.."."..config.dsize.."."..config.lstmtype.."."..config.langPri
  doc_samples_tables_sec = test_doc_data_path..class.."."..config.dtype.."."..config.dsize.."."..config.lstmtype.."."..config.langSec
  
  if config.langPri == "En" and (config.langSec == "De" or config.langSec == "Fr") then
    doc_samples_tables_pri = doc_samples_tables_pri.."Multi"
  end
  if config.langSec == "En" and (config.langPri == "De" or config.langPri == "Fr") then
    doc_samples_tables_sec = doc_samples_tables_sec.."Multi"
  end
  
  print(doc_samples_tables_pri..".train.pri.input.pch".."  "..doc_samples_tables_pri..".train.pri.targets.pch")
  allPri=torch.load(doc_samples_tables_pri..".train.pri.input.pch")
  targetsPri=torch.load(doc_samples_tables_pri..".train.pri.targets.pch")
  pc = torch.sum(targetsPri)
  nc = targetsPri:size()[1] - torch.sum(targetsPri)
  print("positive train example : "..pc.." - negative train example : "..nc)
  allPri,targetsPri= selectvectors(allPri,targetsPri,pc,nc)
  local n = allPri:size()[1]
  allPri = allPri[{{1,n},{}}]:cuda()
  targetsPri = targetsPri[{{1,n}}]:cuda()
  pc = torch.sum(targetsPri)
  nc = targetsPri:size()[1] - torch.sum(targetsPri)
  print("positive train example : "..pc.." - negative train example : "..nc)

  score_loggerTrPri = optim.Logger('./log/cdlc-lstm-train-pri-'..class..'-'..config.dtype.."-"..config.dsize..'-score-log',true,class.."-".."train-pri-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  score_loggerTrSec = optim.Logger('./log/cdlc-lstm-train-sec-'..class..'-'..config.dtype.."-"..config.dsize..'-score-log',true,class.."-".."train-sec-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  score_loggerTstPri = optim.Logger('./log/cdlc-lstm-test-pri-'..class..'-'..config.dtype.."-"..config.dsize..'-score-log',true,class.."-".."test-pri-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  score_loggerTstSec = optim.Logger('./log/cdlc-lstm-test-sec-'..class..'-'..config.dtype.."-"..config.dsize..'-score-log',true,class.."-".."test-sec-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  

  config.hidden_size = config.vector_size
  local mlpClassifier = nn.Sequential()
  mlpClassifier:add(nn.Linear(config.vector_size, config.hidden_size))
  mlpClassifier:add(nn.Tanh())
  mlpClassifier:add(nn.Linear(config.hidden_size, 1))
  mlpClassifier:add(nn.Sigmoid())
  criterion=nn.BCECriterion():cuda()
  mlpClassifier:cuda()
  
  mlpClassifier:getParameters():uniform(-1*config.init,config.init)
  
  --if trainDataTestPri == true then
    scoreTrPri = 0
    precision_accTrPri = 0
    recall_accTrPri = 0
    f1_scoreTrPri = 0
    mcc_accTrPri = 0
    fall_outTrPri = 0
  --end
  
  --if trainDataTestSec == true then
    scoreTrSec = 0
    precision_accTrSec = 0
    recall_accTrSec = 0
    f1_scoreTrSec = 0
    mcc_accTrSec = 0
    fall_outTrSec = 0
  --end
  
  --if testDataTestPri == true then
    scoreTstPri = 0
    precision_accTstPri = 0
    recall_accTstPri = 0
    f1_scoreTstPri = 0
    mcc_accTstPri = 0
    fall_outTstPri = 0
  --end
  
  --if testDataTestSec == true then
    scoreTstSec = 0
    precision_accTstSec = 0
    recall_accTstSec = 0
    f1_scoreTstSec = 0
    mcc_accTstSec = 0
    fall_outTstSec = 0
  --end
  
  trainDataLoadedPri = false
  trainDataLoadedSec = false
  testDataLoadedPri = false
  testDataLoadedSec = false
  
  train_size = math.floor(allPri:size(1)*config.train_per)
  print("Train size/all : "..train_size.."/"..allPri:size(1))
  loss_logger_train = optim.Logger('./log/cdlc-lstm-'..class.."-"..config.dtype.."-"..config.dsize..'-train_loss-log',true,class.."-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  grad_logger_train = optim.Logger('./log/cdlc-lstm-'..class..
    "-"..config.dtype.."-"..config.dsize..'-grad-log',true,class.."-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  loss_logger_test = optim.Logger('./log/cdlc-lstm-'..class.."-"..config.dtype.."-"..config.dsize..'-test_loss-log',true,class.."-"..config.dtype.."-"..config.dsize.."-"..config.out_file)  
  
  loopcnt = 1
  totalfoldcnt = 0
  local errorsTrain = {}
  local errorsTest = {}
  while true do
  
  for fold = 1, config.folds do
    totalfoldcnt = totalfoldcnt + 1
    print("fold begin..."..os.date("%Y_%m_%d_%X"))
    allPri,targetsPri = shuffleTensor(allPri,targetsPri)
    mlpClassifier:training()
    for epoch = 1, config.max_epoch do
      k=0
      local totalGradWeight = 0
      local inds = torch.range(1, math.floor(train_size),config.batch_size)
      local shuffle = torch.totable(torch.randperm(inds:size(1)))

      for i = 1, inds:size(1) do
        local start = inds[shuffle[i]]
        local endd = inds[shuffle[i]]+config.batch_size-1
        if((start > train_size) or (endd > train_size)) then
          k = k + 1
          endd = train_size
        end
        local x = allPri[{{start,endd},{}}]
        local y = targetsPri[{{start,endd}}]
        err,gradWeight = gradUpdate(mlpClassifier, x, y, criterion, config.lr)
        totalGradWeight = totalGradWeight + gradWeight
        table.insert(errorsTrain, err)
      end
      if epoch % config.threshold == 0 then 
        config.lr = config.lr * config.lr_decay 
      end
      if epoch % config.graphshow == 0 then 
        if log_flag == true then
          loss_logger_train:plot()
          --grad_logger_train:plot() 
        end
      end
      loss_logger_train:add{['training error mean'] = torch.mean( torch.Tensor( errorsTrain))}
      loss_logger_train:style{['training error mean'] = '+-'}
      grad_logger_train:add{['grad weight'] = totalGradWeight }
      grad_logger_train:style{['grad weight'] = '+-'}
    end -- end for mlp train loop
    if log_flag == true then
          loss_logger_train:plot()
          --grad_logger_train:plot() 
    end  
    print(class.." fold :"..fold.." - fold loop cnt : "..loopcnt.." model trained ")
    print("fold end..."..os.date("%Y_%m_%d_%X"))
    
    if trainDataTestPri == true then
    -- PRIMARY TRAIN DATA ACC 
    
      correctTrPri = 0
      predicted_positivesTrPri = 0
      true_positivesTrPri = 0
      true_negativesTrPri = 0
      false_positivesTrPri = 0
      false_negativesTrPri = 0
      all_positivesTrPri = 0
      mlpClassifier:evaluate()
      print("Primary Train Test Size "..allPri:size(1))
      for i = 1, allPri:size(1) do
        if targetsPri[i] == 1 then all_positivesTrPri = all_positivesTrPri + 1 end
        x = allPri[i]
        pred = mlpClassifier:forward(x)
        if pred[1] < 0.5 then
          output = 0 
        else 
          output = 1 
          predicted_positivesTrPri = predicted_positivesTrPri + 1
        end
        if output == targetsPri[i] then
          correctTrPri = correctTrPri + 1 
          if targetsPri[i] == 1 then true_positivesTrPri = true_positivesTrPri + 1 end
          if targetsPri[i] == 0 then true_negativesTrPri = true_negativesTrPri + 1 end
        else
          if output == 1 then false_positivesTrPri = false_positivesTrPri + 1 end
          if output == 0 then false_negativesTrPri = false_negativesTrPri + 1 end
        end
      end  -- End for primary train-test loop
      
      print("TP:"..true_positivesTrPri..", FP:"..false_positivesTrPri..", TN:"..true_negativesTrPri..", FN:"..false_negativesTrPri)
      if not(predicted_positivesTrPri == 0 or all_positivesTrPri == 0) then 
        precisionTrPri = true_positivesTrPri / predicted_positivesTrPri
        recallTrPri = true_positivesTrPri / all_positivesTrPri
        precision_accTrPri = precision_accTrPri + precisionTrPri
        recall_accTrPri = recall_accTrPri + recallTrPri
        fall_outTrPri = false_positivesTrPri / (false_positivesTrPri + true_negativesTrPri)
        
        tmp = (true_positivesTrPri + false_positivesTrPri)*(true_positivesTrPri + false_negativesTrPri)*(true_negativesTrPri + false_positivesTrPri)*(true_negativesTrPri + false_negativesTrPri)
        mcc = (true_positivesTrPri * true_negativesTrPri - false_positivesTrPri * false_negativesTrPri)/math.sqrt(tmp)
        mcc_accTrPri = mcc_accTrPri + mcc
        print(" mcc div "..tmp.." mcc "..mcc)
        f1_score = 0
        if (precisionTrPri+recallTrPri) ~=  0 then
          f1_score = (2 * precisionTrPri * recallTrPri / (precisionTrPri+recallTrPri))
          f1_scoreTrPri = f1_scoreTrPri + f1_score
        else
          print(class.." fold :"..fold.." - fold loop cnt : "..loopcnt.." model trained ")
        end
        scoreTrPri = scoreTrPri + correctTrPri / allPri:size(1)
        
        --score_loggerTrPri:add{['f1-score'] = f1_score,['score'] = correctTrPri / allPri:size(1),['mcc-score'] = mcc}
        --score_loggerTrPri:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
        --score_loggerTrPri:plot()
        print("fr score tr pri "..f1_score)
      else 
        print("predicted_positivesTrPri :"..predicted_positivesTrPri.." - all_positivesTrPri :"..all_positivesTrPri)         
        fold = fold - 1
      end
      collectgarbage()
    end
    
    if trainDataSec == true then
    -- SECONDARY TRAIN DATA ACC 
      if trainDataLoadedSec == false then
        pc = 0
        nc = 0
        print(doc_samples_tables_sec..".train.sec.input.pch".."  "..doc_samples_tables_sec..".train.sec.targets.pch")
        allTrnSec=torch.load(doc_samples_tables_sec..".train.sec.input.pch")
        targetsTrnSec=torch.load(doc_samples_tables_sec..".train.sec.targets.pch")
        pc = torch.sum(targetsTrnSec)
        nc = targetsTrnSec:size()[1] - torch.sum(targetsTrnSec)
        print("positive secondary train-test example : "..pc.." - negative secondary train-test example : "..nc)
        local n = allTrnSec:size()[1]
        allTrnSec = allTrnSec[{{1,n},{}}]:cuda()
        targetsTrnSec = targetsTrnSec[{{1,n}}]:cuda()
        trainDataLoadedSec = true
      else
        print("secondary train examples already loaded ")
      end
    
      correctTrSec = 0
      predicted_positivesTrSec = 0
      true_positivesTrSec = 0
      true_negativesTrSec = 0
      false_positivesTrSec = 0
      false_negativesTrSec = 0
      all_positivesTrSec = 0
      mlpClassifier:evaluate()
      print("Secondary Train Test Size "..allTrnSec:size(1))
      for i = 1, allTrnSec:size(1) do
        if targetsTrnSec[i] == 1 then all_positivesTrSec = all_positivesTrSec + 1 end
        x = allTrnSec[i]
        pred = mlpClassifier:forward(x)
        if pred[1] < 0.5 then
          output = 0 
        else 
          output = 1 
          predicted_positivesTrSec = predicted_positivesTrSec + 1
        end
        if output == targetsTrnSec[i] then
          correctTrSec = correctTrSec + 1 
          if targetsTrnSec[i] == 1 then true_positivesTrSec = true_positivesTrSec + 1 end
          if targetsTrnSec[i] == 0 then true_negativesTrSec = true_negativesTrSec + 1 end
        else
          if output == 1 then false_positivesTrSec = false_positivesTrSec + 1 end
          if output == 0 then false_negativesTrSec = false_negativesTrSec + 1 end
        end
      end  -- End for secondary train-test loop
      
      print("TP:"..true_positivesTrSec..", FP:"..false_positivesTrSec..", TN:"..true_negativesTrSec..", FN:"..false_negativesTrSec)
      if not(predicted_positivesTrSec == 0 or all_positivesTrSec == 0) then 
        precisionTrSec = true_positivesTrSec / predicted_positivesTrSec
        recallTrSec = true_positivesTrSec / all_positivesTrSec
        precision_accTrSec = precision_accTrSec + precisionTrSec
        recall_accTrSec = recall_accTrSec + recallTrSec
        fall_outTrSec = false_positivesTrSec / (false_positivesTrSec + true_negativesTrSec)
        
        tmp = (true_positivesTrSec + false_positivesTrSec)*(true_positivesTrSec + false_negativesTrSec)*(true_negativesTrSec + false_positivesTrSec)*(true_negativesTrSec + false_negativesTrSec)
        mcc = (true_positivesTrSec * true_negativesTrSec - false_positivesTrSec * false_negativesTrSec)/math.sqrt(tmp)
        mcc_accTrSec = mcc_accTrSec + mcc
        print(" mcc div "..tmp.." mcc "..mcc)
        f1_score = 0
        if (precisionTrSec+recallTrSec) ~=  0 then
          f1_score = (2 * precisionTrSec * recallTrSec / (precisionTrSec+recallTrSec))
          f1_scoreTrSec = f1_scoreTrSec + f1_score
        else
          print(class.." fold :"..fold.." - fold loop cnt : "..loopcnt.." model trained ")
        end
        scoreTrSec = scoreTrSec + correctTrSec / allTrnSec:size(1)
        
        score_loggerTrSec:add{['f1-score'] = f1_score,['score'] = correctTrSec / allTrnSec:size(1),['mcc-score'] = mcc}
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
        pc = 0
        nc = 0
        print(doc_samples_tables_pri..".test.pri.input.pch".."  "..doc_samples_tables_pri..".test.pri.targets.pch")
        allTstPri=torch.load(doc_samples_tables_pri..".test.pri.input.pch")
        targetsTstPri=torch.load(doc_samples_tables_pri..".test.pri.targets.pch")
        pc = torch.sum(targetsTstPri)
        nc = targetsTstPri:size()[1] - torch.sum(targetsTstPri)
        print("positive primary test example : "..pc.." - negative primary test example : "..nc)
        local n = allTstPri:size()[1]
        allTstPri = allTstPri[{{1,n},{}}]:cuda()
        targetsTstPri = targetsTstPri[{{1,n}}]:cuda()
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
      print("Primary Test Size "..allTstPri:size(1))
      for i = 1, allTstPri:size(1) do
        if targetsTstPri[i] == 1 then all_positivesTstPri = all_positivesTstPri + 1 end
        x = allTstPri[i]
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
        fall_outTstPri = false_positivesTstPri / (false_positivesTstPri + true_negativesTstPri)
        
        tmp = (true_positivesTstPri + false_positivesTstPri)*(true_positivesTstPri + false_negativesTstPri)*(true_negativesTstPri + false_positivesTstPri)*(true_negativesTstPri + false_negativesTstPri)
        mcc = (true_positivesTstPri * true_negativesTstPri - false_positivesTstPri * false_negativesTstPri)/math.sqrt(tmp)
        mcc_accTstPri = mcc_accTstPri + mcc
        print(" mcc div "..tmp.." mcc "..mcc)
        f1_score = 0
        if (precisionTstPri+recallTstPri) ~= 0 then
          f1_score = (2 * precisionTstPri * recallTstPri / (precisionTstPri+recallTstPri))
          f1_scoreTstPri = f1_scoreTstPri + f1_score
        else
          print(class.." fold :"..fold.." - fold loop cnt : "..loopcnt.." model trained ")
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
        pc = 0
        nc = 0
        print(doc_samples_tables_sec..".test.sec.input.pch".."  "..doc_samples_tables_sec..".test.sec.targets.pch")
        allTstSec=torch.load(doc_samples_tables_sec..".test.sec.input.pch")
        targetsTstSec=torch.load(doc_samples_tables_sec..".test.sec.targets.pch")
        pc = torch.sum(targetsTstSec)
        nc = targetsTstSec:size()[1] - torch.sum(targetsTstSec)
        print("positive secondary test example : "..pc.." - negative secondary test example : "..nc)
        local n = allTstSec:size()[1]
        allTstSec = allTstSec[{{1,n},{}}]:cuda()
        targetsTstSec = targetsTstSec[{{1,n}}]:cuda()
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
      mlpClassifier:evaluate()
      print("secondary Test Size "..allTstSec:size(1))
      for i = 1, allTstSec:size(1) do
        if targetsTstSec[i] == 1 then all_positivesTstSec = all_positivesTstSec + 1 end
        x = allTstSec[i]
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
        fall_outTstSec = false_positivesTstSec / (false_positivesTstSec + true_negativesTstSec)
        
        tmp = (true_positivesTstSec + false_positivesTstSec)*(true_positivesTstSec + false_negativesTstSec)*(true_negativesTstSec + false_positivesTstSec)*(true_negativesTstSec + false_negativesTstSec)
        mcc = (true_positivesTstSec * true_negativesTstSec - false_positivesTstSec * false_negativesTstSec)/math.sqrt(tmp)
        mcc_accTstSec = mcc_accTstSec + mcc
        print(" mcc div "..tmp.." mcc "..mcc)
        f1_score = 0
        if (precisionTstSec+recallTstSec) ~= 0 then
          f1_score = (2 * precisionTstSec * recallTstSec / (precisionTstSec+recallTstSec))
          f1_scoreTstSec = f1_scoreTstSec + f1_score
        else
          print(class.." fold :"..fold.." - fold loop cnt : "..loopcnt.." model trained ")
        end
        scoreTstSec = scoreTstSec + correctTstSec / allTstSec:size(1)
        score_loggerTstSec:add{['f1-score'] = f1_score,['score'] = correctTstSec / allTstSec:size(1),['mcc-score'] = mcc}
        score_loggerTstSec:style{['f1-score'] = '+-',['score'] = '+-',['mcc-score'] = '+-'}
        --score_loggerTstSec:plot()
        print("fr score tst sec "..f1_score)
      else 
        print("predicted_positivesTstSec :"..predicted_positivesTstSec.." - all_positivesTstSec :"..all_positivesTstSec)         
        fold = fold - 1
      end
      collectgarbage()
    end
    print("\n")
  end -- End for fold loop
  
  loopcnt = loopcnt + 1
  
  if tstDataSec == true and f1_scoreTstSec > 0 then 
    break -- exit while
  end
  
  if loopcnt > config.max_loop then 
    print("Max loop reached : "..loopcnt)
    break -- exit while
  end
  
end -- while for loop 

print("fold cnt "..totalfoldcnt.."\n")

  if trainDataPri == true then
    print("Class: "..class.." "..lang_pair)
    print("Primary Train Score: " .. (scoreTrPri / totalfoldcnt) * 100 .. "%")
    print("Primary Train Precision: " .. precision_accTrPri / totalfoldcnt)
    print("Primary Train Recall: " .. recall_accTrPri / totalfoldcnt)
    print("Primary Train F1-Score: " .. f1_scoreTrPri / totalfoldcnt)
    print("Primary Train MCC-Score: " .. mcc_accTrPri / totalfoldcnt)
    print("Primary Train Fall-out: " .. fall_outTrPri / totalfoldcnt)
  end
  
  if trainDataSec == true then
    print("Class: "..class.." "..lang_pair)
    print("Secondary Train Score: " .. (scoreTrSec / totalfoldcnt) * 100 .. "%")
    print("Secondary Train Precision: " .. precision_accTrSec / totalfoldcnt)
    print("Secondary Train Recall: " .. recall_accTrSec / totalfoldcnt)
    print("Secondary Train F1-Score: " .. f1_scoreTrSec / totalfoldcnt)
    print("Secondary Train MCC-Score: " .. mcc_accTrSec / totalfoldcnt)
    print("Secondary Train Fall-out: " .. fall_outTrSec / totalfoldcnt)
  end

  if tstDataPri == true then
    print("Class: "..class.." "..lang_pair)
    print("Primary Test Score: " .. (scoreTstPri / totalfoldcnt) * 100 .. "%")
    print("Primary Test Precision: " .. precision_accTstPri / totalfoldcnt)
    print("Primary Test Recall: " .. recall_accTstPri / totalfoldcnt)
    print("Primary Test F1-Score: " .. f1_scoreTstPri / totalfoldcnt)
    print("Primary Test MCC-Score: " .. mcc_accTstPri / totalfoldcnt)
    print("Primary Test Fall-out: " .. fall_outTstPri / totalfoldcnt)
  end
  
  if tstDataSec == true then
    print("Class: "..class.." "..lang_pair)
    print("Secondary Test Score: " .. (scoreTstSec / totalfoldcnt) * 100 .. "%")
    print("Secondary Test Precision: " .. precision_accTstSec / totalfoldcnt)
    print("Secondary Test Recall: " .. recall_accTstSec / totalfoldcnt)
    print("Secondary Test F1-Score: " .. f1_scoreTstSec / totalfoldcnt)
    print("Secondary Test MCC-Score: " .. mcc_accTstSec / totalfoldcnt)
    print("Secondary Test Fall-out: " .. fall_outTstSec / totalfoldcnt)
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
    output_file:write("Secondary Train Fall-out: " .. fall_outTrSec / totalfoldcnt)
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
    output_file:write("Primary Test Fall-out: " .. fall_outTstPri / totalfoldcnt)
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
    output_file:write("Secondary Test Fall-out: " .. fall_outTstSec / totalfoldcnt)
    output_file:write("\n")
    output_file:write("\n")
  end

  output_file:flush()
  
  if trainDataPri == true then
    f1_score_avgTrPri = f1_score_avgTrPri + f1_scoreTrPri / totalfoldcnt
    score_avgTrPri = score_avgTrPri +  (scoreTrPri / totalfoldcnt) * 100
    mcc_acc_avgTrPri = mcc_acc_avgTrPri + mcc_accTrPri / totalfoldcnt
    fall_out_avgTrPri = fall_out_avgTrPri + fall_outTrPri / totalfoldcnt
  end

  if trainDataSec == true then
    f1_score_avgTrSec = f1_score_avgTrSec + f1_scoreTrSec / totalfoldcnt
    score_avgTrSec = score_avgTrSec +  (scoreTrSec / totalfoldcnt) * 100
    mcc_acc_avgTrSec = mcc_acc_avgTrSec + mcc_accTrSec / totalfoldcnt
    fall_out_avgTrSec = fall_out_avgTrSec + fall_outTrSec / totalfoldcnt
  end

  if tstDataPri == true then
    f1_score_avgTstPri = f1_score_avgTstPri + f1_scoreTstPri / totalfoldcnt
    score_avgTstPri = score_avgTstPri +  (scoreTstPri / totalfoldcnt) * 100
    mcc_acc_avgTstPri = mcc_acc_avgTstPri + mcc_accTstPri / totalfoldcnt
    fall_out_avgTstPri = fall_out_avgTstPri + fall_outTstPri / totalfoldcnt
  end

  if tstDataSec == true then
    f1_score_avgTstSec = f1_score_avgTstSec + f1_scoreTstSec / totalfoldcnt
    score_avgTstSec = score_avgTstSec +  (scoreTstSec / totalfoldcnt) * 100
    mcc_acc_avgTstSec = mcc_acc_avgTstSec + mcc_accTstSec / totalfoldcnt
    fall_out_avgTstSec = fall_out_avgTstSec + fall_outTstSec / totalfoldcnt
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
--summary_file:write("\n")
--summary_file:flush()

if trainDataPri == true then
  f1_avg=f1_score_avgTrPri / #classes
  mcc_avg=mcc_acc_avgTrPri / #classes
  score_avg=score_avgTrPri / #classes
  print("Average Primary Train F1-Score: " .. f1_avg)
  print("Average Primary Train MCC-Score: " .. mcc_avg)  
  print("Average Primary Train Acc: " .. score_avg)
  output_file:write("Average Primary Train F1-Score: " .. f1_avg)
  output_file:write("\n")
  output_file:write("Average Primary Train MCC-Score: " .. mcc_avg)
  output_file:write("\n")
  output_file:write("Average Primary Train Acc: " .. score_avg)
  output_file:write("\n")
  summary_file:write("Train,Pri,"..config.lstmtype..","..config.dtype..","..config.folds..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
end

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
  summary_file:write("Train,Sec,"..config.lstmtype..","..config.dtype..","..config.folds..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
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
  summary_file:write("Test,Pri,"..config.lstmtype..","..config.dtype..","..config.folds..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
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
  summary_file:write("Test,Sec,"..config.lstmtype..","..config.dtype..","..config.folds..","..config.tstmax..","..config.langPri..","..config.langSec..","..f1_avg..","..mcc_avg..","..score_avg.."\n")
end

io.close(summary_file)
io.close(output_file)


