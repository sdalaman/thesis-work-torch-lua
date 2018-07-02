require 'data-additive'
require 'model-additive'
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
model_save_path = path.."model-impl/add-tanh/model-files/multi/"
model_load_path = path.."model-impl/add-tanh/model-files/multi/"
corpus_path = path.."models/corpus/multi/"
data_path = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/corpusfiles/"
--data_path = "/home/saban/work/additive/models/corpus/multi/"
--tok
modelEnFileTok =   "/home/saban/work/additive/models/additive/all/En/1/english.all.tok.en-tu.model"
vocabEnFileTok =  "/home/saban/work/additive/models/corpus/english.all.tok.en-tu.vocab"
--morph
modelEnFileTokMorph =   "/home/saban/work/additive/models/additive-morph/all/En/1/english.all.tok.morph.en-tu.0.01.3200.256.additive-morph.model"
vocabEnFileTokMorph =  "/home/saban/work/additive/models/corpus/english.all.tok.morph.en-tu.vocab"

--modelEnFile =   "/home/saban/work/additive/models/additive-morph/1000/1/english.1000.tok.morph.en-tu.model"
--vocabEnFile =  "/home/saban/work/additive/models/corpus/english.1000.tok.morph.en-tu.vocab"

test_size = 100
cv_max = 10

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

function test_model(model,final,epoch,allEn,inputs_sec,test_size)
  local all_rootsPri = torch.zeros(test_size,allEn:size()[2]):cuda()
  inSec = torch.zeros(test_size,inputs_sec:size()[2]):cuda()
  cl = allEn:size()[1]
  math.randomseed( os.time() )

  for i=1,test_size do
    ln = math.random(1,cl)
    all_rootsPri[i] = allEn[ln]
    inSec[i] = inputs_sec[ln]
  end

  --local inputPri = inputs_pri[{{1,1000},{}}]
  --local inputSec = inputs_sec[{{1,1000},{}}]
  local outputSec = model:getLegSec():forward( inSec)
  all_rootsPri = all_rootsPri:double()
  all_rootsSec = outputSec:double()

  list1 = {}
  list2 = {}
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
    end
  end
  print("Epoch "..epoch.."  Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  print("\n")
  return score,all_rootsPri:size(1)
end

function cv_test(cv,model,maxepoch,allEn,insSec,test_size)
  tot = 0
  for i = 1,cv do
    sc,nsum = test_model(model,true,maxepoch,allEn,insSec,test_size)
    tot = tot + sc
  end
  print("-------"..cv.." Epochs avg: "..tot/cv.."\n")
  return sc,nsum,tot/cv
end

function train_model(model,allEn,corpus_sec,inputs_sec,config)
--  no_of_sents = #corpus_pri.sequences
  print ( 'Training started \n')
  beginning_time = torch.tic()
  --split = nn.SplitTable(2)
  additiveSec = model:getLegSec()
  criterion = model:getCriterion()
 
  k=0
  no_of_sents = math.floor(allEn:size()[1]/config.batch_size)*config.batch_size
  for i =config.prev_epoch + 1,config.max_epoch do
    errors = {}
    local inds = torch.range(1, no_of_sents,config.batch_size)
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    max_batch = no_of_sents/config.batch_size
    for j = 1, no_of_sents/config.batch_size do 
                --get input row and target
        --print("batch number "..j.."/"..max_batch)
        local start = inds[shuffle[j]]
        local endd = inds[shuffle[j]]+config.batch_size-1
        if((start > allEn:size()[1]) or (endd > allEn:size()[1])) then
          k = k + 1
        end
        local outputPri = allEn[{{start,endd},{}}]
        local inputSec = inputs_sec[{{start,endd},{}}]
        
        additiveSec:zeroGradParameters()
        -- print( target)
       
        outputSec = additiveSec:forward( inputSec)
        err = criterion:forward( outputSec, outputPri)
        table.insert( errors, err)
        gradOutputs = criterion:backward(outputSec, outputPri)
        additiveSec:backward(inputSec, gradOutputs)
        additiveSec:updateParameters(config.lr)
        --loss_logger:add{['training error pri'] = errPri ,['training error sec'] = errSec}
        --loss_logger:style{['training error pri'] = '+-' , ['training error sec'] = '+-' }
    end
    --printf ( 'epoch %4d, loss = %6.50f \n', i, torch.mean( torch.Tensor( errors))   )
    frm = 'epoch %4d, lr %5.10f, loss %6.50f \n'
    outStr = frm:format(i, config.lr,torch.mean( torch.Tensor( errors)))
    printf (outStr )
    loss_logger:add{['training error mean'] = torch.mean( torch.Tensor( errors))}
    loss_logger:style{['training error mean'] = '+-'}

    if i % config.dump_freq == 0 then 
      sc,num_of_samples,avg = cv_test(cv_max,model,i,allEn,inputs_sec,test_size)
      score_logger:add{['score '] = avg }
      score_logger:style{['score '] = '+-'}
      score_logger:plot()  
      loss_logger:plot()  
      model_save(config,i,model)
    end
    
    if i % config.threshold == 0 then 
      config.lr = config.lr * config.lr_decay 
    end

  end
  durationSeconds = torch.toc(beginning_time)
  print('time elapsed:'.. printtime(durationSeconds))
  print ( '\n')
  print ( 'Training ended \n')
end

--lr = 0.1
--emb_size = 64
--lr_decay = 1
--threshold = 100
--max_epoch = 500 --500
--batch_size = 100
--dump_freq = 50


-- Default configuration
config = {}
config.corpus1 = "" -- input data for lang 1
config.corpus2 = "" -- input data  for lang 
config.model2 = "" -- model file  for lang 2
config.vocab2 = "" -- vocab file  for lang 2
config.lookuptable2 = "" -- lookup table file  for lang 2
config.emb_size = 64
config.lr = 0.1
config.lr_prev = 0
config.lr_decay = 1
config.threshold = 100
config.max_epoch = 500
config.prev_epoch = 0
config.batch_size = 100
config.dump_freq = 50
config.model_load = 0
config.corpus_load = 0
config.dtype = "tok"

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus1", config.corpus1)
cmd:option("-corpus2", config.corpus2)
cmd:option("-lr", config.lr)
cmd:option("-lr_prev", config.lr_prev)
cmd:option("-lr_decay", config.lr_decay)
cmd:option("-max_epoch", config.max_epoch)
cmd:option("-prev_epoch", config.prev_epoch)
cmd:option("-emb_size", config.emb_size)
cmd:option("-threshold", config.threshold)
cmd:option("-batch_size", config.batch_size)
cmd:option("-dump_freq", config.dump_freq)
cmd:option("-model_load", config.model_load)
cmd:option("-corpus_load", config.corpus_load)
cmd:option("-dtype", config.dtype)
params = cmd:parse(arg)


---corpus1 english.1000.tok -corpus2 turkish.1000.tok -lr 0.1 -lr_decay 1 -max_epoch 300 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -dtype tok

---corpus1 english.10000.tok -corpus2 turkish.10000.tok -lr 0.1 -lr_decay 1 -max_epoch 200 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -dtype tok

--corpus1 english.1000.tok.morph -corpus2 turkish.1000.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 1000 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -dtype tok

--corpus1 english.10000.tok.morph -corpus2 turkish.10000.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 1000 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -dtype tok

---corpus1 english.all.tok -corpus2 turkish.all.tok -lr 0.1 -lr_decay 1 -max_epoch 350 -emb_size 64 -threshold 50 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -dtype tok

--corpus1 english.all.tok.morph -corpus2 turkish.all.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 3000 -emb_size 256 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -prev_epoch 0 -dtype tok

--corpus1 english.all.tok.morph -corpus2 turkish.all.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 5000 -emb_size 256 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -prev_epoch 3000 -dtype tok

-----------------------------------
---corpus1 train.de-en.en.1000.tok -corpus2 train.de-en.de.1000.tok -lr 0.1 -lr_prev 0 -prev_epoch 0 -lr_decay 1 -max_epoch 300 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -dtype tok

for param, value in pairs(params) do
    config[param] = value
end

nm = string.sub(config.corpus1,1,11)
nm1 = string.sub(config.corpus1,13,14)
nm2= string.sub(config.corpus2,13,14)
config.model2 = nm.."."..nm2.."."..config.dtype..".model" -- model file  for lang 2
config.vocab2 = corpus_path..nm.."."..nm2.."."..config.dtype..".vocab" -- vocab file  for lang 2
config.lookuptable2 = corpus_path..nm.."."..nm2.."."..config.dtype..".LT" -- lookup table file  for lang 2
corpus_name_2 = corpus_path..nm.."."..nm2.."."..config.dtype..".corpus.tch"
inputs_name_2 = corpus_path..nm.."."..nm2.."."..config.dtype..".inputs.tch"

for i,j in pairs(config) do
    print(i..": "..j)
end

if config.dtype == "tok" then
  modelEnFile = modelEnFileTok
  vocabEnFile =  vocabEnFileTok
end

if config.dtype == "tok.morph" then
  modelEnFile = modelEnFileTokMorph
  vocabEnFile =  vocabEnFileTokMorph
end

print("model file : "..modelEnFile.."\n")
print("vocab file : "..vocabEnFile.."\n")

modelEn = torch.load(modelEnFile):double()
modelEn:clearState()  -- Clear intermediate module states 
modelEn:cuda()
lp = modelEn:findModules('nn.LookupTableMaskZero')
config.emb_size = lp[1]:parameters()[1]:size()[2]
vocabEn = torch.load(vocabEnFile)

function paddingEn(sequence,longest_seq)
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


function map_dataEn(sentences,longest_seq,vocab)
  local x = torch.zeros(#sentences, longest_seq)
  --local mTable = {}
  local no_of_words = 0
  local unknown = 0
  for i = 1, #sentences do
    local words = stringx.split(sentences[i], '\t') -- tokenized with tab
    no_of_words = no_of_words + #words
    words = paddingEn(words,longest_seq)
    --local x = torch.zeros(longest_seq)
    for j = 1, #words do 
      if vocab[words[j]] == nil then
          x[i][j] = 0
          --x[j] = 0
          unknown = unknown + 1
      else
          --x[j] = vocab[words[j]] 
          x[i][j] = vocab[words[j]] 
      end
    end
    --table.insert(mTable,x)
  end
  print(string.format("Number of words = %d , unknown words = %d", no_of_words,unknown))
  return x
end

function getDataEn(fname,vocab)
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
  no_of_sents = #sentences
  longest_seq = 0
  for i = 1 , #sentences do
     local words = stringx.split(sentences[i],'\t') -- tokenized with tab
     words_len = #words
     if longest_seq < words_len then 
       longest_seq = words_len
     end
  end
  print(string.format("Longest sequence = %d", longest_seq))
  local mapped = map_dataEn(sentences,longest_seq,vocab)
  return mapped
end

function prepareData(model,dataTable,cfg)
  local batch_size = cfg.batch_size
  local no_of_sents = math.floor(dataTable:size()[1]/batch_size)*batch_size
  local inds = torch.range(1, no_of_sents,batch_size)
  local max_batch = no_of_sents/batch_size
  local vectors = torch.zeros(dataTable:size()[1],cfg.emb_size)
  if no_of_sents < dataTable:size()[1] then
    max_batch = max_batch + 1
  end 
  local start = 1
  local endd = 0
  for j = 1, max_batch do 
    endd = start+batch_size-1
    if endd > dataTable:size()[1] then
      endd = dataTable:size()[1]
    end
    --print(j.." "..start.."-"..endd)
    local inp = dataTable[{{start,endd},{}}]
    local t =  model:forward(inp)
     --vectors[{{start,endd},{}}]
    --local t = model:forward(dataTable[i])
    vectors[{{start,endd}}]:copy(t)
    start = endd + 1
  end      
  vectors = vectors:cuda()
  return vectors              
end

function model_save(cfg,epoch,model)
  config.model2 = nm.."."..nm2.."."..cfg.lr.."."..epoch.."."..header..".model" -- model file  for lang 2
  config.lookuptable2 = nm.."."..nm2.."."..cfg.lr.."."..epoch.."."..header..".LT" -- lookup table file  for lang 2
  model:getLegSec():clearState()
  torch.save(model_save_path..config.model2,model:getLegSec())
  torch.save(model_save_path..config.lookuptable2,model:getLookupTableSec())
  print(config.model2.."  "..config.lookuptable2)
  print("model data saved\n")
end


tableEn = getDataEn(data_path..config.corpus1.."."..config.dtype,vocabEn)
tableEn = tableEn:cuda()
allEn = prepareData(modelEn,tableEn,config)
tableEn = {}
modelEn = {}
lp = {}
corpus_1 = {}
inputs_1 = {}


if config.corpus_load == 0 then
  print("corpus data will be prepared\n")
  corpus_2 = Corpus(config.corpus2.."."..config.dtype,data_path)
  corpus_2:prepare_corpus()
  inputs_2=corpus_2:get_data():cuda()

  torch.save(config.vocab2,corpus_2.vocab_map)
  torch.save(corpus_name_2,corpus_2)
  torch.save(inputs_name_2,inputs_2)
  print("corpus data prepared and saved\n")
else 
  print("corpus data will be loaded\n")
  corpus_2 = torch.load(corpus_name_2)
  inputs_2 = torch.load(inputs_name_2)
  print("corpus data loaded\n")
end

--os.exit()
print(corpus_name_2.." "..corpus_2.no_of_words.." words, "..#corpus_2.sequences.." sents, "..corpus_2.longest_seq.." sent max lenght\n")
print(inputs_name_2.."\n")


biLnModel = BiLangModelAdditive(0,corpus_2.no_of_words,config)

header = "additive."..config.dtype.."."..config.emb_size

if config.model_load == 1 then
  config.model2 = nm.."."..nm2.."."..config.lr_prev.."."..config.prev_epoch.."."..header..".model" -- model file  for lang 2
  config.lookuptable2 = nm.."."..nm2.."."..config.lr_prev.."."..config.prev_epoch.."."..header..".LT" -- lookup table file  for lang 2
  print("model data will be loaded\n")
  print(config.model2.."\n")
  print(config.lookuptable2.."\n")
  biLnModel.legSec = torch.load(model_load_path..config.model2)
  biLnModel.ltSec = torch.load(model_load_path..config.lookuptable2)
  print("model data loaded\n")
  config.lr = config.lr * config.lr_decay 
end

function prepheadstr()
  str = string.format("%s-%s-lr=%3.7f-lr_decay=%3.7f-emb_size=%d" ,header,nm,config.lr,config.lr_decay,config.emb_size)
  return str
end

loss_logger = optim.Logger(header..'-loss-log-'..nm..'-'..config.lr,true,prepheadstr())
score_logger = optim.Logger(header..'-score-log-'..nm..'-'..config.lr,true,prepheadstr())

if config.model_load == 1 then
  sc,num_of_samples,avg = cv_test(cv_max,biLnModel,config.prev_epoch,allEn,inputs_2,test_size)
  score_logger:add{['score '] = avg }
  score_logger:style{['score '] = '+-'}
  score_logger:plot()  
end

train_model(biLnModel,allEn,corpus_2,inputs_2,config)
sc,num_of_samples,avg = cv_test(cv_max,biLnModel,config.max_epoch,allEn,inputs_2,test_size)
score_logger:add{['score '] = avg }
score_logger:style{['score '] = '+-'}
score_logger:plot()  
loss_logger:plot()  

model_save(config,config.max_epoch,biLnModel)

print("program ended")
