math.randomseed(1)

require 'nn'
require 'cunn'
require 'rnn'
require 'lfs'
require 'utils'
require 'data'
require 'os'

local stringx = require('pl.stringx')
local file = require('pl.file')
--local data_path = "./data/cdlc_en_tr/"

function gradUpdate(mlp, x, y, criterion, learningRate)
   --print("grad ---")
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   --print("grad ---"..err.." "..gradCriterion[1])
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   --print("grad ---end")
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

function map_data(data,longest_seq,corpus)
  --local vocab = torch.load(vocab_file)
  x = torch.Tensor(#data,longest_seq)
  local miscnt = 0
  local allcnt = 0
  for idx,item in ipairs(data) do
    all_words = stringx.split(item, sep)
    sample = torch.Tensor(longest_seq)
    all_words = padding(all_words,longest_seq)
    allcnt = allcnt + #all_words
    for k,word in ipairs(all_words) do
      if(corpus.vocab_map[word] ~= nil) then
        sample[k] = corpus.vocab_map[word]
      else
        sample[k] = 0
        miscnt = miscnt + 1
      end
    end
    x[idx] = sample  
  end
  --print("all words cnt "..allcnt.."  missing cnt "..miscnt.." "..100*miscnt/allcnt)
  return x
end

function getData(data_path,class,corpus)
  local pos_data = {}
  local neg_data = {}
  local longest_seq = 0
  for f in lfs.dir(data_path..'/'..class..'/positive') do
    local text = file.read(data_path..'/'..class..'/positive/'..f)
    if (text ~= nil) then
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
      no_words = #stringx.split(text,sep)
      --no_words = table.getn(stringx.split(text,sep))
      if longest_seq < no_words then 
        longest_seq = no_words
      end
      table.insert(neg_data,text)
    end
  end
  local pos_mapped = map_data(pos_data,longest_seq,corpus)
  local neg_mapped = map_data(neg_data,longest_seq,corpus)
  return pos_mapped, neg_mapped
end

sep = ' '
--local L1 = 'english'
--local L2 = 'turkish'
--local classes = {'economics'}
local classes = {'art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology'}


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
config.data_path1 = "" -- corpus file  for lang 1
config.data_path2 = "" -- corpus file  for lang 2
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

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus1", config.corpus1)
cmd:option("-corpus2", config.corpus2)
cmd:option("-data_path1", config.data_path1)
cmd:option("-data_path2", config.data_path2)
cmd:option("-lr", config.lr)
cmd:option("-lr_decay", config.lr_decay)
cmd:option("-max_epoch", config.max_epoch)
cmd:option("-inp_size", config.inp_size)
cmd:option("-threshold", config.threshold)
cmd:option("-folds", config.folds)
cmd:option("-init", config.init)
cmd:option("-test_per", config.test_per)
cmd:option("-train_per", config.train_per)
params = cmd:parse(arg)


---corpus1 english.1000.tok -corpus2 turkish.1000.tok -data_path1 "../data/cdlc_en_tr/english" -data_path2 "../data/cdlc_en_tr/turkish" -lr 0.01 -lr_decay 1 -max_epoch 300 -inp_size 64 -threshold 5 -folds 10 -init 0.001 -test_per 0.25 -train_per 1

for param, value in pairs(params) do
    config[param] = value
end

nm = string.sub(config.corpus1,1,2).."-"..string.sub(config.corpus2,1,2)

--config.model1 = "./models/additive/all/"..config.corpus1.."."..nm..".model"-- model file  for lang 1
--config.model2 = "./models/additive/all/"..config.corpus2.."."..nm..".model" -- model file  for lang 2
--config.vocab1 = "./models/corpus/"..config.corpus1.."."..nm..".vocab" -- vocab file  for lang 1
--config.vocab2 = "./models/corpus/"..config.corpus2.."."..nm..".vocab" -- vocab file  for lang 2

--config.model1 = "../models/additive/all/english.all.tok.en-tu.model"
--config.model2 = "../models/additive/all/turkish.all.tok.en-tu.model"
--config.vocab1 = "../models/corpus/english.all.tok.en-tu.vocab"
--config.vocab2 = "../models/corpus/turkish.all.tok.en-tu.vocab"

config.model1 = "../models/additive/all/english.all.tok.en-tu.model"
config.model2 = "../models/additive/all/turkish.all.tok.en-tu.model"
config.vocab1 = "../models/corpus/english.all.tok.en-tu.corpus.tch"
config.vocab2 = "../models/corpus/turkish.all.tok.en-tu.corpus.tch"


for i,j in pairs(config) do
    print(i..": "..j)
end

local model_l1 = torch.load(config.model1):double()
local model_l2 = torch.load(config.model2):double()

local corpus1 = torch.load(config.vocab1)
local corpus2 = torch.load(config.vocab2)

output_file = io.open("f1scorevslr_"..nm.."-additive-all-1-1-train-test.csv", "w")
--output_file = io.open("f1scorevslr_"..nm.."-ddd.csv", "w")

f1_score_avg = 0
score_avg = 0
for _,class in ipairs(classes) do
  
  print("..."..os.date("%Y_%m_%d_%X"))
  model_l1:clearState()  -- Clear intermediate module states 
  model_l2:clearState()  -- Clear intermediate module states 
  
  positive, negative = getData(config.data_path1,class,corpus1)
  local split = nn.SplitTable(2)
  all_raw = nn.JoinTable(1):forward{positive, negative}
  targetsPri = nn.JoinTable(1):forward{torch.Tensor(positive:size(1)):fill(1), torch.Tensor(negative:size(1)):fill(0)}
  --yy = split:forward(all_raw)
  allPri = model_l1:forward(all_raw)
  --torch.save("eng-"..class.."-vectors.bin",all)
  
  
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(config.inp_size, 1))
  mlp:add(nn.Sigmoid())
  criterion=nn.BCECriterion()
  
  

  mlp:getParameters():uniform(-1*config.init,config.init)
  score = 0
  precision_acc = 0
  recall_acc = 0
  f1_score = 0
  
  --print("\n\n")
  
  for fold = 1, config.folds do
    local errors = {}
    --print("Fold .."..fold.."  Train size .."..math.floor(all:size(1)*config.train_per))
    for epoch = 1, config.max_epoch do
      --print("class : "..class..", fold : "..fold..", epoch "..epoch)
      errors = {}
      -- loop across all the samples
      local shuffle = torch.totable(torch.randperm(allPri:size(1)))
      for i = 1, math.floor(allPri:size(1)*config.train_per) do
        x = allPri[shuffle[i]]
        target = targetsPri[shuffle[i]]
        y = torch.Tensor(1):fill(target)
        err = gradUpdate(mlp, x, y, criterion, config.lr)
        table.insert(errors, err)
      end
      if epoch % config.threshold == 0 then config.lr = config.lr / config.lr_decay end
    end
    
    --print("Model trained..")
    
    -- get L2 data
    positive, negative = getData(config.data_path2,class,corpus2)
    positive = positive[{{1,math.floor(positive:size(1)*config.test_per)},{}}]
    negative = negative[{{1,math.floor(negative:size(1)*config.test_per)},{}}]
    all_raw = nn.JoinTable(1):forward{positive, negative}
    targetsSec = nn.JoinTable(1):forward{torch.Tensor(positive:size(1)):fill(1), torch.Tensor(negative:size(1)):fill(0)}
    allSec = model_l2:forward(all_raw)
    --torch.save("tr-"..class.."-vectors.bin",all)
    
    --print("Test size .."..all:size(1).."\n")
    
    correct = 0
    predicted_positives = 0
    true_positives = 0
    true_negatives = 0
    all_positives = positive:size(1)
    for i = 1, allSec:size(1) do
      x = allSec[i]
      pred = mlp:forward(x)
      if pred[1] < 0.5 then
        output = 0 
      else 
        output = 1 
        predicted_positives = predicted_positives + 1
      end
      if output == targetsSec[i] then
        correct = correct + 1 
        if targetsSec[i] == 1 then true_positives = true_positives + 1 end
        if targetsSec[i] == 0 then true_negatives = true_negatives + 1 end
      end
    end
    
    if not(predicted_positives == 0 or all_positives == 0) then 
      precision = true_positives / predicted_positives
      recall = true_positives / all_positives
      precision_acc = precision_acc + precision
      recall_acc = recall_acc + recall
      f1_score = f1_score + (2 * precision * recall / (precision+recall))
      score = score + correct / allSec:size(1)
    else 
      fold = fold -1
    end
  
  end
  
  print("Class: "..class)
  print("Test Score: " .. (score / config.folds) * 100 .. "%")
  print("Precision: " .. precision_acc / config.folds)
  print("Recall: " .. recall_acc / config.folds)
  print("F1-Score: " .. f1_score / config.folds)
  
  output_file:write("Class: "..class)
  output_file:write("\n")
  output_file:write("Test Score: " .. (score / config.folds) * 100 .. "%")
  output_file:write("\n")
  output_file:write("Precision: " .. precision_acc / config.folds)
  output_file:write("\n")
  output_file:write("Recall: " .. recall_acc / config.folds)
  output_file:write("\n")
  output_file:write("F1-Score: " .. f1_score / config.folds)
  output_file:write("\n")
  output_file:write("\n")
  output_file:flush()
  print("..."..os.date("%Y_%m_%d_%X"))
  
  f1_score_avg = f1_score_avg + f1_score / config.folds
  score_avg = score_avg +  (score / config.folds) * 100
  
end

print("Average f1-Score: " .. f1_score_avg / #classes)
print("Average Acc: " .. score_avg / #classes)
output_file:write("Average f1-Score: " .. f1_score_avg / #classes)
output_file:write("\n")
output_file:write("Average Acc: " .. score_avg / #classes)
output_file:write("\n")
io.close(output_file)


