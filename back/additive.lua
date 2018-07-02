require 'data'
require 'model'
require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

--LEnglish = 'english'
--LTurkish = 'turkish'
--data_path = "./data/train_en_tr/"
data_path = "./data/"
corpus_path = "./models/corpus/"

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end


function test_model(model,final,epoch,inputs_pri,inputs_sec,output_file)
  local inputPri = inputs_pri[{{1,1000},{}}]
  local inputSec = inputs_sec[{{1,1000},{}}]
  local outputPri = model:getAdditivePri():forward( inputPri)
  local outputSec = model:getAdditiveSec():forward( inputSec)
  all_rootsPri = outputPri:double()
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
      list2[idxPri] = closest
    else
      if final == true then 
      --  print("Closest to: "..idxPri.." is: ".. closest) 
        list1[idxPri] = closest
      end
    end
  end
  print("Epoch "..epoch.."  Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  print("\n")
  --torch.save(LTurkish..'_model'..epoch,model:getAdditiveSec())
  --torch.save(LTurkish..'_LT'..epoch,ltSec)
  --torch.save(LEnglish..'_model'..epoch,model:getAdditivePri())
  --torch.save(LEnglish..'_LT'..epoch,ltPri)
  output_file:write("Epoch "..epoch.."  Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  output_file:write("\n")
  output_file:flush()
  return list1,list2
end

function train_model(model,corpus_pri,corpus_sec,inputs_pri,inputs_sec,output_file,error_file)
--  no_of_sents = #corpus_pri.sequences
  print ( 'Training started \n')
  beginning_time = torch.tic()
  --split = nn.SplitTable(2)
  additivePri = model:getAdditivePri()
  additiveSec = model:getAdditiveSec()
  criterion = model:getCriterion()
  k=0
  no_of_sents = math.floor(#corpus_pri.sequences/config.batch_size)*config.batch_size
  for i =1,config.max_epoch do
    errors = {}
    local inds = torch.range(1, no_of_sents,config.batch_size)
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    max_batch = no_of_sents/config.batch_size
    for j = 1, no_of_sents/config.batch_size do 
                --get input row and target
        --print("batch number "..j.."/"..max_batch)
        local start = inds[shuffle[j]]
        local endd = inds[shuffle[j]]+config.batch_size-1
        if((start > #corpus_pri.sequences) or (endd > #corpus_pri.sequences)) then
          k = k + 1
        end
        local inputPri = inputs_pri[{{start,endd},{}}]
        local inputSec = inputs_sec[{{start,endd},{}}]
        
        additivePri:zeroGradParameters()
        additiveSec:zeroGradParameters()
        -- print( target)
       
        local outputPri = additivePri:forward( inputPri)
        local outputSec = additiveSec:forward( inputSec)
        local err = criterion:forward( outputPri, outputSec)
        table.insert( errors, err)
        local gradOutputs = criterion:backward(outputPri, outputSec)
        additivePri:backward(inputPri, gradOutputs)
        additivePri:updateParameters(config.lr)
        
        outputPri = additivePri:forward( inputPri)
        outputSec = additiveSec:forward( inputSec)
        err = criterion:forward( outputSec, outputPri)
        table.insert( errors, err)
        gradOutputs = criterion:backward(outputSec, outputPri)
        additiveSec:backward(inputSec, gradOutputs)
        additiveSec:updateParameters(config.lr)
    end
    --printf ( 'epoch %4d, loss = %6.50f \n', i, torch.mean( torch.Tensor( errors))   )
    frm = 'epoch %4d, loss = %6.50f \n'
    outStr = frm:format(i, torch.mean( torch.Tensor( errors)))
    printf (outStr )
    frm = 'epoch %4d,%6.50f \n'
    outStr = frm:format(i, torch.mean( torch.Tensor( errors)))
    error_file:write(outStr)
    io.flush(error_file)
    if i % config.threshold == 0 then config.lr = config.lr / config.lr_decay end
    if i % config.dump_freq == 0 then test_model(model,false,i,inputs_pri,inputs_sec,output_file) end
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
config.model1 = "" -- model file  for lang 1
config.model2 = "" -- model file  for lang 2
config.vocab1 = "" -- vocab file  for lang 1
config.vocab2 = "" -- vocab file  for lang 2
config.lookuptable1 = "" -- lookup table file  for lang 1
config.lookuptable2 = "" -- lookup table file  for lang 2
config.emb_size = 64
config.lr = 0.1
config.lr_decay = 1
config.threshold = 100
config.max_epoch = 500
config.batch_size = 100
config.dump_freq = 50
config.model_load = 0
config.corpus_load = 0

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus1", config.corpus1)
cmd:option("-corpus2", config.corpus2)
cmd:option("-lr", config.lr)
cmd:option("-lr_decay", config.lr_decay)
cmd:option("-max_epoch", config.max_epoch)
cmd:option("-emb_size", config.emb_size)
cmd:option("-threshold", config.threshold)
cmd:option("-batch_size", config.batch_size)
cmd:option("-dump_freq", config.dump_freq)
cmd:option("-model_load", config.model_load)
cmd:option("-corpus_load", config.corpus_load)
params = cmd:parse(arg)


---corpus1 english.1000.tok -corpus2 turkish.1000.tok -lr 0.1 -lr_decay 1 -max_epoch 500 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1

for param, value in pairs(params) do
    config[param] = value
end

nm = string.sub(config.corpus1,1,2).."-"..string.sub(config.corpus2,1,2)
config.model1 = config.corpus1.."."..nm..".model"-- model file  for lang 1
config.model2 = config.corpus2.."."..nm..".model" -- model file  for lang 2
config.vocab1 = config.corpus1.."."..nm..".vocab" -- vocab file  for lang 1
config.vocab2 = config.corpus2.."."..nm..".vocab" -- vocab file  for lang 2
config.lookuptable1 = config.corpus1.."."..nm..".LT" -- lookup table file  for lang 1
config.lookuptable2 = config.corpus2.."."..nm..".LT" -- lookup table file  for lang 2

corpus_name_1 = corpus_path..config.corpus1.."."..nm..".corpus.tch"
corpus_name_2 = corpus_path..config.corpus2.."."..nm..".corpus.tch"
inputs_name_1 = corpus_path..config.corpus1.."."..nm..".inputs.tch"
inputs_name_2 = corpus_path..config.corpus2.."."..nm..".inputs.tch"


for i,j in pairs(config) do
    print(i..": "..j)
end

if config.corpus_load == 0 then
  print("corpus data will be prepared\n")
  corpus_1 = Corpus(config.corpus1)
  corpus_2 = Corpus(config.corpus2)
  corpus_1:prepare_corpus()
  corpus_2:prepare_corpus()
  --no_of_sents = #corpus_en.sequences
  inputs_1=corpus_1:get_data():cuda()
  inputs_2=corpus_2:get_data():cuda()

  torch.save(corpus_name_1,corpus_1)
  torch.save(corpus_name_2,corpus_2)
  torch.save(inputs_name_1,inputs_1)
  torch.save(inputs_name_2,inputs_2)
  torch.save(config.vocab1,corpus_1.vocab_map)
  torch.save(config.vocab2,corpus_2.vocab_map)
  print("corpus data prepared and saved\n")
else 
  print("corpus data will be loaded\n")
  corpus_1 = torch.load(corpus_name_1)
  corpus_2 = torch.load(corpus_name_2)
  inputs_1 = torch.load(inputs_name_1)
  inputs_2 = torch.load(inputs_name_2)
  print("corpus data loaded\n")
end

--os.exit()
print(corpus_name_1.." "..corpus_1.no_of_words.." words, "..#corpus_1.sequences.." sents, "..corpus_1.longest_seq.." sent max lenght\n")
print(corpus_name_2.." "..corpus_2.no_of_words.." words, "..#corpus_2.sequences.." sents, "..corpus_2.longest_seq.." sent max lenght\n")
print(inputs_name_1..inputs_name_2.."\n")

--torch.save(config.vocab1,corpus_1.vocab_map)
--torch.save(config.vocab2,corpus_2.vocab_map)
biLnModel = BiLangModel(corpus_1.no_of_words,corpus_2.no_of_words,config)

if config.model_load == 1 then
  print("model data will be loaded\n")
  print(config.model1..config.model2.."\n")
  print(config.lookuptable1..config.lookuptable2.."\n")
  biLnModel.additivePri = torch.load(config.model1)
  biLnModel.additiveSec = torch.load(config.model2)
  biLnModel.ltPri = torch.load(config.lookuptable1)
  biLnModel.ltSec = torch.load(config.lookuptable2)
  print("model data loaded\n")
end

output_file = io.open("test_results.all."..nm.."-"..config.lr..".txt", "w")
error_file = io.open("error_results.all."..nm.."-"..config.lr..".txt", "w")

if config.model_load == 1 then
  crossList,closeList=test_model(biLnModel,true,0,inputs_1,inputs_2,output_file)
end

train_model(biLnModel,corpus_1,corpus_2,inputs_1,inputs_2,output_file,error_file)
crossList,closeList=test_model(biLnModel,true,'final',inputs_1,inputs_2,output_file)

if true then
  print("model data will be saved\n")
  print(config.model1..config.model2.."\n")
  print(config.lookuptable1..config.lookuptable2.."\n")
  torch.save(config.model2,biLnModel:getAdditiveSec())
  torch.save(config.lookuptable2,biLnModel:getLookupTableSec())
  torch.save(config.model1,biLnModel:getAdditivePri())
  torch.save(config.lookuptable1,biLnModel:getLookupTablePri())
  print("model data saved\n")
end

io.close(output_file)
io.close(error_file)

numberOfWords2 = 0
index2words2 = {}
for k,v in pairs(corpus_2.vocab_map) 
do 
  numberOfWords2 = numberOfWords2 + 1 
  index2words2[v] = k
end
print(config.corpus2.."--"..numberOfWords2)  

index2words1 = {}
numberOfWords1 = 0
for k,v in pairs(corpus_1.vocab_map) 
do 
  numberOfWords1 = numberOfWords1 + 1 
  index2words1[v] = k
end
print(config.corpus1.."--"..numberOfWords1)  

for k,v in pairs(crossList) 
do
  --print(string.format("%s %s",index2words1[k],index2words2[v]))
end

for k,v in pairs(closeList) 
do
  --print(string.format("%s %s",index2words1[k],index2words2[v]))
end

print(numberOfWords1)  
print(numberOfWords2)  
