require 'data_test'
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
model_path = "./models/"
corpus_path = "./models/corpus/"

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end


function test_model(model,final,epoch,inputs_pri,inputs_sec,test_size,output_file)
  local inputPri = inputs_pri[{{1,test_size},{}}]
  local inputSec = inputs_sec[{{1,test_size},{}}]
  local outputPri = model:getAdditivePri():forward( inputPri)
  local outputSec = model:getAdditiveSec():forward( inputSec)
  all_rootsPri = outputPri:double()
  all_rootsSec = outputSec:double()
  local res = torch.FloatTensor(test_size,test_size)

  list1 = {}
  list2 = {}
  score = 0
  for idxPri = 1, all_rootsPri:size(1) do
    closest = idxPri
    for idxSec = 1, all_rootsSec:size(1) do
      dist = torch.dist(all_rootsPri[idxPri],all_rootsSec[idxSec])
      res[idxPri][idxSec] = dist
      if dist < torch.dist(all_rootsPri[idxPri],all_rootsSec[closest]) then
        closest = idxSec
      end
    end
    
    if idxPri == closest then
      score = score + 1 
      list2[idxPri] = closest
    end
    list1[idxPri] = closest
  end
  
  val, list = torch.min(res, 2)
  
  print("Epoch "..epoch.."  Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  print("\n")
  --torch.save(LTurkish..'_model'..epoch,model:getAdditiveSec())
  --torch.save(LTurkish..'_LT'..epoch,ltSec)
  --torch.save(LEnglish..'_model'..epoch,model:getAdditivePri())
  --torch.save(LEnglish..'_LT'..epoch,ltPri)
  output_file:write("Epoch "..epoch.."  Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  output_file:write("\n")
  return list1,list2,score
end

function test_model2(model,final,epoch,inputs_pri,inputs_sec,test_size,output_file)
  local inputPri = inputs_pri[{{1,test_size},{}}]
  local inputSec = inputs_sec[{{1,test_size},{}}]
  local outputPri = model:getAdditivePri():forward( inputPri)
  local outputSec = model:getAdditiveSec():forward( inputSec)
  all_rootsPri = outputPri:double()
  all_rootsSec = outputSec:double()
  local res = torch.FloatTensor(test_size,test_size)

  list1 = {}
  list2 = {}
  score = 0
  for idxPri = 1, all_rootsPri:size(1) do
    for idxSec = 1, all_rootsSec:size(1) do
      dist = torch.dist(all_rootsPri[idxPri],all_rootsSec[idxSec]) 
      res[idxPri][idxSec] = dist
    end
  end
  
  val, list = torch.min(res, 2)
  
  score = 0
  for i = 1,list:size(1) do
    if i == list[i][1] then 
      score = score + 1 
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
  return list1,list2
end



function train_model(model,corpus_pri,corpus_sec,inputs_pri,inputs_sec,output_file)
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
    printf ( 'epoch %4d, loss = %6.50f \n', i, torch.mean( torch.Tensor( errors))   )
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
config.testcorpus1 = "" -- input data for lang 1
config.testcorpus2 = "" -- input data  for lang 
config.model1 = "" -- model file  for lang 1
config.model2 = "" -- model file  for lang 2
config.vocab1 = "" -- vocab file  for lang 1
config.vocab2 = "" -- vocab file  for lang 2
config.lookuptable1 = "" -- lookup table file  for lang 1
config.lookuptable2 = "" -- lookup table file  for lang 2
config.emb_size = 64
config.dump_freq = 50
config.model_load = 0
--config.corpus_load = 0
config.test_size = 100
config.out_file  = "test-file.txt"
config.max_cv = 10

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus1", config.corpus1)
cmd:option("-corpus2", config.corpus2)
cmd:option("-testcorpus1", config.testcorpus1)
cmd:option("-testcorpus2", config.testcorpus2)
cmd:option("-emb_size", config.emb_size)
cmd:option("-dump_freq", config.dump_freq)
cmd:option("-model_load", config.model_load)
--cmd:option("-corpus_load", config.corpus_load)
cmd:option("-test_size", config.test_size)
cmd:option("-out_file", config.out_file)
cmd:option("-max_cv", config.max_cv)
params = cmd:parse(arg)


---corpus1 english.1000.tok -testcorpus1 en.tok -corpus2 turkish.1000.tok -testcorpus2 tr.tok -emb_size 64 -dump_freq 50 -model_load 1 -test_size 100 -max_cv 10 -out_file 

for param, value in pairs(params) do
    config[param] = value
end

output_file = io.open(config.out_file, "w")

nm = string.sub(config.corpus1,1,2).."-"..string.sub(config.corpus2,1,2)
config.model1 = model_path..config.corpus1.."."..nm..".model"-- model file  for lang 1
config.model2 = model_path..config.corpus2.."."..nm..".model" -- model file  for lang 2
config.vocab1 = corpus_path..config.corpus1.."."..nm..".vocab" -- vocab file  for lang 1
config.vocab2 = corpus_path..config.corpus2.."."..nm..".vocab" -- vocab file  for lang 2
config.lookuptable1 = model_path..config.corpus1.."."..nm..".LT" -- lookup table file  for lang 1
config.lookuptable2 = model_path..config.corpus2.."."..nm..".LT" -- lookup table file  for lang 2

corpus_name_1 = corpus_path..config.corpus1.."."..nm..".corpus.tch"
corpus_name_2 = corpus_path..config.corpus2.."."..nm..".corpus.tch"

for i,j in pairs(config) do
    print(i..": "..j)
end

sellist = {}

if true then
  print("test corpus data will be prepared\n")
  testcorpus_1 = Corpus(config.testcorpus1)
  testcorpus_2 = Corpus(config.testcorpus2)
  testcorpus_1:prepare_corpus(nil,config.test_size)
  testcorpus_2:prepare_corpus(nil,config.test_size)
  testcorpus_name_1 = config.testcorpus1
  testcorpus_name_2 = config.testcorpus2
  print("testcorpus data prepared\n")
  
  print("corpus data will be loaded\n")
  corpus_1 = torch.load(corpus_name_1)
  corpus_2 = torch.load(corpus_name_2)
  print("corpus data loaded\n")
  
  testinputs_1= testcorpus_shapedata(corpus_1,testcorpus_1):cuda()
  testinputs_2= testcorpus_shapedata(corpus_2,testcorpus_2):cuda()
end

print(testcorpus_name_1.." "..testcorpus_1.no_of_words.." words, "..#testcorpus_1.sequences.." sents, "..testcorpus_1.longest_seq.." sent max lenght\n")
print(testcorpus_name_2.." "..testcorpus_2.no_of_words.." words, "..#testcorpus_2.sequences.." sents, "..testcorpus_2.longest_seq.." sent max lenght\n")
--print(testinputs_name_1..testinputs_name_2.."\n")

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

cvscore = {}
crossList,closeList,score=test_model(biLnModel,true,0,testinputs_1,testinputs_2,#testcorpus_1.sequences,output_file)
cvscore[1] = score
print("CV 1 \n")

for cv = 2,config.max_cv do
  sellist = {}
  print("test corpus data will be prepared\n")
  testcorpus_1 = Corpus(config.testcorpus1)
  testcorpus_2 = Corpus(config.testcorpus2)
  testcorpus_1:prepare_corpus(nil,config.test_size)
  testcorpus_2:prepare_corpus(nil,config.test_size)
  testcorpus_name_1 = config.testcorpus1
  testcorpus_name_2 = config.testcorpus2
  print("testcorpus data prepared\n")
  
  testinputs_1= testcorpus_shapedata(corpus_1,testcorpus_1):cuda()
  testinputs_2= testcorpus_shapedata(corpus_2,testcorpus_2):cuda()

  crossList,closeList,score=test_model(biLnModel,true,0,testinputs_1,testinputs_2,#testcorpus_1.sequences,output_file)
  cvscore[cv] = score
  print("CV "..cv.."\n")

end

totscore = 0
for k,v in pairs(cvscore) do
  print(" "..k..".."..v.."\n")
  totscore = totscore + v
end
print("mean score "..totscore/config.max_cv.."/"..#testcorpus_1.sequences.."\n")

output_file:write("\n\n")
output_file:write("-------List of sentences closest each other-----------------\n\n")
for k,v in pairs(closeList) do
  output_file:write("---"..testcorpus_1.sequences[k].."\n")
  output_file:write("   "..testcorpus_2.sequences[v].."\n")
  output_file:write("\n")
end

output_file:write("\n\n")

output_file:write("-------List of each sentence with closest sentence-----------------\n\n")
for k,v in pairs(crossList) do
  flag = "F"
  if k == v then
    flag = "T"
  end
  output_file:write(flag.."---"..testcorpus_1.sequences[k].."\n")
  output_file:write("    "..testcorpus_2.sequences[v].."\n")
  output_file:write("\n")
end

output_file:write("\n\n")
io.close(output_file)

numberOfWords2 = 0
index2words2 = {}
for k,v in pairs(testcorpus_2.vocab_map) 
do 
  numberOfWords2 = numberOfWords2 + 1 
  index2words2[v] = k
end
print(config.testcorpus2.."--"..numberOfWords2)  

index2words1 = {}
numberOfWords1 = 0
for k,v in pairs(testcorpus_1.vocab_map) 
do 
  numberOfWords1 = numberOfWords1 + 1 
  index2words1[v] = k
end
print(config.testcorpus1.."--"..numberOfWords1)  

print(numberOfWords1)  
print(numberOfWords2)  
