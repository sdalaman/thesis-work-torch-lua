require 'data-lstm-test'
require 'model-lstm'
require 'fileListTr'
require 'fileListTst'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'

data_path = "./"
model_path = "../models/lstm/"
corpus_path = "../models/corpus/"
test_data_pathEn = "/home/saban/work/additive/data/cdlc_en_tr/test/englishTok"
test_data_pathTr = "/home/saban/work/additive/data/cdlc_en_tr/test/turkishTok"
train_data_pathEn = "/home/saban/work/additive/data/cdlc_en_tr/train/englishTok"
train_data_pathTr = "/home/saban/work/additive/data/cdlc_en_tr/train/turkishTok"

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

function test_model(model,final,epoch,inPri,inSec,test_size,output_file)
  local split = nn.SplitTable(2)
  local inputPri = split:forward(inPri)
  local inputSec = split:forward(inSec)
  --inputPri = inputs_pri[{{1,test_size},{}}]
  --inputSec = inputs_sec[{{1,test_size},{}}]
  local outputPri = model:getModelPri():forward( inputPri)
  local outputSec = model:getModelSec():forward( inputSec)
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
  
  print("- Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  print("\n")
  output_file:write("- Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  output_file:write("\n")
  return list1,list2,score
end

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
config.hdd_size = 64
config.test_size = 100
config.out_file  = "test-lstm-file.txt"
config.max_cv = 10
config.win_size = 1
config.init = 1
config.dsize = "all"

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus1", config.corpus1)
cmd:option("-corpus2", config.corpus2)
cmd:option("-testcorpus1", config.testcorpus1)
cmd:option("-testcorpus2", config.testcorpus2)
cmd:option("-emb_size", config.emb_size)
cmd:option("-test_size", config.test_size)
cmd:option("-out_file", config.out_file)
cmd:option("-max_cv", config.max_cv)
cmd:option("-hdd_size", config.hdd_size)
cmd:option("-dsize", config.dsize)
params = cmd:parse(arg)


---corpus1 english -testcorpus1 english.1000.tok -corpus2 turkish -testcorpus2 turkish.1000.tok -emb_size 64 -hdd_size 128 -test_size 2000 -max_cv 1 -out_file test-lstm-1000.txt -dsize 1000


for param, value in pairs(params) do
    config[param] = value
end

model_path = model_path..config.dsize.."/"
config.corpus1 = config.corpus1.."."..config.dsize..".".."tok"
config.corpus2 = config.corpus2.."."..config.dsize..".".."tok"

output_file = io.open(config.out_file, "w")
output_file:write("Model corpus: "..config.corpus1.."/"..config.corpus1.."  Test Corpus: " ..config.testcorpus1.."/"..config.testcorpus1.."\n")
output_file:write("Test size: "..config.test_size.."\n")

nm = string.sub(config.corpus1,1,2).."-"..string.sub(config.corpus2,1,2)
config.model1 = model_path..config.corpus1.."."..nm..".model"-- model file  for lang 1
config.model2 = model_path..config.corpus2.."."..nm..".model" -- model file  for lang 2
config.vocab1 = corpus_path..config.corpus1.."."..nm..".vocab" -- vocab file  for lang 1
config.vocab2 = corpus_path..config.corpus2.."."..nm..".vocab" -- vocab file  for lang 2
config.lookuptable1 = model_path..config.corpus1.."."..nm..".LT" -- lookup table file  for lang 1
config.lookuptable2 = model_path..config.corpus2.."."..nm..".LT" -- lookup table file  for lang 2

if config.dsize == "1000" then
  config.model2 = model_path.."en-tu_tu_0.0001_1700.lstm.128.1000.model"
  config.model1 = model_path.."en-tu_en_0.0001_1700.lstm.128.1000.model"
  config.lookuptable2 = model_path.."en-tu_tu_0.0001_1700.lstm.128.1000.LT"
  config.lookuptable1 = model_path.."en-tu_en_0.0001_1700.lstm.128.1000.LT"
end

if config.dsize == "10000" then
  config.model2 = model_path.."en-tu_tu_0.0001_2000.lstm.512.10000.model"
  config.model1 = model_path.."en-tu_en_0.0001_2000.lstm.512.10000.model"
  config.lookuptable2 = model_path.."en-tu_tu_0.0001_2000.lstm.512.10000.LT"
  config.lookuptable1 = model_path.."en-tu_en_0.0001_2000.lstm.512.10000.LT"
end

corpus_name_1 = corpus_path..config.corpus1.."."..nm..".corpus.tch"
corpus_name_2 = corpus_path..config.corpus2.."."..nm..".corpus.tch"

for i,j in pairs(config) do
    print(i..": "..j)
end

sellist = {}

print("corpus data will be loaded\n")
corpus_1 = torch.load(corpus_name_1)
corpus_2 = torch.load(corpus_name_2)
print("corpus data loaded\n")

seq_lenPri = 10
seq_lenSec = 10
config.win_size = 1

biLnModel = BiLangModelLSTM(corpus_1.no_of_words,corpus_2.no_of_words,seq_lenPri,seq_lenSec,config)  

print("model data will be loaded\n")
print(config.model1.." "..config.model2.."\n")
print(config.lookuptable1.." "..config.lookuptable2.."\n")
biLnModel.mdlPri = torch.load(config.model1)
biLnModel.mdlSec = torch.load(config.model2)
biLnModel.ltPri = torch.load(config.lookuptable1)
biLnModel.ltSec = torch.load(config.lookuptable2)
print("model data loaded\n")

totalLines = 0
totscore = 0
fileList = fileListTst
for fcnt = 1,#fileList do
  config.testcorpus1 = fileList[fcnt]..".tok"
  config.testcorpus2 = fileList[fcnt]..".tok"
  sellist = {}
  f1,f1err = io.open(test_data_pathEn..config.testcorpus1,'r') 
  f2,f2err = io.open(test_data_pathTr..config.testcorpus2,'r')
  if f1 and f2 then
    f1:close()
    f2:close()
    print("test corpus data will be prepared\n")
    testcorpus_1 = Corpus(config.testcorpus1,test_data_pathEn)
    testcorpus_2 = Corpus(config.testcorpus2,test_data_pathTr)
    testcorpus_1:prepare_corpus(nil,config.test_size)
    testcorpus_2:prepare_corpus(nil,config.test_size)
    testcorpus_name_1 = config.testcorpus1
    testcorpus_name_2 = config.testcorpus2
    print("testcorpus data prepared\n")
    testinputs_1= testcorpus_shapedataLSTM(corpus_1,testcorpus_1):cuda()
    testinputs_2= testcorpus_shapedataLSTM(corpus_2,testcorpus_2):cuda()

    print(testcorpus_name_1.." "..testcorpus_1.no_of_words.." words, "..#testcorpus_1.sequences.." sents, "..testcorpus_1.longest_seq.." sent max lenght\n")
    print(testcorpus_name_2.." "..testcorpus_2.no_of_words.." words, "..#testcorpus_2.sequences.." sents, "..testcorpus_2.longest_seq.." sent max lenght\n")

    totscoreAll = 0
    for cv = 1,config.max_cv do
      crossList,closeList,score=test_model(biLnModel,true,0,testinputs_1,testinputs_2,#testcorpus_1.sequences,output_file)
      totscore = totscore + score
      totscoreAll = totscoreAll + score/#testcorpus_1.sequences
      totalLines = totalLines + #testcorpus_1.sequences
    end
    print("mean score for "..fileList[fcnt].." = "..totscoreAll/config.max_cv.."\n")
    output_file:write("mean score for "..fileList[fcnt].." = "..totscoreAll/config.max_cv.."\n")
  end
end

print("mean score for lstm/"..config.dsize.." = "..totscore.."/"..totalLines.."="..totscore/totalLines.."\n")
output_file:write("mean score for lstm/"..config.dsize.." = "..totscore.."/"..totalLines.."="..totscore/totalLines.."\n")
output_file:write("\n\n")
io.close(output_file)

