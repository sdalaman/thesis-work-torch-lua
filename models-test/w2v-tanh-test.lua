require 'data-tanh-test'
require 'model-tanh'
require 'model-additive'
require 'fileListTr'
require 'fileListTst'
require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

path = "/home/saban/work/additive/"
path2 = "/home/saban/work/python/pytorch-works/additive/data/ted/"
model_path1 = path.."models/tanh"
model_path2 = model_path1
corpus_path1 = path.."models/corpus/"
corpus_path2 = path.."models/corpus/"

test_data_pathEnTrTok = path.."data/cdlc_en_tr/test/englishTok"
test_data_pathTrEnTok = path.."data/cdlc_en_tr/test/turkishTok"
test_data_pathEnTrMorph = path.."data/cdlc_en_tr/test/englishMorph"
test_data_pathTrEnMorph = path.."data/cdlc_en_tr/test/turkishMorph"

test_data_pathEnTrTokMulti = path2.."en-tr-tok/test"
test_data_pathTrEnTokMulti = path2.."tr-en-tok/test"
test_data_pathEnTrMorphMulti = path2.."en-tr-morph/test"
test_data_pathTrEnMorphMulti = path2.."tr-en-morph/test"

test_data_pathEnDeTok = path2.."en-de-tok/test"
test_data_pathDeEnTok = path2.."de-en-tok/test"
test_data_pathEnDeMorph = path2.."en-de-morph/test"
test_data_pathDeEnMorph = path2.."de-en-morph/test"

test_data_pathEnFrTok = path2.."en-fr-tok/test"
test_data_pathFrEnTok = path2.."fr-en-tok/test"
test_data_pathEnFrMorph = path2.."en-fr-morph/test"
test_data_pathFrEnMorph = path2.."fr-en-morph/test"

train_data_pathEn = path.."data/cdlc_en_tr/train/englishTok"
train_data_pathTr = path.."data/cdlc_en_tr/train/turkishTok"

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

corpusDeTok="train.de-en.de.tok.tanh"
corpusFrTok="train.en-fr.fr.tok.tanh"
corpusDeMorph="train.de-en.de.tok.morph.tanh"
corpusFrMorph="train.en-fr.fr.tok.morph.tanh"

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end


function test_model(model,final,epoch,inputs_pri,inputs_sec,test_size,output_file)
  local inputPri = inputs_pri[{{1,test_size},{}}]
  local inputSec = inputs_sec[{{1,test_size},{}}]
  local outputPri = model:getLegPri():forward(inputPri)
  local outputSec = model:getLegSec():forward(inputSec)
  all_rootsPri = outputPri:double()
  all_rootsSec = outputSec:double()

  list1 = {}
  list2 = {}
  score = 0
  for idxPri = 1, all_rootsPri:size(1) do
    closest = idxPri
    for idxSec = 1, all_rootsSec:size(1) do
      dist = torch.dist(all_rootsPri[idxPri],all_rootsSec[idxSec])
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
  
  print("- Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  print("\n")
  output_file:write("- Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  output_file:write("\n")
  return list1,list2,score
end

config = {}
config.corpus1 = "" -- input data for lang 1
config.corpus2 = "" -- input data  for lang 
config.model1 = "" -- model file  for lang 1
config.model2 = "" -- model file  for lang 2
config.lookuptable1 = "" -- lookup table file  for lang 1
config.lookuptable2 = "" -- lookup table file  for lang 2
config.emb_size = 0
config.test_size = 100
config.win_size = 2
config.out_file  = "test-file.txt"
config.max_cv = 10
config.dsize = "all"
config.dtype = "morph"
config.lang1 = "En"
config.lang2=  "Tr"
config.mdlImp=  ""  -- Tanh-1,Tanh-2,Add
config.seq_lenPri = 0
config.seq_lenSec = 0

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-emb_size", config.emb_size)
cmd:option("-test_size", config.test_size)
cmd:option("-win_size", config.win_size)
cmd:option("-out_file", config.out_file)
cmd:option("-max_cv", config.max_cv)
cmd:option("-dsize", config.dsize)
cmd:option("-dtype", config.dtype)
cmd:option("-lang1", config.lang1)
cmd:option("-lang2", config.lang2)
cmd:option("-mdlImp", config.mdlImp)
params = cmd:parse(arg)

-- -corpus1 english.all.tok.en-tu -corpus2 turkish.all.tok.en-tu -test_size 100 -max_cv 1 -out_file test-tanh-all-tok -dsize all -dtype tok -model1 1/english.all.tok.en-tu.tanh -lookuptable1 1/english.all.tok.en-tu.tanh -model2 1/turkish.all.tok.en-tu.tanh -lookuptable2 1/turkish.all.tok.en-tu.tanh -lang1 En -lang2 Tr -mdlImp Tanh-1

-- -corpus1 english.all.tok.morph.en-tu -corpus2 turkish.all.tok.morph.en-tu -test_size 100 -max_cv 1 -out_file test-tanh-all-morph -dsize all -dtype morph -model1 2/english.all.tok.morph.en-tu.0.01.1550.tanh.256 -lookuptable1 2/english.all.tok.morph.en-tu.0.01.1550.tanh.256 -model2 2/turkish.all.tok.morph.en-tu.0.01.1550.tanh.256 -lookuptable2 2/turkish.all.tok.morph.en-tu.0.01.1550.tanh.256 -lang1 En -lang2 Tr 


for param, value in pairs(params) do
    config[param] = value
end

if config.mdlImp ~= "Tanh-1" and config.mdlImp ~= "Tanh-2" and config.mdlImp ~= "Add" then
  print("mdlImp prm should be one of them [Tanh-1|Tanh-2|Add] \n")
  os.exit()
end

if config.dtype == "morph" then
  if config.lang1 == "En" then
    config.model1 = modelEnMorph
    config.corpus1 = corpusEnMorph
  end
  if config.lang1 == "Tr" then
    config.model1 = modelTrMorph
    config.corpus1 = corpusTrMorph
  end
  if config.lang1 == "De" then
    config.model1 = modelDeMorph
    config.corpus1 = corpusDeMorph
  end
  if config.lang1 == "Fr" then
    config.model1 = modelFrMorph
    config.corpus1 = corpusFrMorph
  end

  if config.lang2 == "En" then
    config.model2 = modelEnMorph
    config.corpus2 = corpusEnMorph
  end
  if config.lang2 == "Tr" then
    config.model2 = modelTrMorph
    config.corpus2 = corpusTrMorph
  end
  if config.lang2 == "De" then
    config.model2 = modelDeMorph
    config.corpus2 = corpusDeMorph
  end
  if config.lang2 == "Fr" then
    config.model2 = modelFrMorph
    config.corpus2 = corpusFrMorph
  end

else
  if config.lang1 == "En" then
    config.model1 = modelEnTok
    config.corpus1 = corpusEnTok
  end
  if config.lang1 == "Tr" then
    config.model1 = modelTrTok
    config.corpus1 = corpusTrTok
  end
  if config.lang1 == "De" then
    config.model1 = modelDeTok
    config.corpus1 = corpusDeTok
  end
  if config.lang1 == "Fr" then
    config.model1 = modelFrTok
    config.corpus1 = corpusFrTok
  end

  if config.lang2 == "En" then
    config.model2 = modelEnTok
    config.corpus2 = corpusEnTok
  end
  if config.lang2 == "Tr" then
    config.model2 = modelTrTok
    config.corpus2 = corpusTrTok
  end
  if config.lang2 == "De" then
    config.model2 = modelDeTok
    config.corpus2 = corpusDeTok
  end
  if config.lang2 == "Fr" then
    config.model2 = modelFrTok
    config.corpus2 = corpusFrTok
  end
end

if config.lang1 == "En" and config.lang2 == "Tr" then
  test_data_path1 = test_data_pathEnTrTok
  test_data_path2 = test_data_pathTrEnTok
end
if config.lang1 == "Tr" and config.lang2 == "En" then
  test_data_path1 = test_data_pathTrEnTok
  test_data_path2 = test_data_pathEnTrTok
end
if config.lang1 == "En" and config.lang2 == "De"  then
  test_data_path1 = test_data_pathEnDeTok
  test_data_path2 = test_data_pathDeEnTok
  corpus_path2 = corpus_path2.."multi/"
end
if config.lang1 == "De" and config.lang2 == "En"  then
  test_data_path1 = test_data_pathDeEnTok
  test_data_path2 = test_data_pathEnDeTok
  corpus_path1 = corpus_path1.."multi/"
end
if config.lang1 == "En" and config.lang2 == "Fr"  then
  test_data_path1 = test_data_pathEnFrTok
  test_data_path2 = test_data_pathFrEnTok
  corpus_path2 = corpus_path2.."multi/"
end
if config.lang1 == "Fr" and config.lang2 == "En"  then
  test_data_path1 = test_data_pathFrEnTok
  test_data_path2 = test_data_pathEnFrTok
  corpus_path1 = corpus_path1.."multi/"
end
if config.lang1 == "De" and config.lang2 == "Fr"  then
  test_data_path1 = test_data_pathDeEnTok
  test_data_path2 = test_data_pathFrEnTok
  corpus_path1 = corpus_path1.."multi/"
  corpus_path2 = corpus_path2.."multi/"
end
if config.lang1 == "Fr" and config.lang2 == "De"  then
  test_data_path1 = test_data_pathFrEnTok
  test_data_path2 = test_data_pathDeEnTok
  corpus_path1 = corpus_path1.."multi/"
  corpus_path2 = corpus_path2.."multi/"
end
if config.lang1 == "De" and config.lang2 == "Tr"  then
  test_data_path1 = test_data_pathDeEnTok
  test_data_path2 = test_data_pathTrEnTokMulti
  corpus_path1 = corpus_path1.."multi/"
end
if config.lang1 == "Tr" and config.lang2 == "De"  then
  test_data_path1 = test_data_pathTrEnTokMulti
  test_data_path2 = test_data_pathDeEnTok
  corpus_path2 = corpus_path2.."multi/"
end
if config.lang1 == "Fr" and config.lang2 == "Tr"  then
  test_data_path1 = test_data_pathFrEnTok
  test_data_path2 = test_data_pathTrEnTokMulti
  corpus_path1 = corpus_path1.."multi/"
end
if config.lang1 == "Tr" and config.lang2 == "Fr"  then
  test_data_path1 = test_data_pathTrEnTokMulti
  test_data_path2 = test_data_pathFrEnTok
  corpus_path2 = corpus_path2.."multi/"
end


if config.dtype == "morph" then
  if config.lang1 == "En" and config.lang2 == "Tr" then
    test_data_path1 = test_data_pathEnTrMorph
    test_data_path2 = test_data_pathTrEnMorph
  end
  if config.lang1 == "Tr" and config.lang2 == "En" then
    test_data_path1 = test_data_pathTrEnMorph
    test_data_path2 = test_data_pathEnTrMorph
  end
  if config.lang1 == "En" and config.lang2 == "De"  then
    test_data_path1 = test_data_pathEnDeMorph
    test_data_path2 = test_data_pathDeEnMorph
  end
  if config.lang1 == "De" and config.lang2 == "En"  then
    test_data_path1 = test_data_pathDeEnMorph
    test_data_path2 = test_data_pathEnDeMorph
  end
  if config.lang1 == "En" and config.lang2 == "Fr"  then
    test_data_path1 = test_data_pathEnFrMorph
    test_data_path2 = test_data_pathFrEnMorph
  end
  if config.lang1 == "Fr" and config.lang2 == "En"  then
    test_data_path1 = test_data_pathFrEnMorph
    test_data_path2 = test_data_pathEnFrMorph
  end
  if config.lang1 == "De" and config.lang2 == "Fr"  then
    test_data_path1 = test_data_pathDeEnMorph
    test_data_path2 = test_data_pathFrEnMorph
  end
  if config.lang1 == "Fr" and config.lang2 == "De"  then
    test_data_path1 = test_data_pathFrEnMorph
    test_data_path2 = test_data_pathDeEnMorph
  end
  if config.lang1 == "De" and config.lang2 == "Tr"  then
    test_data_path1 = test_data_pathDeEnMorph
    test_data_path2 = test_data_pathTrEnMorphMulti
  end
  if config.lang1 == "Tr" and config.lang2 == "De"  then
    test_data_path1 = test_data_pathTrEnMorphMulti
    test_data_path2 = test_data_pathDeEnMorph
  end
  if config.lang1 == "Fr" and config.lang2 == "Tr"  then
    test_data_path1 = test_data_pathFrEnMorph
    test_data_path2 = test_data_pathTrEnMorphMulti
  end
  if config.lang1 == "Tr" and config.lang2 == "Fr"  then
    test_data_path1 = test_data_pathTrEnMorphMulti
    test_data_path2 = test_data_pathFrEnMorph
  end
  
  model_path1 = model_path1.."-morph"
  model_path2 = model_path2.."-morph"
end

lang_pair = config.lang1.."-"..config.lang2
model_path1 = model_path1.."/"..config.dsize.."/"..config.lang1.."/"
model_path2 = model_path2.."/"..config.dsize.."/"..config.lang2.."/"

output_file = io.open(config.out_file.."-"..lang_pair..".txt", "w")
output_file:write("Model corpus: "..config.corpus1.."/"..config.corpus1.."\n")
output_file:write("Test size: "..config.test_size.."\n")
summary_file = io.open("summary.tanh."..config.dsize..".txt", "a+")

--nm = string.sub(config.corpus1,1,2).."-"..string.sub(config.corpus2,1,2)
--config.vocab1 = corpus_path..config.corpus1.."."..nm..".vocab" -- vocab file  for lang 1
--config.vocab2 = corpus_path..config.corpus2.."."..nm..".vocab" -- vocab file  for lang 2
config.model1 = model_path1..config.model1 --model file  for lang 1
config.model2 = model_path2..config.model2 -- model file  for lang 2
config.lookuptable1 = config.model1 -- lookup table file  for lang 1
config.lookuptable2 = config.model2 -- lookup table file  for lang 2

corpus_name_1 = corpus_path1..config.corpus1..".corpus.tch"
corpus_name_2 = corpus_path2..config.corpus2..".corpus.tch"
config.model1 = config.model1..".model"
config.model2 = config.model2..".model"
config.lookuptable1 = config.lookuptable1..".LT"
config.lookuptable2 = config.lookuptable2..".LT"

for i,j in pairs(config) do
    print(i..": "..j)
end

sellist = {}

print("\ncorpus data will be loaded\n")
print(corpus_name_1.." "..corpus_name_2)
corpus_1 = torch.load(corpus_name_1)
corpus_2 = torch.load(corpus_name_2)
print("\ncorpus data loaded\n")

print("training corpus\n")
print(corpus_name_1.." "..corpus_1.no_of_words.." words, "..#corpus_1.sequences.." sents, "..corpus_1.longest_seq.." sent max lenght\n")
print(corpus_name_2.." "..corpus_2.no_of_words.." words, "..#corpus_2.sequences.." sents, "..corpus_2.longest_seq.." sent max lenght\n")

print("model data will be loaded\n")
print(config.lookuptable1.." "..config.lookuptable2.."\n")
local ltPri = torch.load(config.lookuptable1)
local ltSec = torch.load(config.lookuptable2)
config.emb_size = ltPri.weight:size(2)

print("Model implementation : "..config.mdlImp.."\n")
if config.mdlImp == "Tanh-1" then
  config.seq_lenPri = corpus_1.longest_seq
  if config.lang2 == "De" or config.lang2 == "Fr" then
    config.seq_lenSec = 100
  else
    config.seq_lenSec = corpus_2.longest_seq
  end
  biLnModel = BiLangModelTanh(corpus_1.no_of_words,corpus_2.no_of_words,config)
  biLnModel.legPri = torch.load(config.model1)
  biLnModel.legSec = torch.load(config.model2)
  --biLnModel.ltPri.weight:copy(ltPri.weight)
  --biLnModel.ltSec.weight:copy(ltSec.weight)
end

if config.mdlImp == "Add" then
  biLnModel = BiLangModelAdditive(corpus_1.no_of_words,corpus_2.no_of_words,config)
  biLnModel.ltPri.weight:copy(ltPri.weight)
  biLnModel.ltSec.weight:copy(ltSec.weight)
end


totalLines = 0
totscore = 0

if config.lang1 == "En" or config.lang2 == "En" then
  if config.lang1 == "Tr" or config.lang2 == "Tr" then
    fileList = fileListTstEnTr
  end
  if config.lang1 == "De" or config.lang2 == "De" then
    fileList = fileListTstEnDe
  end
  if config.lang1 == "Fr" or config.lang2 == "Fr" then
    fileList = fileListTstEnFr
  end
end

if config.lang1 ~= "En" and config.lang2 ~= "En" then
  fileList = fileListTstMulti
end

notfound = 0
for fcnt = 1,#fileList do
  totscoreAll = 0
  config.testcorpus1 = fileList[fcnt]..".tok"
  config.testcorpus2 = fileList[fcnt]..".tok"
  if config.dtype == "morph" then
    config.testcorpus1 = config.testcorpus1..".morph"
    config.testcorpus2 = config.testcorpus2..".morph"
  end
  f1,f1err = io.open(test_data_path1..config.testcorpus1,'r') 
  f2,f2err = io.open(test_data_path2..config.testcorpus2,'r')
  if f1 and f2 then
    f1:close()
    f2:close()
    for cv = 1,config.max_cv do
      sellist = {}
      print("test corpus data will be prepared\n")
      testcorpus_1 = Corpus(config.testcorpus1,test_data_path1)
      testcorpus_2 = Corpus(config.testcorpus2,test_data_path2)
      testcorpus_1:prepare_corpus(nil,config.test_size)
      testcorpus_2:prepare_corpus(nil,config.test_size)
      testcorpus_name_1 = config.testcorpus1
      testcorpus_name_2 = config.testcorpus2
      print("testcorpus data prepared\n")
  
      if config.mdlImp == "Tanh-1" then
        testcorpus_1.longest_seq = config.seq_lenPri
        testcorpus_2.longest_seq = config.seq_lenSec
      end

      testinputs_1= testcorpus_shapedata(corpus_1,testcorpus_1):cuda()
      testinputs_2= testcorpus_shapedata(corpus_2,testcorpus_2):cuda()

      print("test corpus\n")
      print(testcorpus_name_1.." "..testcorpus_1.no_of_words.." words, "..#testcorpus_1.sequences.." sents, "..testcorpus_1.longest_seq.." sent max lenght\n")
      print(testcorpus_name_2.." "..testcorpus_2.no_of_words.." words, "..#testcorpus_2.sequences.." sents, "..testcorpus_2.longest_seq.." sent max lenght\n")

      if config.mdlImp == "Tanh-2" then
      -- Create model evertime with new parameters and load lookuptable weights
        config.seq_lenPri = testcorpus_1.longest_seq
        config.seq_lenSec = testcorpus_2.longest_seq
        biLnModel = BiLangModelTanh(corpus_1.no_of_words,corpus_2.no_of_words,config)
        biLnModel.ltPri.weight:copy(ltPri.weight)
        biLnModel.ltSec.weight:copy(ltSec.weight)
      end

      crossList,closeList,score=test_model(biLnModel,true,0,testinputs_1,testinputs_2,#testcorpus_1.sequences,output_file)
      totscore = totscore + score
      totscoreAll = totscoreAll + score/#testcorpus_1.sequences
      totalLines = totalLines + #testcorpus_1.sequences
    end
    print("mean score for "..fileList[fcnt].." = "..totscoreAll/config.max_cv.."\n")
    output_file:write("mean score for "..fileList[fcnt].." = "..totscoreAll/config.max_cv.."\n")
  else
    print(config.testcorpus1.." not found \n")
    notfound = notfound + 1
  end
end
print(notfound .." files not found \n")

print("mean score for tanh/"..config.dsize.." = "..totscore.."/"..totalLines.."="..totscore/totalLines.."\n")
output_file:write("mean score for tanh/"..config.dsize.." = "..totscore.."/"..totalLines.."="..totscore/totalLines.."\n")
output_file:write("\n\n")
summary_file:write("\n")
summary_file:write(config.dsize..","..config.dtype..","..config.mdlImp..","..config.win_size..","..config.lang1..","..config.lang2..","..totscore.."/"..totalLines..","..totscore/totalLines)
summary_file:write("\n")

io.close(output_file)
io.close(summary_file)


