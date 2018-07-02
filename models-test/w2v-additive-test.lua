require 'data-additive-test'
require 'model-additive'
require 'fileListTr'
require 'fileListTst'
require 'fileListTstSmp'
require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

path = "/home/saban/work/additive/"
path2 = "/home/saban/work/python/pytorch-works/additive/data/ted/"
model_path1 = path.."models/additive"
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


train_data_pathEnTok = path.."data/cdlc_en_tr/train/englishTok"
train_data_pathTrTok = path.."data/cdlc_en_tr/train/turkishTok"

train_data_pathEnMorph = path.."data/cdlc_en_tr/train/englishMorph"
train_data_pathTrMorph = path.."data/cdlc_en_tr/train/turkishMorph"

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

vocabDeTok="train.de-en.de.tok"
vocabFrTok="train.en-fr.fr.tok"
vocabDeMorph="train.de-en.de.tok.morph"
vocabFrMorph="train.en-fr.fr.tok.morph"

corpusEnTok="english.all.tok.en-tu"
corpusTrTok="turkish.all.tok.en-tu"
corpusEnMorph="english.all.tok.morph.en-tu"
corpusTrMorph="turkish.all.tok.morph.en-tu"

corpusDeTok="train.de-en.de.tok"
corpusFrTok="train.en-fr.fr.tok"
corpusDeMorph="train.de-en.de.tok.morph"
corpusFrMorph="train.en-fr.fr.tok.morph"

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

-- Default configuration
config = {}
config.corpus1 = "" -- input data for lang 1
config.corpus2 = "" -- input data  for lang 
config.model1 = "" -- model file  for lang 1
config.model2 = "" -- model file  for lang 2
config.lookuptable1 = "" -- lookup table file  for lang 1
config.lookuptable2 = "" -- lookup table file  for lang 2
config.test_size = 100
config.emb_size = 0
config.out_file  = "test-file.txt"
config.max_cv = 10
config.dsize = "all"
config.dtype = "morph"
config.lang1 = "En"
config.lang2= "Tr"

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-test_size", config.test_size)
cmd:option("-out_file", config.out_file)
cmd:option("-max_cv", config.max_cv)
cmd:option("-dsize", config.dsize)
cmd:option("-dtype", config.dtype)
cmd:option("-lang1", config.lang1)
cmd:option("-lang2", config.lang2)
params = cmd:parse(arg)

---corpus1 english.all.tok.en-tu -corpus2 turkish.all.tok.en-tu -test_size 100 -max_cv 1 -out_file test-additive-all-tok-1 -dsize all -dtype tok -model1  1/english.all.tok.en-tu.model -lookuptable1 1/english.all.tok.en-tu.LT -model2  1/turkish.all.tok.en-tu.model -lookuptable2 1/turkish.all.tok.en-tu.LT  -lang1 En -lang2 Tr 

for param, value in pairs(params) do
    config[param] = value
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
summary_file = io.open("summary.additive."..config.dsize..".txt", "a+")


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

tmp = torch.load(config.lookuptable1)
config.emb_size = tmp.weight:size(2)
tmp = {}

biLnModel = BiLangModelAdditive(corpus_1.no_of_words,corpus_2.no_of_words,config)

print("model data will be loaded\n")
print(config.model1.." "..config.model2.."\n")
print(config.lookuptable1.." "..config.lookuptable2.."\n")
biLnModel.legPri = torch.load(config.model1)
biLnModel.legSec = torch.load(config.model2)
--biLnModel.ltPri = torch.load(config.lookuptable1)
--biLnModel.ltSec = torch.load(config.lookuptable2)
config.emb_size = biLnModel.ltPri.weight:size(2)
print("model data loaded\n")

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
  config.testcorpus1 = fileList[fcnt]..".tok"
  config.testcorpus2 = fileList[fcnt]..".tok"
  if config.dtype == "morph" then
    config.testcorpus1 = config.testcorpus1..".morph"
    config.testcorpus2 = config.testcorpus2..".morph"
  end
  sellist = {}
  f1,f1err = io.open(test_data_path1..config.testcorpus1,'r') 
  f2,f2err = io.open(test_data_path2..config.testcorpus2,'r')
  if f1 and f2 then
    f1:close()
    f2:close()
    print("test corpus data will be prepared\n")
    testcorpus_1 = Corpus(config.testcorpus1,test_data_path1)
    testcorpus_2 = Corpus(config.testcorpus2,test_data_path2)
    testcorpus_1:prepare_corpus(nil,config.test_size)
    testcorpus_2:prepare_corpus(nil,config.test_size)
    testcorpus_name_1 = config.testcorpus1
    testcorpus_name_2 = config.testcorpus2
    print("testcorpus data prepared\n")
    testinputs_1= testcorpus_shapedata(corpus_1,testcorpus_1):cuda()
    testinputs_2= testcorpus_shapedata(corpus_2,testcorpus_2):cuda()

    print(testcorpus_name_1.." "..testcorpus_1.no_of_words.." words, "..#testcorpus_1.sequences.." sents, "..testcorpus_1.longest_seq.." sent max lenght\n")
    print(testcorpus_name_2.." "..testcorpus_2.no_of_words.." words, "..#testcorpus_2.sequences.." sents, "..testcorpus_2.longest_seq.." sent max lenght\n")
    
    if #testcorpus_1.sequences == #testcorpus_2.sequences then
      totscoreAll = 0
      for cv = 1,config.max_cv do
        crossList,closeList,score=test_model(biLnModel,true,0,testinputs_1,testinputs_2,#testcorpus_1.sequences,output_file)
        totscore = totscore + score
        totscoreAll = totscoreAll + score/#testcorpus_1.sequences
        totalLines = totalLines + #testcorpus_1.sequences
      end
      print("mean score for "..fileList[fcnt].." = "..totscoreAll/config.max_cv.."\n")
      output_file:write("mean score for "..fileList[fcnt].." = "..totscoreAll/config.max_cv.."\n")
    else
      print("# of sentences in each file is different \n")
    end
  else
    print(config.testcorpus1.." not found \n")
    notfound = notfound + 1
  end
end

print(notfound .." files not found \n")

print("mean score for additive/"..config.dsize.." = "..totscore.."/"..totalLines.."="..totscore/totalLines.."\n")
output_file:write("mean score for additive/"..config.dsize.." = "..totscore.."/"..totalLines.."="..totscore/totalLines.."\n")
output_file:write("\n\n")

summary_file:write("\n")
summary_file:write(config.dsize..","..config.dtype..","..config.lang1..","..config.lang2..","..totscore.."/"..totalLines..","..totscore/totalLines)
summary_file:write("\n")
io.close(output_file)
io.close(summary_file)


--[[totscore = 0
for k,v in pairs(cvscore) do
  print(" "..k..".."..v.."\n")
  totscore = totscore + v
end
per = totscore/(config.max_cv * #testcorpus_1.sequences)
print("mean score "..totscore/config.max_cv.."/"..#testcorpus_1.sequences.." "..per.."%".."\n")
output_file:write("Mean score "..totscore/config.max_cv.."/"..#testcorpus_1.sequences.." "..per.."%".."\n")
]]--

--[[output_file:write("\n\n")
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
]]--

