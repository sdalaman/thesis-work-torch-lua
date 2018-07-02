require 'data-lstm-test'
require 'model-lstm-drop'
require 'fileListTr'
require 'fileListTrShort'
require 'fileListTst'
--require 'modelCorpusPaths'
require 'nn'
require 'cunn'
require 'rnn'

require("io")
require("os")
require("paths")

path = "/home/saban/work/additive/"
path2 = "/home/saban/work/python/pytorch-works/additive/data/ted/"
model_path1 = path.."models/lstm"
--model_path1 = path.."model-impl/lstm-models/model-files"
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

train_data_pathEnTrTok = path.."data/cdlc_en_tr/train/englishTok"
train_data_pathTrEnTok = path.."data/cdlc_en_tr/train/turkishTok"

corpusEnTokLstm="english.10000.tok.en-tu.60.lstm"
corpusTrTokLstm="turkish.10000.tok.en-tu.60.lstm"
corpusEnMorphLstm="english.10000.tok.morph.en-tu.60.lstm"
corpusTrMorphLstm="turkish.10000.tok.morph.en-tu.60.lstm"

corpusEnTokBiLstm="english.10000.tok.en-tu.60.BiLstm"
corpusTrTokBiLstm="turkish.10000.tok.en-tu.60.BiLstm"
corpusEnMorphBiLstm="english.10000.tok.morph.en-tu.60.BiLstm"
corpusTrMorphBiLstm="turkish.10000.tok.morph.en-tu.60.BiLstm"

corpusDeTokLstm="de-en.de.10000.60.lstm"
corpusFrTokLstm="en-fr.fr.10000.60.lstm"
corpusDeMorphLstm="de-en.de.10000.morph.60.lstm"
corpusFrMorphLstm="en-fr.fr.10000.morph.60.lstm"

corpusDeTokBiLstm="de-en.de.10000.60.Bilstm"
corpusFrTokBiLstm="en-fr.fr.10000.60.Bilstm"
corpusDeMorphBiLstm="de-en.de.10000.morph.60.Bilstm"
corpusFrMorphBiLstm="en-fr.fr.10000.morph.60.Bilstm"

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
  local outputPri = model:modelPri():forward(inputPri)
  local outputSec = model:modelSec():forward(inputSec)
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
config.emb_size = 64
config.hidden_size = 64
config.test_size = 100
config.out_file  = "test-lstm-file.txt"
config.max_cv = 10
config.win_size = 1
config.init = 1
config.dsize = "all"
config.dtype = "tok"
config.lstmtype = "LSTM"
config.seqlen = 30
config.num_of_hidden = 3
config.lang1 = "En"
config.lang2= "Tr"

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-emb_size", config.emb_size)
cmd:option("-test_size", config.test_size)
cmd:option("-out_file", config.out_file)
cmd:option("-max_cv", config.max_cv)
cmd:option("-dsize", config.dsize)
cmd:option("-dtype", config.dtype)
cmd:option("-lstmtype", config.lstmtype)
cmd:option("-seqlen", config.seqlen)
cmd:option("-lang1", config.lang1)
cmd:option("-lang2", config.lang2)
params = cmd:parse(arg)


---corpus1 english.all.tok.en-tu.60.lstm -corpus2 turkish.all.tok.en-tu.60.lstm -test_size 100 -max_cv 1 -out_file test-lstm-10000 -dsize all -dtype tok -lstmtype LSTMScAvg -seqlen 60 -model1 1-Sc-Avg-Tok/en-tu_en_0.001_250.lstm.128.128.60.lstm-ScAvg-drop-all -lookuptable1 1-Sc-Avg-Tok/en-tu_en_0.001_250.lstm.128.128.60.lstm-ScAvg-drop-all -model2 1-Sc-Avg-Tok/en-tu_tu_0.001_250.lstm.128.128.60.lstm-ScAvg-drop-all -lookuptable2 1-Sc-Avg-Tok/en-tu_tu_0.001_250.lstm.128.128.60.lstm-ScAvg-drop-all  -lang1 En -lang2 Tr 

for param, value in pairs(params) do
    config[param] = value
end

if config.dtype == "morph" then
  if config.lang1 == "En" then
    if config.lstmtype == "Lstm" then
      config.model1 = modelEnMorphLstm
      config.corpus1 = corpusEnMorphLstm
    else
      config.model1 = modelEnMorphBiLstm
      config.corpus1 = corpusEnMorphBiLstm
    end
  end
  if config.lang1 == "Tr" then
    if config.lstmtype == "Lstm" then
      config.model1 = modelTrMorphLstm
      config.corpus1 = corpusTrMorphLstm
    else
      config.model1 = modelTrMorphBiLstm
      config.corpus1 = corpusTrMorphBiLstm
    end
  end
  if config.lang1 == "De" then
    if config.lstmtype == "Lstm" then
      config.model1 = modelDeMorphLstm
      config.corpus1 = corpusDeMorphLstm
    else
      config.model1 = modelDeMorphBiLstm
      config.corpus1 = corpusDeMorphBiLstm
    end
  end
  if config.lang1 == "Fr" then
    if config.lstmtype == "Lstm" then
      config.model1 = modelFrMorphLstm
      config.corpus1 = corpusFrMorphLstm
    else
      config.model1 = modelFrMorphBiLstm
      config.corpus1 = corpusFrMorphBiLstm
    end
  end

  if config.lang2 == "En" then
    if config.lstmtype == "Lstm" then
      config.model2 = modelEnMorphLstm
      config.corpus2 = corpusEnMorphLstm
    else
      config.model2 = modelEnMorphBiLstm
      config.corpus2 = corpusEnMorphBiLstm
    end
  end
  if config.lang2 == "Tr" then
    if config.lstmtype == "Lstm" then
      config.model2 = modelTrMorphLstm
      config.corpus2 = corpusTrMorphLstm
    else
      config.model2 = modelTrMorphBiLstm
      config.corpus2 = corpusTrMorphBiLstm
    end
  end
  if config.lang2 == "De" then
    if config.lstmtype == "Lstm" then
      config.model2 = modelDeMorphLstm
      config.corpus2 = corpusDeMorphLstm
    else
      config.model2 = modelDeMorphBiLstm
      config.corpus2 = corpusDeMorphBiLstm
    end
  end
  if config.lang2 == "Fr" then
    if config.lstmtype == "Lstm" then
      config.model2 = modelFrMorphLstm
      config.corpus2 = corpusFrMorphLstm
    else
      config.model2 = modelFrMorphBiLstm
      config.corpus2 = corpusFrMorphBiLstm
    end
  end

else
  if config.lang1 == "En" then
    if config.lstmtype == "Lstm" then
      config.model1 = modelEnTokLstm
      config.corpus1 = corpusEnTokLstm
    else
      config.model1 = modelEnTokBiLstm
      config.corpus1 = corpusEnTokBiLstm
    end
  end
  if config.lang1 == "Tr" then
    if config.lstmtype == "Lstm" then
      config.model1 = modelTrTokLstm
      config.corpus1 = corpusTrTokLstm
    else
      config.model1 = modelTrTokBiLstm
      config.corpus1 = corpusTrTokBiLstm
    end
  end
  if config.lang1 == "De" then
    if config.lstmtype == "Lstm" then
      config.model1 = modelDeTokLstm
      config.corpus1 = corpusDeTokLstm
    else
      config.model1 = modelDeTokBiLstm
      config.corpus1 = corpusDeTokBiLstm
    end
  end
  if config.lang1 == "Fr" then
    if config.lstmtype == "Lstm" then
      config.model1 = modelFrTokLstm
      config.corpus1 = corpusFrTokLstm
    else
      config.model1 = modelFrTokBiLstm
      config.corpus1 = corpusFrTokBiLstm
    end
  end

  if config.lang2 == "En" then
    if config.lstmtype == "Lstm" then
      config.model2 = modelEnTokLstm
      config.corpus2 = corpusEnTokLstm
    else
      config.model2 = modelEnTokBiLstm
      config.corpus2 = corpusEnTokBiLstm
    end
  end
  if config.lang2 == "Tr" then
    if config.lstmtype == "Lstm" then
      config.model2 = modelTrTokLstm
      config.corpus2 = corpusTrTokLstm
    else
      config.model2 = modelTrTokBiLstm
      config.corpus2 = corpusTrTokBiLstm
    end
  end
  if config.lang2 == "De" then
    if config.lstmtype == "Lstm" then
      config.model2 = modelDeTokLstm
      config.corpus2 = corpusDeTokLstm
    else
      config.model2 = modelDeTokBiLstm
      config.corpus2 = corpusDeTokBiLstm
    end
  end
  if config.lang2 == "Fr" then
    if config.lstmtype == "Lstm" then
      config.model2 = modelFrTokLstm
      config.corpus2 = corpusFrTokLstm
    else
      config.model2 = modelFrTokBiLstm
      config.corpus2 = corpusFrTokBiLstm
    end
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
summary_file = io.open("summary.lstm."..config.dsize..".txt", "a+")

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

seq_lenPri = config.seqlen
seq_lenSec = config.seqlen
config.max_seq_len = config.seqlen
config.win_size = 1

tmp = torch.load(config.lookuptable1)
config.emb_size = tmp.weight:size(2)
tmp = {}

if config.lstmtype == "LstmX" then
  biLnModel = BiLangModelLSTM(corpus_1.no_of_words,corpus_2.no_of_words,seq_lenPri,seq_lenSec,config)  
elseif config.lstmtype == "Lstm" then
  biLnModel = BiLangModelLSTMScAvg(corpus_1.no_of_words,corpus_2.no_of_words,seq_lenPri,seq_lenSec,config)  
elseif config.lstmtype == "BiLstm" then
  biLnModel = BiLangModelBiLSTMScAvg2(corpus_1.no_of_words,corpus_2.no_of_words,seq_lenPri,seq_lenSec,config)  
end

print("model data will be loaded\n")
print(config.model1.." "..config.model2.."\n")
print(config.lookuptable1.." "..config.lookuptable2.."\n")
if config.lstmtype == "BiLstm" then
  biLnModel.biSeqPri = torch.load(config.model1)
  biLnModel.biSeqSec = torch.load(config.model2)
  biLnModel.biSeqPri:evaluate()
  biLnModel.biSeqSec:evaluate()
else
  biLnModel.mdlPri = torch.load(config.model1)
  biLnModel.mdlSec = torch.load(config.model2)
  biLnModel.mdlPri:evaluate()
  biLnModel.mdlSec:evaluate()
end
--biLnModel.ltPri = torch.load(config.lookuptable1)
--biLnModel.ltSec = torch.load(config.lookuptable2)
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
    testcorpus_1.longest_seq = config.seqlen
    testcorpus_2.longest_seq = config.seqlen
    print("testcorpus data prepared\n")
    if config.lstmtype == "BiLstmScAvg" then
      testinputs_1= testcorpus_shapedataBiLSTM(corpus_1,testcorpus_1):cuda()
      testinputs_2= testcorpus_shapedataBiLSTM(corpus_2,testcorpus_2):cuda()
    else
      testinputs_1= testcorpus_shapedataLSTM(corpus_1,testcorpus_1):cuda()
      testinputs_2= testcorpus_shapedataLSTM(corpus_2,testcorpus_2):cuda()
    end
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
  else
    print(config.testcorpus1.." not found \n")
    notfound = notfound + 1
  end
end

print("mean score for lstm/"..config.dsize.." = "..totscore.."/"..totalLines.."="..totscore/totalLines.."\n")
output_file:write("mean score for lstm/"..config.dsize.." = "..totscore.."/"..totalLines.."="..totscore/totalLines.."\n")
output_file:write("\n\n")
--summary_file:write("\n")
summary_file:write(config.dsize..","..config.dtype..","..config.lstmtype..","..config.seqlen..","..config.lang1..","..config.lang2..","..totscore.."/"..totalLines..","..totscore/totalLines.."\n")
--summary_file:write("\n")
io.close(output_file)
io.close(summary_file)

