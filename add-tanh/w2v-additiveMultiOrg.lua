require 'data-add-tanh'
require 'model-additive'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'

require 'io'
require 'os'
require 'paths'

path = "/home/saban/work/additive/"
model_save_path = path.."model-impl/add-tanh/model-files/"
model_load_path = path.."model-impl/add-tanh/model-files/"
corpus_path = path.."models/corpus/multi/"
data_path = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/corpusfiles/"

test_size = 100
cv_max = 10

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

function test_model(model,final,epoch,inputs_pri,inputs_sec,inputs_thrd,inputs_frth,test_size)
  inPri = torch.zeros(test_size,inputs_pri:size()[2]):cuda()
  inSec = torch.zeros(test_size,inputs_sec:size()[2]):cuda()
  inThrd = torch.zeros(test_size,inputs_thrd:size()[2]):cuda()
  inFrth = torch.zeros(test_size,inputs_frth:size()[2]):cuda()
  
  cl = inputs_pri:size()[1]
  math.randomseed( os.time() )

  for i=1,test_size do
    ln = math.random(1,cl)
    inPri[i] = inputs_pri[ln]
    inSec[i] = inputs_sec[ln]
    inThrd[i] = inputs_thrd[ln]
    inFrth[i] = inputs_frth[ln]
  end

  --local inputPri = inputs_pri[{{1,1000},{}}]
  --local inputSec = inputs_sec[{{1,1000},{}}]
  local outputPri = model:getLegPri():forward( inPri)
  local outputSec = model:getLegSec():forward( inSec)
  local outputThrd = model:getLegThrd():forward( inThrd)
  local outputFrth = model:getLegFrth():forward( inFrth)
  
  all_rootsPri = outputPri:double()
  all_rootsSec = outputSec:double()
  all_rootsThrd = outputThrd:double()
  all_rootsFrth = outputFrth:double()

  function accScore(all_roots1,all_roots2)
    local score = 0
    for idx1 = 1, all_roots1:size(1) do
      closest = idx1
      for idx2 = 1, all_roots2:size(1) do
        if torch.dist(all_roots1[idx1],all_roots2[idx2]) < torch.dist(all_roots1[idx1],all_roots2[closest]) then
          closest = idx2
        end
      end
    
      if idx1 == closest then
        score = score + 1 
      end
    end
    return score
  end
  
  score1 = accScore(all_rootsPri,all_rootsSec)
  score2 = accScore(all_rootsPri,all_rootsSec)
  score3 = accScore(all_rootsPri,all_rootsThrd)
  score4 = accScore(all_rootsPri,all_rootsFrth)
  
  print("Epoch "..epoch.."  Test Score1: " .. score1.. '/' .. all_rootsPri:size(1))
  print("Epoch "..epoch.."  Test Score2: " .. score2.. '/' .. all_rootsPri:size(1))
  print("Epoch "..epoch.."  Test Score3: " .. score3.. '/' .. all_rootsPri:size(1))
  print("Epoch "..epoch.."  Test Score4: " .. score4.. '/' .. all_rootsPri:size(1))
  print("\n")
  return score1,score2,score3,score4,all_rootsPri:size(1)
end

function cv_test(cv,model,maxepoch,insPri,insSec,insThrd,insFrth,test_size)
  tot1 = 0
  tot2 = 0
  tot3 = 0
  tot4 = 0
  for i = 1,cv do
    sc1,sc2,sc3,sc4,nsum = test_model(model,true,maxepoch,insPri,insSec,insThrd,insFrth,test_size)
    tot1 = tot1 + sc1
    tot2 = tot2 + sc2
    tot3 = tot3 + sc3
    tot4 = tot4 + sc4
  end
  print("-------"..cv.." Epochs avg1: "..tot1/cv.."\n")
  print("-------"..cv.." Epochs avg2: "..tot2/cv.."\n")
  print("-------"..cv.." Epochs avg3: "..tot3/cv.."\n")
  print("-------"..cv.." Epochs avg4: "..tot4/cv.."\n")
  return sc1,sc2,sc3,sc4,nsum,tot1/cv,tot2/cv,tot3/cv,tot4/cv
end

function train_model(model,corpus_pri,corpus_sec,corpus_thrd,corpus_frth,inputs_pri,inputs_sec,inputs_thrd,inputs_frth,config)
--  no_of_sents = #corpus_pri.sequences
  print ( 'Training started \n')
  beginning_time = torch.tic()
  --split = nn.SplitTable(2)
  additivePri = model:getLegPri()
  additiveSec = model:getLegSec()
  additiveThrd = model:getLegThrd()
  additiveFrth = model:getLegFrth()
  
  criterionPri = model:getCriterionPri()
  criterionSec = model:getCriterionSec()
  criterionThrd = model:getCriterionThrd()
  criterionFrth = model:getCriterionFrth()
  
  k=0
  no_of_sents = math.floor(#corpus_pri.sequences/config.batch_size)*config.batch_size
  for i =config.prev_epoch + 1,config.max_epoch do
    errors1 = {}
    errors2 = {}
    errors3 = {}
    errors4 = {}
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
        local inputThrd = inputs_thrd[{{start,endd},{}}]
        local inputFrth = inputs_frth[{{start,endd},{}}]
        
        additivePri:zeroGradParameters()
        additiveSec:zeroGradParameters()
        additiveThrd:zeroGradParameters()
        additiveFrth:zeroGradParameters()
        -- print( target)
       
        criterion = criterionPri
        local outputPri = additivePri:forward( inputPri)
        local outputSec = additiveSec:forward( inputSec)
        local err = criterion:forward( outputPri, outputSec)
        table.insert( errors1, err)
        local gradOutputsPri = criterion:backward(outputPri, outputSec)
        additivePri:backward(inputPri, gradOutputsPri)
        additivePri:updateParameters(config.lr)
        
        criterion = criterionSec
        outputPri = additivePri:forward( inputPri)
        outputSec = additiveSec:forward( inputSec)
        err = criterion:forward( outputSec, outputPri)
        table.insert( errors2, err)
        local gradOutputsSec = criterion:backward(outputSec, outputPri)
        additiveSec:backward(inputSec, gradOutputsSec)
        additiveSec:updateParameters(config.lr)
    
        criterion = criterionThrd
        outputPri = additivePri:forward( inputPri)
        outputThrd = additiveThrd:forward( inputThrd)
        err = criterion:forward( outputThrd, outputPri)
        table.insert( errors3, err)
        local gradOutputsThrd = criterion:backward(outputThrd, outputPri)
        additiveThrd:backward(inputThrd, gradOutputsThrd)
        additiveThrd:updateParameters(config.lr)

        criterion = criterionFrth
        outputPri = additivePri:forward( inputPri)
        outputFrth = additiveFrth:forward( inputFrth)
        err = criterion:forward( outputFrth, outputPri)
        table.insert( errors4, err)
        local gradOutputsFrth = criterion:backward(outputFrth, outputPri)
        additiveFrth:backward(inputFrth, gradOutputsFrth)
        additiveFrth:updateParameters(config.lr)

    end
    --printf ( 'epoch %4d, loss = %6.50f \n', i, torch.mean( torch.Tensor( errors))   )
    frm = 'epoch %4d, lr %5.10f, loss %6.20f, loss %6.20f, loss %6.20f, loss %6.20f \n'
    outStr = frm:format(i, config.lr,torch.mean( torch.Tensor( errors1)),torch.mean( torch.Tensor( errors2)),torch.mean( torch.Tensor( errors3)),torch.mean( torch.Tensor( errors4)))
    printf (outStr )
    loss_logger:add{['lr'] = config.lr,['training error mean1'] = torch.mean( torch.Tensor( errors1)),['training error mean2'] = torch.mean( torch.Tensor( errors2)),['training error mean3'] = torch.mean( torch.Tensor( errors3)),['training error mean4'] = torch.mean( torch.Tensor( errors4))}
    loss_logger:style{['lr'] = '+-',['training error mean1'] = '+-',['training error mean2'] = '+-',['training error mean3'] = '+-',['training error mean4'] = '+-'}

    if i % config.dump_freq == 0 then 
      sc1,sc2,sc3,sc4,num_of_samples,avg1,avg2,avg3,avg4 = cv_test(cv_max,model,i,inputs_pri,inputs_sec,inputs_thrd,inputs_frth,test_size)
      score_logger:add{['lr'] = config.lr, ['score1 '] = avg1, ['score2 '] = avg2, ['score3 '] = avg3, ['score4 '] = avg4 }
      score_logger:style{['lr'] = '+-',['score1 '] = '+-',['score2 '] = '+-',['score3 '] = '+-',['score4 '] = '+-'}
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

function model_save(cfg,epoch,model)
  nm = string.sub(cfg.corpus1,1,2).."-"..string.sub(cfg.corpus2,1,2).."-"..string.sub(cfg.corpus3,1,2).."-"..string.sub(cfg.corpus4,1,2)
  model1nm = model_save_path..cfg.corpus1.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".model"-- model file  for lang 1
  model2nm = model_save_path..cfg.corpus2.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".model" -- model file  for lang 2
  model3nm = model_save_path..cfg.corpus3.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".model"-- model file  for lang 3
  model4nm = model_save_path..cfg.corpus4.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".model" -- model file  for lang 4
  
  lookuptable1nm = model_save_path..cfg.corpus1.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".LT" -- lookup table file  for lang 1
  lookuptable2nm = model_save_path..cfg.corpus2.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".LT" -- lookup table file  for lang 2
  lookuptable3nm = model_save_path..cfg.corpus3.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".LT" -- lookup table file  for lang 3
  lookuptable4nm = model_save_path..cfg.corpus4.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".LT" -- lookup table file  for lang 4
  
  model:getLegSec():clearState()
  torch.save(model2nm,model:getLegSec())
  torch.save(lookuptable2nm,model:getLookupTableSec())
  model:getLegPri():clearState()
  torch.save(model1nm,model:getLegPri())
  torch.save(lookuptable1nm,model:getLookupTablePri())
  model:getLegThrd():clearState()
  torch.save(model3nm,model:getLegThrd())
  torch.save(lookuptable3nm,model:getLookupTableThrd())
  model:getLegFrth():clearState()
  torch.save(model4nm,model:getLegFrth())
  torch.save(lookuptable4nm,model:getLookupTableFrth())
  
  print("model data saved\n")
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
config.corpus2 = "" -- input data  for lang 2
config.corpus3 = "" -- input data  for lang 3 
config.corpus4 = "" -- input data  for lang 4 
config.model1 = "" -- model file  for lang 1
config.model2 = "" -- model file  for lang 2
config.model3 = "" -- model file  for lang 3
config.model4 = "" -- model file  for lang 4
config.vocab1 = "" -- vocab file  for lang 1
config.vocab2 = "" -- vocab file  for lang 2
config.vocab3 = "" -- vocab file  for lang 3
config.vocab4 = "" -- vocab file  for lang 4
config.lookuptable1 = "" -- lookup table file  for lang 1
config.lookuptable2 = "" -- lookup table file  for lang 2
config.lookuptable3 = "" -- lookup table file  for lang 3
config.lookuptable4 = "" -- lookup table file  for lang 4
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

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus1", config.corpus1)
cmd:option("-corpus2", config.corpus2)
cmd:option("-corpus3", config.corpus3)
cmd:option("-corpus4", config.corpus4)
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
params = cmd:parse(arg)


---corpus1 english.1000.tok -corpus2 turkish.1000.tok -corpus3 english.1000.tok -corpus4 turkish.1000.tok -lr 0.1 -lr_decay 1 -max_epoch 300 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1


for param, value in pairs(params) do
    config[param] = value
end

nm = string.sub(config.corpus1,1,2).."-"..string.sub(config.corpus2,1,2).."-"..string.sub(config.corpus3,1,2).."-"..string.sub(config.corpus4,1,2)
config.model1 = config.corpus1.."."..nm..".model"-- model file  for lang 1
config.model2 = config.corpus2.."."..nm..".model" -- model file  for lang 2
config.model3 = config.corpus3.."."..nm..".model" -- model file  for lang 3
config.model4 = config.corpus4.."."..nm..".model" -- model file  for lang 4
config.vocab1 = config.corpus1.."."..nm..".vocab" -- vocab file  for lang 1
config.vocab2 = config.corpus2.."."..nm..".vocab" -- vocab file  for lang 2
config.vocab3 = config.corpus3.."."..nm..".vocab" -- vocab file  for lang 3
config.vocab4 = config.corpus4.."."..nm..".vocab" -- vocab file  for lang 4
config.lookuptable1 = config.corpus1.."."..nm..".LT" -- lookup table file  for lang 1
config.lookuptable2 = config.corpus2.."."..nm..".LT" -- lookup table file  for lang 2
config.lookuptable3 = config.corpus3.."."..nm..".LT" -- lookup table file  for lang 3
config.lookuptable4 = config.corpus4.."."..nm..".LT" -- lookup table file  for lang 4

corpus_name_1 = corpus_path..config.corpus1.."."..nm..".corpus.tch"
corpus_name_2 = corpus_path..config.corpus2.."."..nm..".corpus.tch"
corpus_name_3 = corpus_path..config.corpus3.."."..nm..".corpus.tch"
corpus_name_4 = corpus_path..config.corpus4.."."..nm..".corpus.tch"


inputs_name_1 = corpus_path..config.corpus1.."."..nm..".inputs.tch"
inputs_name_2 = corpus_path..config.corpus2.."."..nm..".inputs.tch"
inputs_name_3 = corpus_path..config.corpus3.."."..nm..".inputs.tch"
inputs_name_4 = corpus_path..config.corpus4.."."..nm..".inputs.tch"


for i,j in pairs(config) do
    print(i..": "..j)
end

if config.corpus_load == 0 then
  print("corpus data will be prepared\n")
  corpus_1 = Corpus(config.corpus1,data_path)
  corpus_2 = Corpus(config.corpus2,data_path)
  corpus_3 = Corpus(config.corpus4,data_path)
  corpus_4 = Corpus(config.corpus3,data_path)
  
  corpus_1:prepare_corpus()
  corpus_2:prepare_corpus()
  corpus_3:prepare_corpus()
  corpus_4:prepare_corpus()
  
  --no_of_sents = #corpus_en.sequences
  inputs_1=corpus_1:get_data():cuda()
  inputs_2=corpus_2:get_data():cuda()
  inputs_3=corpus_3:get_data():cuda()
  inputs_4=corpus_4:get_data():cuda()

  torch.save(config.vocab1,corpus_1.vocab_map)
  torch.save(config.vocab2,corpus_2.vocab_map)
  torch.save(config.vocab3,corpus_3.vocab_map)
  torch.save(config.vocab4,corpus_4.vocab_map)
  
  torch.save(corpus_name_1,corpus_1)
  torch.save(corpus_name_2,corpus_2)
  torch.save(corpus_name_3,corpus_3)
  torch.save(corpus_name_4,corpus_4)
  
  torch.save(inputs_name_1,inputs_1)
  torch.save(inputs_name_2,inputs_2)
  torch.save(inputs_name_3,inputs_3)
  torch.save(inputs_name_4,inputs_4)
  print("corpus data prepared and saved\n")
else 
  print("corpus data will be loaded\n")
  corpus_1 = torch.load(corpus_name_1)
  corpus_2 = torch.load(corpus_name_2)
  corpus_3 = torch.load(corpus_name_3)
  corpus_4 = torch.load(corpus_name_4)
  inputs_1 = torch.load(inputs_name_1)
  inputs_2 = torch.load(inputs_name_2)
  inputs_3 = torch.load(inputs_name_3)
  inputs_4 = torch.load(inputs_name_4)
  print("corpus data loaded\n")
end

--os.exit()
print(corpus_name_1.." "..corpus_1.no_of_words.." words, "..#corpus_1.sequences.." sents, "..corpus_1.longest_seq.." sent max lenght\n")
print(corpus_name_2.." "..corpus_2.no_of_words.." words, "..#corpus_2.sequences.." sents, "..corpus_2.longest_seq.." sent max lenght\n")
print(corpus_name_3.." "..corpus_3.no_of_words.." words, "..#corpus_3.sequences.." sents, "..corpus_3.longest_seq.." sent max lenght\n")
print(corpus_name_4.." "..corpus_4.no_of_words.." words, "..#corpus_4.sequences.." sents, "..corpus_4.longest_seq.." sent max lenght\n")
print(inputs_name_1..inputs_name_2.."\n")
print(inputs_name_3..inputs_name_4.."\n")

--torch.save(config.vocab1,corpus_1.vocab_map)
--torch.save(config.vocab2,corpus_2.vocab_map)
biLnModel = BiLangModelAdditiveMulti(corpus_1.no_of_words,corpus_2.no_of_words,corpus_3.no_of_words,corpus_4.no_of_words,config)

header = "additive-multi-tok"

if config.model_load == 1 then
  config.model1 = config.corpus1.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".model"-- model file  for lang 1
  config.model2 = config.corpus2.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".model" -- model file  for lang 2
  config.model3 = config.corpus3.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".model"-- model file  for lang 3
  config.model4 = config.corpus4.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".model" -- model file  for lang 4
  
  config.lookuptable1 = config.corpus1.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".LT" -- lookup table file  for lang 1
  config.lookuptable2 = config.corpus2.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".LT" -- lookup table file  for lang 2
  config.lookuptable3 = config.corpus3.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".LT" -- lookup table file  for lang 3
  config.lookuptable4 = config.corpus4.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".LT" -- lookup table file  for lang 4
  
  print("model data will be loaded\n")
  print(config.model1.." "..config.model2.."\n")
  print(config.model3.." "..config.model4.."\n")
  print(config.lookuptable1.." "..config.lookuptable2.."\n")
  print(config.lookuptable3.." "..config.lookuptable4.."\n")
  biLnModel.legPri = torch.load(model_load_path..config.model1)
  biLnModel.legSec = torch.load(model_load_path..config.model2)
  biLnModel.legThrd = torch.load(model_load_path..config.model3)
  biLnModel.legFrth = torch.load(model_load_path..config.model4)
  biLnModel.ltPri = torch.load(model_load_path..config.lookuptable1)
  biLnModel.ltSec = torch.load(model_load_path..config.lookuptable2)
  biLnModel.ltThrd = torch.load(model_load_path..config.lookuptable3)
  biLnModel.ltFrth = torch.load(model_load_path..config.lookuptable4)
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
  sc1,sc2,sc3,sc4,num_of_samples,avg1,avg2,avg3,avg4 = cv_test(cv_max,biLnModel,config.prev_epoch,inputs_1,inputs_2,inputs_3,inputs_4,test_size)
  score_logger:add{['lr'] = config.lr, ['score1 '] = avg1, ['score2 '] = avg2, ['score3 '] = avg3, ['score4 '] = avg4 }
  score_logger:style{['lr'] = '+-',['score1 '] = '+-',['score2 '] = '+-',['score3 '] = '+-',['score4 '] = '+-'}
  score_logger:plot()  
end

train_model(biLnModel,corpus_1,corpus_2,corpus_3,corpus_4,inputs_1,inputs_2,inputs_3,inputs_4,config)
sc1,sc2,sc3,sc4,num_of_samples,avg1,avg2,avg3,avg4 = cv_test(cv_max,biLnModel,config.max_epoch,inputs_1,inputs_2,inputs_3,inputs_4,test_size)
score_logger:add{['lr'] = config.lr, ['score1 '] = avg1, ['score2 '] = avg2, ['score3 '] = avg3, ['score4 '] = avg4 }
score_logger:style{['lr'] = '+-',['score1 '] = '+-',['score2 '] = '+-',['score3 '] = '+-',['score4 '] = '+-'}
score_logger:plot()  
loss_logger:plot()  

model_save(config,config.max_epoch,biLnModel)

print("program ended")
