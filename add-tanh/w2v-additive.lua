require 'data-additive'
require 'model-additive'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'

require 'io'
require 'os'
require 'paths'

path = "/home/saban/work/additive/"
data_path = path.."data/"
model_save_path = path.."model-impl/add-tanh/model-files/"
model_load_path = path.."model-impl/add-tanh/model-files/"
corpus_path = path.."models/corpus/"

test_size = 100
cv_max = 10

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end

function test_model(model,final,epoch,inputs_pri,inputs_sec,test_size)
  inPri = torch.zeros(test_size,inputs_pri:size()[2]):cuda()
  inSec = torch.zeros(test_size,inputs_sec:size()[2]):cuda()
  cl = inputs_pri:size()[1]
  math.randomseed( os.time() )

  for i=1,test_size do
    ln = math.random(1,cl)
    inPri[i] = inputs_pri[ln]
    inSec[i] = inputs_sec[ln]
  end

  --local inputPri = inputs_pri[{{1,1000},{}}]
  --local inputSec = inputs_sec[{{1,1000},{}}]
  local outputPri = model:getLegPri():forward( inPri)
  local outputSec = model:getLegSec():forward( inSec)
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
    end
  end
  print("Epoch "..epoch.."  Test Score: " .. score.. '/' .. all_rootsPri:size(1))
  print("\n")
  return score,all_rootsPri:size(1)
end

function cv_test(cv,model,maxepoch,insPri,insSec,test_size)
  tot = 0
  for i = 1,cv do
    sc,nsum = test_model(model,true,maxepoch,insPri,insSec,test_size)
    tot = tot + sc
  end
  print("-------"..cv.." Epochs avg: "..tot/cv.."\n")
  return sc,nsum,tot/cv
end

function train_model(model,corpus_pri,corpus_sec,inputs_pri,inputs_sec,config)
--  no_of_sents = #corpus_pri.sequences
  print ( 'Training started \n')
  beginning_time = torch.tic()
  --split = nn.SplitTable(2)
  additivePri = model:getLegPri()
  additiveSec = model:getLegSec()
  criterion = model:getCriterion()
 
  k=0
  no_of_sents = math.floor(#corpus_pri.sequences/config.batch_size)*config.batch_size
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
        local err = criterion:forward( outputSec, outputPri)
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
    loss_logger:add{['lr'] = config.lr,['training error mean'] = torch.mean( torch.Tensor( errors))}
    loss_logger:style{['lr'] = '+-',['training error mean'] = '+-'}

    if i % config.dump_freq == 0 then 
      sc,num_of_samples,avg = cv_test(cv_max,model,i,inputs_pri,inputs_sec,test_size)
      score_logger:add{['lr'] = config.lr, ['score '] = avg }
      score_logger:style{['lr'] = '+-',['score '] = '+-'}
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
  nm = string.sub(cfg.corpus1,1,2).."-"..string.sub(cfg.corpus2,1,2)
  model1nm = model_save_path..cfg.corpus1.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".model"-- model file  for lang 1
  model2nm = model_save_path..cfg.corpus2.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".model" -- model file  for lang 2
  lookuptable1nm = model_save_path..cfg.corpus1.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".LT" -- lookup table file  for lang 1
  lookuptable2nm = model_save_path..cfg.corpus2.."."..nm.."."..cfg.lr.."."..epoch.."."..cfg.emb_size.."."..header..".LT" -- lookup table file  for lang 2
  model:getLegSec():clearState()
  torch.save(model2nm,model:getLegSec())
  torch.save(lookuptable2nm,model:getLookupTableSec())
  model:getLegPri():clearState()
  torch.save(model1nm,model:getLegPri())
  torch.save(lookuptable1nm,model:getLookupTablePri())
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
config.corpus2 = "" -- input data  for lang 
config.model1 = "" -- model file  for lang 1
config.model2 = "" -- model file  for lang 2
config.vocab1 = "" -- vocab file  for lang 1
config.vocab2 = "" -- vocab file  for lang 2
config.lookuptable1 = "" -- lookup table file  for lang 1
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
params = cmd:parse(arg)


---corpus1 english.1000.tok -corpus2 turkish.1000.tok -lr 0.1 -lr_decay 1 -max_epoch 300 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1

---corpus1 english.10000.tok -corpus2 turkish.10000.tok -lr 0.1 -lr_decay 1 -max_epoch 200 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1

--corpus1 english.1000.tok.morph -corpus2 turkish.1000.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 1000 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1

--corpus1 english.10000.tok.morph -corpus2 turkish.10000.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 1000 -emb_size 64 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1

---corpus1 english.all.tok -corpus2 turkish.all.tok -lr 0.1 -lr_decay 1 -max_epoch 350 -emb_size 64 -threshold 50 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1

--corpus1 english.all.tok.morph -corpus2 turkish.all.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 3000 -emb_size 256 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -prev_epoch 0

--corpus1 english.all.tok.morph -corpus2 turkish.all.tok.morph -lr 0.01 -lr_decay 1 -max_epoch 5000 -emb_size 256 -threshold 100 -batch_size 100 -dump_freq 50 -model_load 0 -corpus_load 1 -prev_epoch 3000

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
  corpus_1 = Corpus(config.corpus1,data_path)
  corpus_2 = Corpus(config.corpus2,data_path)
  corpus_1:prepare_corpus()
  corpus_2:prepare_corpus()
  --no_of_sents = #corpus_en.sequences
  inputs_1=corpus_1:get_data():cuda()
  inputs_2=corpus_2:get_data():cuda()

  torch.save(config.vocab1,corpus_1.vocab_map)
  torch.save(config.vocab2,corpus_2.vocab_map)
  torch.save(corpus_name_1,corpus_1)
  torch.save(corpus_name_2,corpus_2)
  torch.save(inputs_name_1,inputs_1)
  torch.save(inputs_name_2,inputs_2)
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
biLnModel = BiLangModelAdditive(corpus_1.no_of_words,corpus_2.no_of_words,config)

header = "additive-morph"

if config.model_load == 1 then
  config.model1 = config.corpus1.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".model"-- model file  for lang 1
  config.model2 = config.corpus2.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".model" -- model file  for lang 2
  config.lookuptable1 = config.corpus1.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".LT" -- lookup table file  for lang 1
  config.lookuptable2 = config.corpus2.."."..nm.."."..config.lr_prev.."."..config.prev_epoch.."."..config.emb_size.."."..header..".LT" -- lookup table file  for lang 2
  print("model data will be loaded\n")
  print(config.model1.." "..config.model2.."\n")
  print(config.lookuptable1.." "..config.lookuptable2.."\n")
  biLnModel.legPri = torch.load(model_load_path..config.model1)
  biLnModel.legSec = torch.load(model_load_path..config.model2)
  biLnModel.ltPri = torch.load(model_load_path..config.lookuptable1)
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
  sc,num_of_samples,avg = cv_test(cv_max,biLnModel,config.prev_epoch,inputs_1,inputs_2,test_size)
  score_logger:add{['lr'] = config.lr,['score '] = avg }
  score_logger:style{['lr'] = '+-',['score '] = '+-'}
  score_logger:plot()  
end

train_model(biLnModel,corpus_1,corpus_2,inputs_1,inputs_2,config)
sc,num_of_samples,avg = cv_test(cv_max,biLnModel,config.max_epoch,inputs_1,inputs_2,test_size)
score_logger:add{['lr'] = config.lr,['score '] = avg }
score_logger:style{['lr'] = '+-',['score '] = '+-'}
score_logger:plot()  
loss_logger:plot()  

model_save(config,config.max_epoch,biLnModel)

print("program ended")
