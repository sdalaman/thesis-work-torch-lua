require 'model-lstm-drop'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'
require 'io'
require 'os'
require 'paths'

function printf(s,...)
  return io.write(s:format(...))
end

scenarios= {}

scenarios[#scenarios+1] = { lr = 0.0001;prev_lr = 0.0001;alpha = 0.95;lr_decay = 0;weight_decay = 0; lr_list = nil ;emb_size = 128;hidden_size = 512;num_of_hidden=1;out_size = 128;gate_type = 'LSTM';short_cut=0;momentum=0.5; threshold = 50 ;dump_frq = 50;max_epoch = 5000;epoch=1000;batch_size = 50;init = 1;max_seq_len = 60;win_size = 1;opm_name = 'rmsprop';opm = optim.rmsprop; train_score = 0;test_score = 0;num_of_samples = 0;testcorpus1 = "english.10000.tok";testcorpus2 = "turkish.10000.tok";test_size = 100;err_test_size = 1000;out_file = "test-lstm-10000.txt"; dsize ="1000";max_cv = 10;coefL1=0;coefL2=0;fhead=""}

vocab_size1 = 100
vocab_size2 = 100
seq_len1 = 60
seq_len2 = 60
prm_list = scenarios[1]

print("####################################################################")
print("----BiLangModelLSTM----")
modelLSTM = BiLangModelLSTM(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
print (modelLSTM.mdlPri)
print("####################################################################")
print("----BiLangModelLSTMAvg----")
modelLSTM = BiLangModelLSTMAvg(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
print (modelLSTM.mdlPri)
print("####################################################################")
print("----BiLangModelLSTMScAvg----")
modelLSTM = BiLangModelLSTMScAvg(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
print (modelLSTM.mdlPri)
print("####################################################################")
print("----BiLangModelLSTM2LyScAvg----")
modelLSTM = BiLangModelLSTM2LyScAvg(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
print (modelLSTM.mdlPri)
print("####################################################################")
print("----BiLangModelLSTM3LyScAvg----")
modelLSTM = BiLangModelLSTM3LyScAvg(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
print (modelLSTM.mdlPri)
print("----BiLangModelBiLSTMAvg2----")
print("####################################################################")
modelLSTM = BiLangModelBiLSTMAvg2(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
print (modelLSTM.biSeqPri)
print("####################################################################")
print("----BiLangModelBiLSTMScAvg2----")
modelLSTM = BiLangModelBiLSTMScAvg2(vocab_size1,vocab_size2,seq_len1,seq_len2,prm_list)  
print (modelLSTM.biSeqPri)
print("####################################################################")

print ("end")


