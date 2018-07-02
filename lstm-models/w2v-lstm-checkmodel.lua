require 'data-lstm-drop'
require 'model-lstm-drop'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'
require 'io'
require 'os'
require 'paths'

cutorch.setHeapTracking(true)
collectgarbage()

mdlLayout4Str = "LSTM SC AVG"
fname4 = "/home/saban/work/additive/model-impl/lstm-models/model-files/10000/4-ScAvg-tok-morph/en-tu_tu_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000-morph.model"

mdlLayout5Str = "BILSTM AVG"
fname5 = "/home/saban/work/additive/model-impl/lstm-models/model-files/10000/5-Bilstm-Avg-tok-morph/en-tu_tu_1e-06_2500.lstm.128.512.60.BiLstm-Avg-drop-10000-morph.model"

mdlLayout6Str = "LSTM SC AVG"
fname6 = "/home/saban/work/additive/model-impl/lstm-models/model-files/10000/6-ScAvg-tok-morph/en-tu_en_0.01_100.lstm.128.128.60.lstm-ScAvg-drop-10000-morph.model"

mdlLayout7Str = "BILSTM SC AVG"
fname7 = "/home/saban/work/additive/model-impl/lstm-models/model-files/10000/7-Bilstm_Sc-Avg-tok/en-tu_en_0.001_450.lstm.128.256.60.BiLstmSc-Avg-drop-10000.model"

function printModelLayout(fname,mdlLayoutStr)
  print("-----------------------------------------------")
  print("---- file name "..fname)
  model = torch.load(fname)
  print("---- model layout "..mdlLayoutStr)
  print("   ")
  print(model)
  print("-----------------------------------------------")
end

printModelLayout(fname6,mdlLayout6Str)
printModelLayout(fname7,mdlLayout7Str)

print("end")
