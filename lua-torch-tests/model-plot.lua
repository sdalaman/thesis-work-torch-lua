require('gnuplot')
require('torch')
require('nn')
require('rnn')
require('cunn')
 
 
 -- Build Model
lm = nn.Sequential()
lstm = nn.Sequencer(nn.LSTM(2, 30)) 
lm:add(lstm)
lm:add(nn.Linear(30, 1))
 
w,g = lm:parameters()
 
modelFile = 'en-tu_en_0.0001_500.lstm.512.10000.model'
model = torch.load(modelFile)
w,g = model:parameters()
fw,fg = model:getParameters()
lstm = model.modules[3].modules[3].modules[1].modules[1].modules[1]
lw,lg = lstm:parameters()
gnuplot.pngfigure('linear1.png')
gnuplot.imagesc(model.modules[1].weight, 'color')
gnuplot.plotflush()
  
gnuplot.pngfigure('linear3.png')
gnuplot.imagesc(model.modules[3].weight, 'color')
gnuplot.plotflush()

