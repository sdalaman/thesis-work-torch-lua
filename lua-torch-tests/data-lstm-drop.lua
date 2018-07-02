local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'


local Corpus = torch.class("Corpus")

function Corpus:__init(fname,d_path)
  self.vocab_idx = 0
  self.vocab_map = {['<pad>'] = 0}
  self.longest_seq = 0
  self.data_file = d_path..fname
  self.x = nil
  self.no_of_words = 0
  self.no_of_sents = 0
  self. sep = '\t'
  self.f = nil
  self.b = nil
end

function Corpus:padding(sequence)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , self.longest_seq - #sequence do
    new_sequence[i] = '<pad>'
  end
  j = 1
  for i = (self.longest_seq - math.min(self.longest_seq,#sequence))+1, self.longest_seq do
    new_sequence[i] = sequence[j]
    j = j + 1
  end
  return new_sequence
end

function Corpus:paddingLSTM(sequence)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , math.min(self.longest_seq,#sequence) do
    new_sequence[i] = sequence[i]
  end
  for i = math.min(self.longest_seq,#sequence)+1, self.longest_seq do
    new_sequence[i] = '<pad>'
  end
  return new_sequence
end


function Corpus:load_data(fname,max_len)
   local data = file.read(fname)
   -- data = stringx.replace(data, '\n', '<eos>')
   self.sequences = stringx.split(data, '\n')
   --all_words = stringx.split(data,self.sep) -- tokenized with tab
   print(string.format("Loading %s, Number of sentences = %d", fname, #self.sequences))
   self.no_of_sents = #self.sequences
   for i = 1 , #self.sequences do
     local words = stringx.split(self.sequences[i],self.sep) -- tokenized with tab
     words_len = #words
     if self.longest_seq < words_len then 
       self.longest_seq = words_len
     end
   end
   print(string.format("Longest sequence = %d", self.longest_seq))
   if max_len ~= nil then
     self.longest_seq = math.min(self.longest_seq,max_len)
     print(string.format("Reduced down to = %d", self.longest_seq))
   end
   
end

function Corpus:shape_data()
  local x = torch.zeros(#self.sequences, self.longest_seq)
  local totseq=0
   for i = 1, #self.sequences do
     local words = stringx.split(self.sequences[i], self.sep) -- tokenized with tab
     totseq=totseq+#words
     words = self:padding(words)
     for j = 1, #words do 
       if self.vocab_map[words[j]] == nil then
         self.vocab_idx = self.vocab_idx + 1
         self.vocab_map[words[j]] = self.vocab_idx
       end
       x[i][j] = self.vocab_map[words[j]] 
     end

   end
   print(string.format("Number of words = %d , mean sntc len %4.2f", self.vocab_idx,totseq/#self.sequences))
   self.no_of_words = self.vocab_idx
   return x
end

function Corpus:shape_dataLSTM()
  local x = torch.zeros(#self.sequences, self.longest_seq)
  local totseq=0
   for i = 1, #self.sequences do
     local words = stringx.split(self.sequences[i], self.sep) -- tokenized with tab
     totseq=totseq+#words
     words = self:paddingLSTM(words)
     for j = 1, #words do 
       if self.vocab_map[words[j]] == nil then
         self.vocab_idx = self.vocab_idx + 1
         self.vocab_map[words[j]] = self.vocab_idx
       end
       x[i][j] = self.vocab_map[words[j]] 
     end

   end
   print(string.format("Number of words = %d , mean sntc len %4.2f", self.vocab_idx,totseq/#self.sequences))
   self.no_of_words = self.vocab_idx
   return x
end


function Corpus:prepare_corpus(max_len,max_sent)
   self:load_data(self.data_file,max_len,max_sent)
end

function Corpus:get_data()
   self.x = self:shape_data()
   return self.x
end

function Corpus:get_dataLSTM()
   self.x = self:shape_dataLSTM()
   return self.x
end

function testcorpus_shapedataLSTM(corpus,test_corpus)
  local x = torch.zeros(#test_corpus.sequences, test_corpus.longest_seq)
   for i = 1, #test_corpus.sequences do
     local words = stringx.split(test_corpus.sequences[i], test_corpus.sep) -- tokenized with tab
     words = test_corpus:paddingLSTM(words)
     for j = 1, #words do 
       if corpus.vocab_map[words[j]] == nil then
         x[i][j] = 0 
      else
         x[i][j] = corpus.vocab_map[words[j]] 
      end
     end

   end
   return x
end


--------------------------------------

function Corpus:shape_dataBiLSTM()
  local fwd = torch.zeros(#self.sequences, self.longest_seq)
  local bck = torch.zeros(#self.sequences, self.longest_seq)
  local totseq=0
   for i = 1, #self.sequences do
     local wordsFrw = stringx.split(self.sequences[i], self.sep) -- tokenized with tab
     wordsBck = {}
     for i = #wordsFrw,1,-1 do
        table.insert(wordsBck,wordsFrw[i])
     end
     --wordsBck = wordsFwd:index(1 ,torch.linspace(wordsFwd:size(1),1,wordsFwd:size(1)):long())
     totseq=totseq + #wordsFrw
     wordsFrw = self:paddingLSTM(wordsFrw)
     wordsBck = self:paddingLSTM(wordsBck)
     for j = 1, #wordsFrw do 
       if self.vocab_map[wordsFrw[j]] == nil then
         self.vocab_idx = self.vocab_idx + 1
         self.vocab_map[wordsFrw[j]] = self.vocab_idx
       end
       fwd[i][j] = self.vocab_map[wordsFrw[j]] 
     end
     for j = 1, #wordsBck do 
       if self.vocab_map[wordsBck[j]] == nil then
         self.vocab_idx = self.vocab_idx + 1
         self.vocab_map[wordsBck[j]] = self.vocab_idx
       end
       bck[i][j] = self.vocab_map[wordsBck[j]] 
     end
   end
   print(string.format("Number of words = %d , mean sntc len %4.2f", self.vocab_idx,totseq/#self.sequences))
   self.no_of_words = self.vocab_idx
   return fwd,bck
end


function Corpus:get_dataBiLSTM()
   self.f,self.b = self:shape_dataBiLSTM()
   return self.f,self.b
end

function strcmp(str1,str2)
  local l1 = str1:len()
  local l2 = str2:len()
  cnt = 0
  for i=1,l1 do
    if i > l2 then
        break
    end
    if str1:sub(i,i) == str2:sub(i,i) then
      cnt = cnt + 1
    else
      break
    end
  end
  return cnt,cnt/l1
end

function findSimilarWord(vocabMap,word)
  score = 0
  indexSel = 0
  valueSel = ""
  for index,value in pairs(vocabMap) do
    cnt,sc = strcmp(index,word)
    if sc > score then
      score = sc
      indexSel = index
      valueSel = value
    end
  end
  return score,indexSel,valueSel
end

function testcorpus_shapedataBiLSTM(corpus,test_corpus)
  local fwd = torch.zeros(#test_corpus.sequences, test_corpus.longest_seq)
  local bck = torch.zeros(#test_corpus.sequences, test_corpus.longest_seq)
   for i = 1, #test_corpus.sequences do
     local wordsFrw = stringx.split(test_corpus.sequences[i], test_corpus.sep) -- tokenized with tab
     wordsBck = {}
     for i = #wordsFrw,1,-1 do
        table.insert(wordsBck,wordsFrw[i])
     end
     wordsFrw = test_corpus:paddingLSTM(wordsFrw)
     wordsBck = test_corpus:paddingLSTM(wordsBck)
     for j = 1, #wordsFrw do 
       if corpus.vocab_map[wordsFrw[j]] == nil then
         fwd[i][j] = 0
      else
         fwd[i][j] = corpus.vocab_map[wordsFrw[j]] 
      end
     end
     for j = 1, #wordsBck do 
       if corpus.vocab_map[wordsBck[j]] == nil then
         bck[i][j] = 0
      else
         bck[i][j] = corpus.vocab_map[wordsBck[j]] 
      end
     end
   end
   return fwd,bck
end

function testcorpus_shapedataBiLSTMNew(corpus,test_corpus)
  local fwd = torch.zeros(#test_corpus.sequences, test_corpus.longest_seq)
  local bck = torch.zeros(#test_corpus.sequences, test_corpus.longest_seq)
   for i = 1, #test_corpus.sequences do
     local wordsFrw = stringx.split(test_corpus.sequences[i], test_corpus.sep) -- tokenized with tab
     wordsBck = {}
     for i = #wordsFrw,1,-1 do
        table.insert(wordsBck,wordsFrw[i])
     end
     wordsFrw = test_corpus:paddingLSTM(wordsFrw)
     wordsBck = test_corpus:paddingLSTM(wordsBck)
     for j = 1, #wordsFrw do 
       if corpus.vocab_map[wordsFrw[j]] == nil then
         sc,ind,val = findSimilarWord(corpus.vocab_map,wordsFrw[j])
         fwd[i][j] = val
      else
         fwd[i][j] = corpus.vocab_map[wordsFrw[j]] 
      end
     end
     for j = 1, #wordsBck do 
       if corpus.vocab_map[wordsBck[j]] == nil then
         sc,ind,val = findSimilarWord(corpus.vocab_map,wordsBck[j])
         bck[i][j] = val 
      else
         bck[i][j] = corpus.vocab_map[wordsBck[j]] 
      end
     end
   end
   return fwd,bck
end

function dumpCorpus(corpus,fname)
  output_file = io.open(fname, "w")
  for index,value in pairs(corpus.vocab_map) do
    output_file:write(index.." "..value.."\n")
  end
  io.close(output_file)
end
