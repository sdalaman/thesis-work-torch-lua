require 'data'
require 'nn'
require 'cunn'
require 'rnn'
require("io")
require("os")
require("paths")

local stringx = require('pl.stringx')

function sortPairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

function cossimilarity(vec1,vec2)
  dot = 0.0
  length1 = 0.0
  length2 = 0.0
  v1len = table.getn(vec1) - 2
  v2len = table.getn(vec2) - 2
  for indx = 2 ,65 ,1
  do
    if(v1len >= indx) then
      vl1 = tonumber(vec1[indx])
    else
      vl1= 0.0
    end
    if(v2len >= indx) then
      vl2 = tonumber(vec2[indx])
    else
      vl2=0.0
    end 
    dot = dot +  vl1*vl2
    length1 = length1 +  vl1*vl1
    length2 = length2 +  vl2*vl2
  end
  return (dot / (math.sqrt(length1)*math.sqrt(length2)))
end

function wordVectors(fname,vocab,weight)
  output_file = io.open(fname, "w")
  i=0
  for word,indx in pairs(vocab) 
  do 
    if indx ~= 0 then
      vec = stringx.split(tostring(weight[indx]), '\n')
      output_file:write(string.format("%s ; %d \n %s \n\n",word,indx,table.concat(vec, ";" , 2 , 65)) )
    end
  end
  io.close(output_file)
end


function wordVectorsSim(fname,vocab,weight)
  output_file = io.open(fname, "w")
  cosSim = {}
  i=0
  for word1,indx1 in pairs(vocab) 
  do 
    if indx1 ~= 0 then
      rw1 = {}
      for word2,indx2 in pairs(vocab) 
      do
        if indx2 ~= 0 and word1 ~= word2 then
          vec1 = stringx.split(string.gsub(tostring(weight[indx1]),",","."), '\n')
          vec2 = stringx.split(string.gsub(tostring(weight[indx2]),",","."), '\n')
          if(cosSim[word1] == nil and cosSim[word2] == nil) then
            vl = cossimilarity(vec1,vec2)
            rw1[word2] = vl
            --cosSim[word1][word2] = vl
            output_file:write(string.format("%s ; %s ; %f \n",word1,word2,vl))
            i=i+1
            print(string.format("%i %s %s",i,word1,word2))
          end
        end
      end
      cosSim[word1] = rw1
    end
  end
  io.close(output_file)
end

function listWords(fname,vocab)
  output_file = io.open(fname, "w")
  for word1,indx1 in pairs(vocab) 
  do 
    output_file:write(string.format("%s %d \n",word1,indx1))
  end
  io.close(output_file)
end


function normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
    	m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function get_sim_words(word_vecs_norm,vocab,index2word,w, k)
    if type(w) == "string" then
        if vocab[w] == nil then
          print("'"..w.."' does not exist in vocabulary.")
          return nil
        else
            w = word_vecs_norm[vocab[w]]
        end
    end
    local sim = torch.mv(word_vecs_norm, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {index2word[idx[i]], -sim[i]}
    end
    return r
end

function print_sim_words(words, k)
  for i = 1, #words do
    	r = get_sim_words(words[i], k)
      if r ~= nil then
   	    print("-------"..words[i].."-------")
        for j = 1, k do
	        print(string.format("%s, %.4f", r[j][1], r[j][2]))
        end
      end
  end
end

-- -model english_LT -vocab english_vocab -listfilename englishwordvectors.txt
-- Parse input arguments
config = {}
config.model = "" -- model file name
config.vocab = "" -- vocab file name
config.listfilename = "" -- list output filen name

cmd = torch.CmdLine()
cmd:option("-model", config.model)
cmd:option("-vocab", config.vocab)
cmd:option("-listfilename", config.listfilename)
params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end

for i,j in pairs(config) do
    print(i..": "..j)
end

weight_w2v = torch.load(config.model):double()
vocab = torch.load(config.vocab)
print(weight_w2v.weight:size())
--print(vocab)

index2words = {}
numberOfWords = 0
for k,v in pairs(vocab) 
do 
  numberOfWords = numberOfWords + 1 
  index2words[v] = k
end
print("Number of words "..numberOfWords)  

weight_w2v_norm = normalize(weight_w2v.weight)

wordVectors(config.listfilename,vocab,weight_w2v_norm)

--exm= {"males","your","Democracy"}

--wls = get_sim_words(weight_w2v_norm,vocab,index2words,"males",5)

--listWords("sd-englishwordlist.txt",english_vocab)
--listWords("sd-turkishwordlist.txt",turkish_vocab)

--[[
if pcall(wordVectorsSim,"sd-englishwordvectorssorted.txt",english_vocab,english_w2v.weight) then
  -- no errors while running `foo'
else
  -- `foo' raised an error: take appropriate actions
  print("error")
end
]]    
--wordVectorsSim("sd-englishwordvectorssorted.txt",english_vocab,english_w2v.weight)

--for k,v in sortPairs(cosSim) do
--    sortedKey[k] = v
--end

-- this uses an custom sorting function ordering by score descending
--for k,v in sortPairs(cosSim, function(t,a,b) return t[b] < t[a] end) do
--    sortedValue[k] = v
--end
--for k,v in pairs(wls) do
--  print(v[1].."-"..v[2])
--end
--i = i + 1


