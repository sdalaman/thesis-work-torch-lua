-- How to copy a Torch model to a new type without explicit cloning.

-- Based on code in checkpoints.lua in http://github.com/facebook/fb.resnet.torch

require 'torch'

function copy_convert(obj, t)
   local copy = {}
   for k, v in pairs(obj) do
      if type(v) == 'table' then
         copy[k] = copy_convert(v, t)
      elseif torch.isTensor(v) then
         copy[k] = v:type(t)
      elseif k == '_type' then
         copy[k] = t
      else
         copy[k] = v
      end
   end
   if torch.typename(obj) then
      torch.setmetatable(copy, torch.typename(obj))
   end
   return copy
end

require 'cunn'

m_gpu = nn.Sequential()
m_gpu:add(nn.VolumetricConvolution(1, 4, 3, 3, 3, 1, 1, 1))
m_gpu:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
m_gpu:cuda()