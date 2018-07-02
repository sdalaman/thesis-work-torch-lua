require 'torch'

local function pca(X)
  -- PCA -------------------------------------------------------------------------
  -- X is m x n
  local mean = torch.mean(X, 1) -- 1 x n
  local m = X:size(1)
  local Xm = X - torch.ones(m, 1) * mean
  Xm:div(math.sqrt(m - 1))
  local v,s,_ = torch.svd(Xm:t())
  s:cmul(s) -- n
   
  return mean, v, s -- v:= eigenVectors, s:=eigenValues
end


x = torch.randn(5,10)
mn,v,s = pca(x)
print(1)