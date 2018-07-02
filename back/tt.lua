require 'torch'

function prepare_datasets(data,train_pct)
  local data_len = data:size(1)
  local idx_prm = torch.randperm(data_len)
  local train_size = math.floor(idx_prm:size(1)*train_pct)
  local train = torch.Tensor(train_size,data:size(2))
  local test = torch.Tensor((data_len-train_size),data:size(2))
  for idx = 1,e1 do
    train[idx] = data[idx_prm[idx]]
  end
  for idx = 1,(data_len-train_size) do
    test[idx] = data[idx_prm[train_size+idx]]
  end
  return train,test 
end
data = torch.Tensor({{1,2,2},{3,4,4},{5,6,6},{7,8,7},{9,10,13},{11,12,22},{13,14,33}})
tr,ts = prepare_datasets(data,0.75)
print(xx)



