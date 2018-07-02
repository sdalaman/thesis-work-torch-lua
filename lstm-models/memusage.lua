require "cutorch"
--cutorch.setDevice(3) 

a = torch.Tensor(1024,1024):uniform() 
initFree = {} 

print('## Initial memory ##') 
for i=1,cutorch.getDeviceCount() do 
    initFree[i], _ = cutorch.getMemoryUsage(i) 
    print(i, initFree[i]) 
end 

a = a:cuda() -- Send to gpu

print() 
print('## Input on GPU ##') 
for i=1,cutorch.getDeviceCount() do 
    after, _ = cutorch.getMemoryUsage(i) 
    print(i, initFree[i] - after) 
end