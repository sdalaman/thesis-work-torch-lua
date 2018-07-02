mlp = nn.Sequential();
c = nn.Parallel(1,2)     -- Parallel container will associate a module to each slice of dimension 1
                         -- (row space), and concatenate the outputs over the 2nd dimension.           

for i=1,10 do            -- Add 10 Linear+Reshape modules in parallel (input = 3, output = 2x1)
 local t=nn.Sequential()
 t:add(nn.Linear(3,2))   -- Linear module (input = 3, output = 2)
 t:add(nn.Reshape(2,1))  -- Reshape 1D Tensor of size 2 to 2D Tensor of size 2x1
 c:add(t)
end

mlp:add(c)               -- Add the Parallel container in the Sequential container

pred = mlp:forward(torch.randn(10,3)) -- 2D Tensor of size 10x3 goes through the Sequential container
                                      -- which contains a Parallel container of 10 Linear+Reshape.
                                      -- Each Linear+Reshape module receives a slice of dimension 1
                                      -- which corresponds to a 1D Tensor of size 3.
                                      -- Eventually all the Linear+Reshape modules' outputs of size 2x1
                                      -- are concatenated alond the 2nd dimension (column space)
                                      -- to form pred, a 2D Tensor of size 2x10.


for i = 1, 10000 do     -- Train for a few iterations
 x = torch.randn(10,3);
 y = torch.ones(2,10);
 pred = mlp:forward(x)

 criterion = nn.MSECriterion()
 local err = criterion:forward(pred,y)
 local gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion); 
 mlp:updateParameters(0.01);
 print(err)
end
