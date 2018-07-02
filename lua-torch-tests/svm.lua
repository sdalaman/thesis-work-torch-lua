require "nnx"
ninputs = 2; noutputs = 1;
svm = nn.Linear(ninputs, noutputs)
loss = nn.MarginCriterion()   ----

input= torch.randn(2);     -- normally distributed example in 2d
output= torch.Tensor(1);
if input[1]*input[2] > 0 then  -- calculate label for XOR function
   output[1] = -1
else
   output[1] = 1
end

for i = 1,2500 do
  -- feed it to the neural network and the criterion
  a = loss:forward(svm:forward(input), output)
 print(a)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  svm:zeroGradParameters()
  -- (2) accumulate gradients
  svm:backward(input, loss:backward(svm.output, output))
  -- (3) update parameters with a 0.01 learning rate
  svm:updateParameters(0.01)
end

