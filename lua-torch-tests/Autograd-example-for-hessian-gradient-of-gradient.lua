--
-- Created by IntelliJ IDEA.
-- User: taineleau
-- Date: 10/7/16
-- Time: 10:16
-- To change this template use File | Settings | File Templates.
--

local grad = require 'autograd'
grad.optimize(true)

----------------------------------------------------------------------------
--- example 1: ax^2 + by^2 + cxy
----------------------------------------------------------------------------

a = torch.randn(1, 1)
b = torch.randn(1, 1)
c = torch.randn(1, 1)
x = torch.randn(1, 1)
y = torch.randn(1, 1)

print('a', a, 'b', b, 'c', c, 'x', x, 'y', y)


local innerFn = function(x, a, b, c, y)
   local result = torch.sum(torch.cmul(a, torch.cmul(x, x))) +
           torch.sum(torch.cmul(b, torch.cmul(y, y))) + torch.sum(torch.cmul(c, torch.cmul(x, y)))
   return result
end

params_inner = {x = x}

local dinnerFn = grad(innerFn)

print('first dervative size w.r.t. x')
grads, ans = dinnerFn(x, a, b, c, y)
print(grads)
print(2 * torch.sum(torch.cmul(a, x)) + torch.sum(torch.cmul(c, y)))



local outerFn = function(y, a, b, c, x)
   local grads, ans = dinnerFn(x, a, b, c, y)
   local result = torch.sum(grads)
   return result
end

local douterFn = grad(outerFn)
grads, ans = douterFn(y, a, b, c, x)
print('second dervative (dxdy)')
print(grads)


----------------------------------------------------------------------------
--- example 2
----------------------------------------------------------------------------

--simple linear model with squared loss
local params = {
   W = torch.randn(100,10),
   L = torch.randn(100,10),
}

print(params)

local innerFn = function(params, x, y)
   local yHat = x*params.W
   local squaredLoss = torch.sum(torch.pow(y - yHat,2))
   local loss = torch.cmul(params.L, torch.cmul(params.W, params.W))
   return torch.sum(loss) + squaredLoss
end

local dneuralNet = grad(innerFn)

--synthetic data
local x = torch.randn(1,100)
local y = torch.randn(1,10)

print('first derivative size')
local grads = dneuralNet(params,x,y)
print(grads.W, grads.L)


--- the outer function computes the sum of the gradient of the neural network.
-- Therefore, differentiating this returns the diagonal of the Hessian
local outerFn = function(params,x,y)
   local grad = dneuralNet(params, x, y)
   local sum = torch.sum(grad.W) + torch.sum(grad.L)
   return sum
end

print('outer function value')
print(outerFn(params,x,y))

local ddf = grad(outerFn)
local gradGrads = ddf(params,x,y)

print('second derivative')
print(gradGrads.L, gradGrads.W)