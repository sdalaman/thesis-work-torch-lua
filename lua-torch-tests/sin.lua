require 'nn'
local sinx, Parent = torch.class('nn.sinx', 'nn.Module')

function sinx:__init()
	Parent.__init(self)
end

function sinx:updateOutput(input)
	self.output:resizeAs(input):copy(input)
	self.output:sin()
	return self.output
end

---function sinx:updateGradInput( input, gradOutput )
--	self.gradInput:resizeAs(gradOutput):copy(gradOutput)
--	return self.gradInput:cos()  -- gradient of sin
--end

--Your derivation of the chain rule is not correct:
--If you module input is x and output is y, you have y = sin(x). Given that gradOuput is dL/dy and you want dL/dx.
--dL/dx = dL/dy * dy/dx = dL/dy * cos(x) = gradOutput * cos(input)   Where * is the element-wise product.

--Thus your `updateGradInput` function should be:

function sinx:updateGradInput( input, gradOutput )
   self.gradInput:resizeAs(input):copy(input):cos() -- dy/dx
   return self.gradInput:cmul(gradOutput) -- element-wise product
end



-- testing gradient implementation with Jacobian 

local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)

local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.sinx()

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
	print(err)
	print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end