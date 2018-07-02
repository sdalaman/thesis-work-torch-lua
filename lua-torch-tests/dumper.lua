#! /usr/local/torch/install/bin/th

require "nn"

function grab_weights(weight_accumulator, bias_accumulator, module)
	if type(module) == "table" then
		if module.weight ~= nil then
			table.insert(weight_accumulator, module.weight)
		end
		if module.bias ~= nil then
			table.insert(bias_accumulator, module.bias)
		end
		for k, v in pairs(module) do
			grab_weights(weight_accumulator, bias_accumulator, v)
		end
	end
	return weight_accumulator, bias_accumulator
end

function main()
	local mlp = nn.Sequential():
		add(nn.Linear(2,3)):
		add(nn.Tanh()):
		add(nn.Linear(3,1)):
		add(nn.Sigmoid())

	local bury_it_deep = {
		a={
			b={
				c='DECOY! HA!',
				d=12
			}
		},
		x={
			y={
				z=mlp
			}
		}
	}
	weights, biases = grab_weights({},{},bury_it_deep)
	print(weights)
	print(biases)
end

main()
