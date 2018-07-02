gfx = require ("gfx.js")
 
local groups = {}
for k = 1, 4 do
   local group = {}
   group.key = "Group" .. k
   
   group.values = {}
   for i = 1, 40 do
      table.insert(group.values, 
        { 
           x = math.random(), 
           y = math.random(), 
           size = torch.random(1,16)
        })
   end
 
   groups[k] = group   
end
 
local config = 
{
      chart = 'scatter', 
      width = 640,
      height = 480,
      tooltipContent = "$: function(key) {return '<h3>' + key + '</h3>'; }", } 
 
gfx.chart(groups, config)