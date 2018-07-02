gfx = require ("gfx.js")
 
local x = torch.linspace(-2*math.pi,2*math.pi)
 
local y1 = torch.sin(x)
local y2 = torch.cos(x):mul(0.5)
 
local data = {
   {
       key    = 'Sine Wave',
       values = x:cat(y1, 2),
       color  = '#ff7f0e',
   },
   {
       key = 'Cosine Wave',
       values = x:cat(y2, 2),
       color  = '#2ca02c',
   }
}
 
local config = 
{
      chart = 'line', 
      width = 640,
      height = 480,
      xLabel = "time (ms)",
      yLabel = "",
      useInteractiveGuideline = true
}
 
gfx.chart(data, config)
