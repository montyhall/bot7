------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Six-Hump Camel 2-dimensional benchmarking function:

  f(x) = ((a*z1^2 + b)*z1^2 + c)*z1^2 + z1*z2 + c*(z2^2 - 1)z2^2
       
where
  z1 := 6*(x - 0.5)
  z2 := 4*(x - 0.5)
  a  := 1/3
  b  := -2.1
  c  := 4.0

and, domain(x) := [0,1]^2

Global Minima:
f(x*)~= -1.0316 at:
  x* ~= {(.5150,  .3219), (.4850,  .6782)}
  z* ~= {(.0898, -.7126), (-.0898, .7126)}


Authored: 2015-10-02 (jwilson)
Modified: 2015-11-17
--]]

---------------- External Dependencies
math = require('math')

---------------- Constants
local a = 1/3
local b = -2.0
local c = 4
local offset = -0.5
local scale  = torch.Tensor{{6,4}}

------------------------------------------------
--                                   camel_hump6
------------------------------------------------
local camel_hump6 = function(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end
  Z:cmul(scale:expandAs(Z))

  -------- Compute Six-Hump Camel Function
  local zz1 = Z:select(2,1):clone():pow(2)
  local zz2 = Z:select(2,2):clone():pow(2)
  local Y   = torch.mul(zz1, a):add(b):cmul(zz1):add(c):cmul(zz1)
                 :add(Z:prod(2)):add(torch.add(zz2, -1):cmul(zz2):mul(c))
  return Y
end

return camel_hump6
