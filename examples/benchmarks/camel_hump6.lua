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
Modified: 2015-10-10
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

-- function camel_hump6_grad(X)
--   -------- Transform x -> (z1, z2)
--   local z1    = torch.add(X[{{},{1}}], offset):mul(scale[{1,1}])
--   local z2    = torch.add(X[{{},{2}}], offset):mul(scale[{1,2}])
--   local grads = torch.Tensor(X:size(1),2)

--   -------- Gradient of f(x) w.r.t. z1
--   grads[{{},{1}}] = torch.pow(z1, 5):mul(6*a):add(
--                     torch.pow(z1, 3):mul(4*b)):add(
--                     z1:clone():mul(2*c)):add(z2)

--   -------- Gradient of f(x) w.r.t. z2           
--   grads[{{},{2}}] = torch.pow(z2, 3):mul(4):add(torch.mul(z2, -2)):mul(c):add(z1)

--   -------- Factor in dz/dx
--   grads:cmul(scale:expandAs(grads))
--   return grads
-- end