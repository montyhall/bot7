------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Three-Hump Camel 2-dimensional benchmarking function:

  f(x) = ((a*z1^2 + b)*z1^2 + c)*z1^2 + z1*z2 + z2^2
       

where
  z := 10*(x - 0.5)
  a := 1/6,  b := -1.05
  c := 2

and, domain(x) := [0,1]^2

Global Minima:
f(x*) = 0.0 at x* = (0.5, 0.5)


Authored: 2015-10-02 (jwilson)
Modified: 2015-10-02
--]]

---------------- External Dependencies
math = require('math')

---------------- Constants
local a = 1/6
local b = -1.05
local c = 2.0
local offset = -0.5
local scale  = 10

------------------------------------------------
--                                   camel_hump3
------------------------------------------------

function camel_hump3(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset):mul(scale)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end

  -------- Compute Three-Hump Camel Function
  local zz1 = Z:select(2,1):clone():pow(2)
  local zz2 = Z:select(2,2):clone():pow(2)
  local Y   = torch.mul(zz1, a):add(b):cmul(zz1):add(c)
                     :cmul(zz1):add(Z:prod(2)):add(zz2)
  return Y
end

