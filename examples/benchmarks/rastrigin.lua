------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Rastrigin n-dimensional benchmarking function.

  f(x) = a*n + sum(z^2 - a*cos(b*z), 2)

where:
  n := dimensionality of x
  a := 10
  b := 2*pi
  z := 10.24*(x - 0.5)

Global Minima:
f(x*) = 0 at x* = (0, ..., 0)

Authored: 2015-10-09 (jwilson)
Modified: 2015-10-09
--]]

---------------- External Dependencies
local math = require('math')

---------------- Constants
local a = 10.0
local b = 2*math.pi
local offset = -0.5
local scale  = 10.24


------------------------------------------------
--                                     rastrigin
------------------------------------------------
function rastrigin(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset):mul(scale)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end

  -------- Compute Rastrigin function
  local Y = torch.pow(Z, 2):add(torch.mul(Z, b):cos():mul(-a)):sum(2):add(a*Z:size(2))
  return Y
end
