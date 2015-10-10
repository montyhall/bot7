------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Griewank n-dimensional benchmarking function:

  f(x) = a*mean(z^2) - prod[cos(z_k/sqrt(k))] + b

where
  z := 1200*(x - 0.5)
  a := 1/4000, b := 1

and, domain(x) := [0,1]^n

Global Minima:
f(x*) = 0.0 at x* = (0.5, ..., 0.5)

Authored: 2015-10-02 (jwilson)
Modified: 2015-10-10
--]]

---------------- External Dependencies
math = require('math')

---------------- Constants
local a = 1/4000
local b = 1
local offset = -0.5
local scale  = 1200

------------------------------------------------
--                                      griewank
------------------------------------------------
local griewank = function(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset):mul(scale)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end

  local zDim  = Z:size(2)
  local denom = torch.ones(1, zDim):cdiv(torch.range(1, zDim):sqrt())

  -------- Compute Ackley Function
  local Y = Z:clone():pow(2):mean(2):mul(a):add(b)
             :add(-Z:cmul(denom:expandAs(Z)):cos():prod(2))
  return Y
end

return griewank

