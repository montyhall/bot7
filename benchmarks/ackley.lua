------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Ackley n-dimensional benchmarking function:

  f(x) = a*[1 - exp(-b*sqrt(mean(z^2))] + d - exp(mean(cos(c*z)))

where
  z := 65.536*(x - 0.5)
  a := 20,   b := 0.2
  c := 2*pi, d := exp(1)

and, domain(x) := [0,1]^n

Global Minima:
f(x*) = 0.0 at x* = (0.5, ..., 0.5)


Authored: 2015-10-02 (jwilson)
Modified: 2015-10-10
--]]

---------------- External Dependencies
math = require('math')

---------------- Constants
local a = 20
local b = -0.2
local c = 2.0*math.pi
local d = math.exp(1.0)
local offset = -0.5
local scale  = 65.536

------------------------------------------------
--                                        ackley
------------------------------------------------
local ackley = function(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset):mul(scale)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end

  -------- Compute Ackley Function
  local Y = Z:clone():pow(2):mean(2):sqrt():mul(b):exp():mul(-a)
             :add(-Z:mul(c):cos():mean(2):exp()):add(a + d)
  return Y
end

return ackley