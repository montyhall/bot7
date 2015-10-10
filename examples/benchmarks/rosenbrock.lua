------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
(Rescaled) Rosenbrock Valley n-dimensional
benchmarking function:

  f(x) = sum_{i=[1, n-1]}[a*(z_i^2 - z_{i+1})^2 + (z_i + b)^2]

where:
 a := 100
 b := -1.0
 z := 15*(x - 1/3)

and, domain(x) := [0,1]^n

Global Minima:
f(x*) = 0 at 
  x* := (0.4, 0.4, 0.4, 0.4)
  z* := (1.0, 1.0, 1.0, 1.0)

  
Authored: 2015-09-18 (jwilson)
Modified: 2015-10-10
--]]

---------------- Constants
local a = 100
local b = -1.0
local offset = -1/3
local scale  = 15
------------------------------------------------
--                                    rosenbrock
------------------------------------------------
local rosenbrock = function(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset):mul(scale)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end
  -------- Compute Rosenbrock Valley Function (Rescaled)
  local Y = torch.add(torch.pow(Z:narrow(2,1,3), 2), -Z:narrow(2,2,3)):pow(2):mul(a)
                 :add(torch.add(Z:narrow(2,1,3), b):pow(2)):sum(2)
  return Y
end

return rosenbrock