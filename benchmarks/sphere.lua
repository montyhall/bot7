------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Sphere n-dimensional benchmarking function

  f(x) = ||z||^2

where:
  z := 10.24*(x - 0.5)

Global Minima:
f(x*) = 0 at x* = (0, ..., 0)

Authored: 2015-10-09 (jwilson)
Modified: 2015-10-10
--]]

---------------- Constants
local offset = -0.5
local scale  = 10.24

------------------------------------------------
--                                        sphere
------------------------------------------------
local sphere = function(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset):mul(scale)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end

  -------- Compute Sphere function
  local Y = Z:pow(2):sum(2)
  return Y
end

return sphere