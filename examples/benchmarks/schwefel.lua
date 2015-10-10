------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Schwefel n-dimensional benchmarking function.
  
  f(x) = a*n - sum(z * sin(sqrt(|z|)), 2)

where: 
  n := dimensionality of x
  a := 418.9829
  z := 1000 * (x - 0.5)

Global Minima:
f(x*) = 0 at:
  x* ~= (0.920969, ..., 0.920969)
  z* ~= (420.9687, ..., 420.9687)

Authored: 2015-10-09 (jwilson)
Modified: 2015-10-10
--]]

---------------- Constants
local a = 418.9829
local offset = -0.5
local scale  = 1000

------------------------------------------------
--                                      schwefel
------------------------------------------------
local schwefel = function(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset):mul(scale)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end

  -------- Compute Schwefel function
  local Y = Z:cmul(torch.abs(Z):sqrt():sin()):sum(2):mul(-1):add(a*Z:size(2))
  return Y
end

return schwefel
