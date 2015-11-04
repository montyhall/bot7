------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Styblinski-Tang n-dimensional benchmarking 
function:
  
  f(x) = a * sum(z^4 - b*z^3 + c*z, 2)

where:
  a := 0.5
  b := 16
  c := 5
  z := 10 * (x - 0.5)

Global Minima:
f(x*) = -39.16599*n at:
    x* ~= (0.2096466, ..., 0.2096466)
    z* ~= (-2.903534, ..., -2.903534)  

Authored: 2015-10-09 (jwilson)
Modified: 2015-10-10
--]]

---------------- Constants
local a = 0.5
local b = 16
local c = 5
local offset = -0.5
local scale  = 10

------------------------------------------------
--                               styblinski_tang
------------------------------------------------
local styblinski_tang = function(X)
  -------- Transform X -> Z 
  local Z = torch.add(X, offset):mul(scale)
  if (Z:dim() == 1 or Z:size(1) == Z:nElement()) then
    Z:resize(1, Z:nElement())
  end

  -------- Compute Styblinski-Tang function
  local Y = torch.pow(Z, 4):add(-torch.pow(Z, 2):mul(b))
                    :add(torch.mul(Z, c)):sum(2):mul(a)
  return Y
end

return styblinski_tang