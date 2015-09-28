------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Rosenbrock Valley 4D benchmarking function (rescaled):

  f(x) = c1*(sum_{i=[1,3]}[100(z_{i+1} - z_i^2)^2 + (1-z_i)^2] - c2)

where:
 c1  := 3.755e5^(-1)
 c2  := 3.827e5
 z_i := 15x_i - 5 forall i = [1,4]

and, domain(x) := [0,1]^4

Global Minima:
f(x*) = 0 at x* := (1, 1, 1, 1)

  
Authored: 2015-09-18 (jwilson)
Modified: 2015-09-25
--]]

---------------- External Dependencies
math = require('math')

---------------- Constants
local c1 = 1.0/3.755e6
local c2 = -3.827e6
------------------------------------------------
--                                    rosenbrock
------------------------------------------------
function rosenbrock(X)
  if (X:dim() == 1 or X:size(1) == X:nElement()) then
    X = X:resize(1, X:nElement())
  end
  assert(X:size(2) == 4)

  print('Warning: Rosenbrock Valley function may be bugged.'..
        'At x=(1,1,1,1), function returns -0.3720 instead of 0.0')

  -------- Transform X -> Z 
  local Z = torch.mul(X, 15):add(-5)


  -------- Compute Rosenbrock Valley Function (rescaled)   
  local Y = Z:narrow(2,2,3):clone():add(Z:narrow(2,1,3):clone():pow(2):mul(-1)):pow(2):mul(100)
             :add(Z:narrow(2,1,3):clone():mul(-1):add(1):pow(2)):sum(2):add(c2):mul(c1)

  return Y
end

-------- Expanded Version 
-- local Y = torch.zeros(X:size(1),1)
-- for idx = 1,3 do
--   Y:add((Z:select(2,idx+1):clone() - Z:select(2,idx):clone():pow(2)):pow(2):mul(100)
--           :add(Z:select(2,idx):clone():mul(-1):add(1):pow(2)))
-- end
-- Y:add(c2):mul(c1)

