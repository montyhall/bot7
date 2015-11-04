------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for benchmarking functions
module.

Difficulty ratings assigned as 100 - score,
according to scores posted on:
  http://infinity77.net/global_optimization
  /test_functions.html#

Scores for n-dimensional functions were
evaluated for n=2. 

Authored: 2015-10-10 (jwilson)
Modified: 2015-10-10
--]]

---------------- External Dependencies
local paths = require('paths') 

------------------------------------------------
--                                    benchmarks
------------------------------------------------
local handles = { --[[
-------------------------------------------------------
|    Function Handle    | Dim | Diff. | #gmin | #lmin |
------------------------------------------------------- ]]
       'ackley',     -- |  2  | 51.75 |   1   |  >>n  |
      'braninhoo',   -- |  2  | 30.50 |   1   |   3   |
    'camel_hump3',   -- |  2  | 21.17 |   1   |   3   |
    'camel_hump6',   -- |  2  | 17.83 |   2   |   4   |
      'griewank',    -- |  n  | 93.92 |   1   |  >>n  |
     'hartmann3',    -- |  3  | 40.17 |   1   |   4   |
     'hartmann6',    -- |  6  | 51.50 |   1   |   6   |
     'rastrigin',    -- |  n  | 60.50 |   1   |  >>n  |
     'rosenbrock',   -- |  n  | 55.83 |   1   |   1   |
     'schwefel',     -- |  n  | 17.00 |   1   |  >>n  |
      'sphere',      -- |  n  | 17.25 |   1   |   n   | 
  'styblinski_tang', -- |  n  | 29.50 |   1   |  >n   |
-------------------------------------------------------
}          
local benchmarks = benchmarks or {}
for _, handle in pairs(handles) do
  benchmarks[handle] = paths.dofile(handle..'.lua')
end

return benchmarks