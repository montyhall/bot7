------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initializer for bot7 package.

Package Listing:
  -----------------------------------------
  | Directory  | Content                  |
  -----------------------------------------
  |   utils    | Utility methods          |
  |   models   | Target function models   |
  |   grids    | Input grids              |
  |   scores   | Acquisition functions    |
  |  samplers  | Sampling methods         |
  | scientists | Experiment runners       |
  -----------------------------------------

To Do:
  - Neural net demos

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-27
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = {} -- leaks a global called bot7

--------------------------------
--            Standalone Modules
--------------------------------
bot7['utils'] = require('bot7.utils')

--------------------------------
--                 Class Modules
--------------------------------
require('bot7.models')
require('bot7.grids')
require('bot7.scores')
require('bot7.samplers')
require('bot7.scientists')

return bot7