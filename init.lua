------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initializer for bot7 package.

Package Listing:
  ----------------------------------------
  | Directory  | Content                 |
  ----------------------------------------
  |   utils    | Utility methods         |
  |   models   | Target function models  |
  |   grids    | Input grids             |
  |   scores   | Acquisition functions   |
  |  samplers  | Sampling methods        |
  |    bots    | Automated expt. runners |
  ----------------------------------------

To Do:
  - TPE implementation

Authored: 2015-09-28 (jwilson)
Modified: 2015-10-30
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = {} -- leaks a global called bot7

--------------------------------
--            Standalone Modules
--------------------------------
bot7['utils'] = require('bot7.utils')
include('hyperparam.lua')

--------------------------------
--                 Class Modules
--------------------------------
require('bot7.models')
require('bot7.grids')
require('bot7.scores')
require('bot7.samplers')
require('bot7.bots')
require('bot7.nnTools')

return bot7