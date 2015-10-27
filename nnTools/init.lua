------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for nnTools class.

Authored: 2015-09-18 (jwilson)
Modified: 2015-10-27
--]]

---------------- External Dependencies
local paths = require('paths')

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = bot7 or {}
bot7['nnTools'] = {}
bot7.nnTools['builder']   = paths.dofile('builder.lua')
bot7.nnTools['trainer']   = paths.dofile('trainer.lua')
bot7.nnTools['evaluator'] = paths.dofile('evaluator.lua')
return bot7.nnTools

