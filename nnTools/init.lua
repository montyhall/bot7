------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for nnTools class.

Authored: 2015-09-18 (jwilson)
Modified: 2015-11-16
--]]

---------------- External Dependencies
local paths = require('paths')

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = bot7 or {}
bot7['nnTools'] = {}
include('buffers.lua')
include('timer.lua')

bot7.nnTools['builder']   = paths.dofile('builder.lua')
bot7.nnTools['trainer']   = paths.dofile('trainer.lua')
bot7.nnTools['evaluator'] = paths.dofile('evaluator.lua')
bot7.nnTools['automator'] = paths.dofile('automator.lua')
bot7.nnTools['preprocessor'] = paths.dofile('preprocessor.lua')
return bot7.nnTools

