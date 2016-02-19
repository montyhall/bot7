------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for bot7 grid class.

Authored: 2015-09-18 (jwilson)
Modified: 2016-02-19
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = bot7 or {}
bot7['grids'] = {}
include('abstract.lua')
include('random.lua')
include('sobol.lua')
return bot7.grids