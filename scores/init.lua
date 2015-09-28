------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for bot7 acquisition
functions class (scores).

Authored: 2015-09-17 (jwilson)
Modified: 2015-09-28
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = bot7 or {}
bot7['scores'] = {}
include('metascore.lua')
include('expected_improvement.lua')
include('confidence_bound.lua')
return bot7.scores