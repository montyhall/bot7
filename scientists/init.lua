------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for bot7 scientist class.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-28
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = bot7 or {}
bot7['scientists'] = {}
include('metascientist.lua')
include('random_search.lua')
include('bayesopt.lua')
return bot7.scientists
