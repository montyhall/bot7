------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for gpTorch7 models class.

Authored: 2015-09-12 (jwilson)
Modified: 2015-10-29
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = bot7 or {}
bot7['models'] = require('gp.models')
include('abstract.lua')
include('dngo.lua')

return bot7.models
