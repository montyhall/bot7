------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initializer for bot7 utilities.

Authored: 2015-10-30 (jwilson)
Modified: 2015-11-04
--]]

---------------- External Dependencies
local paths = require('paths')

------------------------------------------------
--                                         utils
------------------------------------------------
local bot7 = bot7 or {}
bot7['utils'] = {}

local utils     = bot7.utils
utils['table']  = paths.dofile('table.lua')
utils['tensor'] = paths.dofile('tensor.lua')
utils['math']   = paths.dofile('math.lua')
utils['ui']     = paths.dofile('ui.lua')

return bot7.utils
