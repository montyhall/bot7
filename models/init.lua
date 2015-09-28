------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Initialization file for gpTorch7 models class.

Authored: 2015-09-12 (jwilson)
Modified: 2015-09-28
--]]

------------------------------------------------
--                                   Initializer
------------------------------------------------
bot7 = bot7 or {}
bot7['models'] = require('gp.models')

-------- Check for req. methods
-- local msg
-- for model, val in pairs(bot7.models) do
--   if not model().predict then
--     msg = string.format('Warning: Unable to locate'..
--      'methods %s.predict()', model)
--   end

--   if not model().fantasize then
--     msg = string.format('Warning Unable to locate'..
--       'methods %s.fantasize()', model
--   end
-- end
return bot7.models
