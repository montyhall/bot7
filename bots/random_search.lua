------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Random search bot class for bot7.

Authored: 2015-09-18 (jwilson)
Modified: 2015-10-10
--]]

---------------- External Dependencies
local math  = require('math')
local utils = require('bot7.utils')

------------------------------------------------
--                                 random_search
------------------------------------------------
local title  = 'bot7.bots.random_search'
local parent = 'bot7.bots.metabot'
local bot, parent = torch.class(title, parent)

function bot:eval()
  -- Random search; so, do nothing. 
  -- Message here to appease warning.
end

---------------- Nominate a candidate 
function bot:nominate()
  return torch.rand(1):mul(self.candidates:size(1)-1):add(1.5):long()
end
