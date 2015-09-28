------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstact base class for bot7 bots.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-28
--]]

------------------------------------------------
--                                       metabot
------------------------------------------------
local title = 'bot7.bots.metabot'
local bot   = torch.class(title)

function bot:__init()
end

function bot:__call__()
end

function bot.eval()
  print('Error: eval() method not implemented')
end

function bot.nominate()
  print('Error: nominate() method not implemented')
end
