------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstact base class for bot7 scientists.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-27
--]]

------------------------------------------------
--                                 metascientist
------------------------------------------------
local title     = 'bot7.scientists.metascientist'
local scientist = torch.class(title)

function scientist:__init()
end

function scientist:__call__()
end

function scientist.eval()
  print('Error: eval() method not implemented')
end

function scientist.nominate()
  print('Error: nominate() method not implemented')
end
