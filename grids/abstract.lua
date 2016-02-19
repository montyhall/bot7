------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Abstract base class for bot7 grids.

Authored: 2015-09-18 (jwilson)
Modified: 2016-02-24
--]]

------------------------------------------------
--                                      abstract
------------------------------------------------
local title = 'bot7.grids.abstract'
local grid  = torch.class(title)

function grid:__init()
end

function grid:__call__(config)
  local config = config or self.config
  return self:generate(config)
end

function grid.generate(config)
  print('Error: generate() method not implemented')
end
