------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Uniform psuedorandom grid.

Authored: 2015-09-18 (jwilson)
Modified: 2016-02-24
--]]

------------------------------------------------
--                                        random
------------------------------------------------
local title  = 'bot7.grids.random'
local parent = 'bot7.grids.abstract'
local grid, parent = torch.class(title, parent)

function grid:__init(config)
  parent.__init(self)
  self.config = config or {}
end

function grid:generate(config)
  local config = config or self.config
  local grid = torch.rand(config.size, config.dims)

  if config.mins and config.maxes then
    grid:cmul(grid, torch.add(config.maxes, -config.mins):expandAs(grid)):add(config.mins:expandAs(grid))
  elseif config.mins then
    grid:add(torch.add(config.mins, grid:min(1)[1]):expandAs(grid))
  elseif config.maxes then
    grid:cmul(torch.cdiv(config.maxes, grid:max(1)[1]):expandAs(grid))
  end
  return grid
end