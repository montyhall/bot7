------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Uniform psuedorandom grid.

Authored: 2015-09-18 (jwilson)
Modified: 2015-09-30
--]]

------------------------------------------------
--                                        random
------------------------------------------------
local title  = 'bot7.grids.random'
local parent = 'bot7.grids.metagrid'
local grid, parent = torch.class(title, parent)

function grid:__init(config)
  parent.__init(self)
  self.config = config or {}
end

function grid.generate(config)
  local X = torch.rand(config.size, config.dims)
        X:cmul((config.maxes - config.mins):expandAs(X)):add(config.mins:expandAs(X)) 
  return X
end