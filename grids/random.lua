------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Uniform psuedorandom grid.

Authored: 2015-09-18 (jwilson)
Modified: 2015-10-11
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

function grid.generate(config)
  local X = torch.rand(config.size, config.dims)
  if config.mins  then X:add(torch.add(config.mins, X:min(1)[1]):expandAs(X))    end
  if config.maxes then X:cmul(torch.cdiv(config.maxes, X:max(1)[1]):expandAs(X)) end
  return X
end