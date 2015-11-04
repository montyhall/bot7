------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Table utility method for bot7.

Authored: 2015-10-30 (jwilson)
Modified: 2015-11-04
--]]

------------------------------------------------
--                                         table
------------------------------------------------
local self = {}

--------------------------------
--         Imports from penlight
--------------------------------
self.find     = require('pl.tablex').find
self.deepcopy = require('pl.tablex').deepcopy

--------------------------------
--                    Table size
--------------------------------
function self.size(tbl, recurse, max_depth, depth)
  local size = tablex.size
  local max_depth = max_depth or math.huge
  local depth = depth or 0
  if not recurse or depth > max_depth then
    return size(tbl)
  else
    local N = 0
    for _, val in pairs(tbl) do
      if type(val) == 'table' then
        N = N + self.size(val, true, max_depth, depth+1)
      else
        N = N + 1
      end
    end
    return N
  end
end

--------------------------------
--                  Update table
--------------------------------
function self.update(res, src, keep)
  local res, src = res or {}, src or {}
  for key, val in pairs(src) do
    if type(res[key] == 'table') and type(val) == 'table' then
      res[key] = self.update(res[key], val, keep)
    else
      if not keep or not res[key] then
        res[key] = val
      end
    end
  end
  return res
end

return self