------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Hyperparameter class for bot7; used for defining
setting (hyperparameters) of interest in the 
context of experiments run via bot7.

Args:
  keys : Specification of path to hyperparameter
         relative to present stack [Required]
  typ  : Value type: {float, int}
  min  : Minimum feasible value
  min  : Maximum feasible value
  size : Size of hyperparameter <<Only supports size 1>>
  warping: Value warping function 

To Do:
  - Support for hyperparameters of size >1
  - Categorical value types
  - Better integration of discrete value types

Authored: 2015-10-26 (jwilson)
Modified: 2015-10-26
--]]

---------------- External Dependencies
local utils = require('bot7.utils')

------------------------------------------------
--                                    hyperparam
------------------------------------------------
local title = 'bot7.hyperparam'
local hyper = torch.class(title)

function hyper:__init(key, min, max, typ, size, warping)
  -------- Required fields
  self.key = key

  -------- Optional fields
  self.min  = min or 0
  self.max  = max or 1
  self.type = typ or 'float'
  self.size = size or 1

  -------- Type Checking
  assert(self.type == 'float' or self.type == 'int')
  if type == 'int' then self.ttype = 'LongTensor'
  else                  self.ttype = 'Tensor' end

  if type(self.min) == 'table' then
    self.min = torch[self.ttype](self.min)
  elseif not torch.isTensor(self.min) then
    self.min = torch[self.ttype](self.size):fill(self.min)
  end

  if type(self.max) == 'table' then
    self.max = torch[self.ttype](self.max)
  elseif not torch.isTensor(self.max) then
    self.max = torch[self.ttype](self.size):fill(self.max)
  end

  -------- Establish warping function
  self:make_warping(warping or 'default')
end

function hyper:__call__(x)
  return self:warp(x)
end

function hyper:make_warping(f)
  -------- User-defined warping functions
  if type(f) == 'function' then
    self.warp = f
  
  -------- Rescaled log-space warping 
  -- Temp assumption: self.size == 1
  elseif f == 'default' then
    local logspace = false
    local wmin, wmax = nil, nil

    if self.min:eq(0):all() then
      if self.max:gt(10):all() then
        logspace, self.offset = true, torch.Tensor{1}
      end
    elseif torch.cdiv(self.max, self.min):gt(10):all() then
      -- logspace, self.offset = true, torch.zeros(self.size)
      logspace, self.offset = true, torch.add(torch.ones(self.min:size()), -self.min)
    end

    if logspace then
      self.wmin = torch.log(torch.add(self.min, self.offset)) 
      self.wmax = torch.log(torch.add(self.max, self.offset))

      function self:warp(x)
        if torch.isTensor(x) then
          return torch.add(torch.add(torch.ones(x:size()), -x):cmul(self.wmin:expandAs(x)),
                           torch.cmul(x, torch.add(self.wmax, -self.wmin):expandAs(x)))
                          :exp():add(-self.offset:expandAs(x))
        else
          return torch.exp((1-x)*self.wmin[1] + x*(self.wmax[1]-self.wmin[1])) - self.offset[1]
        end
      end
    else
      function self:warp(x) return x end
    end
  end
end

return hyper