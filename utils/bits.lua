------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Bit manipulation methods

Authored: 2016-02-19 (jwilson)
Modified: 2016-02-19
--]]

---------------- External Dependencies
local utils = require('bot7.utils')

---------------- Constants
local bit_precis = 32
local bit_weight = torch.Tensor(bit_precis):fill(2)
                    :cpow(torch.range(bit_precis-1, 0, -1))

------------------------------------------------
--                                        binary
------------------------------------------------
local self = {}

--------------------------------
--             Decimal to Binary
--------------------------------
function self.dec2bin(dec, bprecis)
  ---- Only support single elements currenty
  local dec = dec:clone()
  assert(dec:dim() == 1 or dec:nElement() == dec:size(1))
  local bprecis = bprecis or bit_precis

  local bp  = bprecis
  local bin = torch.ByteTensor(dec:size(1), bprecis)
  local idx = nil

  ---- Negative values
  idx = dec:lt(0)
  if idx:any() then
    idx = idx:nonzero()
    idx:resize(idx:nElement())
    dec[{idx}]:add(overflow)
  end

  ---- Convert to binary tensor
  bin:fill(0)
  idx = dec:gt(0):nonzero()

  local lsb = utils.math.modulus(dec, 2):byte()
  while bp > 0 do
    if lsb:ne(0):any() then
      bin[{{},{bp}}]:indexFill(1, lsb:nonzero()[{{},1}], 1)
    end
    dec:add(-lsb:typeAs(dec)):div(2):floor()
    bp  = bp - 1
    lsb = utils.math.modulus(dec, 2):byte()
  end
 return bin
end

--------------------------------
--             Binary to Decimal
--------------------------------
function self.bin2dec(bin)
  if bin:dim() == 1 then
    local bprecis = bin:size(1)
    return torch.dot(bin:typeAs(bit_weight), bit_weight:sub(1, bprecis))
  elseif utils.tensor.shape(bin):max() == bin:nElement() then
    local bprecis = bin:nElement() 
    return torch.dot(bin:typeAs(bit_weight), bit_weight:sub(1, bprecis))
  else
    return torch.mv(bin:typeAs(bit_weight), bit_weight:sub(1, bin:size(2)))
  end
end

--------------------------------
--                   Bitwise XOR
--------------------------------
function self.bitwise_xor(x, y)
  local xor = self.dec2bin(torch.Tensor{x}):add(self.dec2bin(torch.Tensor{y})):eq(1)
  return self.bin2dec(xor)
end

return self