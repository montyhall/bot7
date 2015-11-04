------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Tensor utility method for bot7.

Authored: 2015-10-30 (jwilson)
Modified: 2015-11-04
--]]

------------------------------------------------
--                                        tensor
------------------------------------------------
local self = {}

--------------------------------
--        Return target as value
--------------------------------
function self.number(x, idx, axis)
  if torch.isTensor(x) then
    local idx   = idx or 1
    local shape = x:size()
    local axis  = axis
    local nDims = x:dim()
    if not axis then
      axis = 1
      while((shape[axis] == 1 or shape[axis] < idx) and axis < nDims) do
        axis = axis + 1
      end
    end
    shape[axis] = idx
    return x[shape]
  else
    local typ = type(x)
    if typ == 'number' then
      return x
    elseif typ == table then
      return self.number(x[idx])
    else
      print('Error: Unrecognized input type '..
            'encountered in self.tensor.number(); '..
            'returning.')
    end
  end
end

--------------------------------
--           Return shape tensor
--------------------------------
function self.shape(tnsr)
   local shape = torch.LongTensor(tnsr:dim())
         shape:storage():copy(tnsr:size())
  return shape
end

--------------------------------
--              Tensor to String
--------------------------------
function self.string(tnsr, config)
  local tnsr    = tnsr
  local nDims   = tnsr:dim()
  local config  = config or {}
  config.delim  = config.delim  or ''
  config.format = config.format or '%.2e'
  config.align  = config.align  or 'horiz'
  config.rectangular = config.rectangular or true

  ---- Handle tensor dimensionality 
  if (nDims == 0) then
    print('0-dimensional tensors cannot be converted to string.')
    return
  end

  local N = tnsr:nElement()
  if nDims == 1 or self.shape(tnsr):max() == N then
    tnsr = tnsr:reshape(1, tnsr:nElement())
  end

  local nRow, nCol = tnsr:size(1), tnsr:size(2)

  if (nDims > 2)  then
    if nRow*nCol == tnsr:nElement() then
      tnsr = tnsr:reshape(nRow, nCol) -- dont squeeze, avoid 1x1
    else
      print('Support for >2 tensors not currently available for tnsr2str()')
      return
    end
  end 

  local vertical = (config.align == 'vert' or align == 'vertical')

  ---- Reshape to specified output style
  if config.rectangular then
    if vertical then
      nCol = math.ceil(math.sqrt(N))
      nRow = math.ceil(N/nCol)
    else
      nRow = math.ceil(math.sqrt(N))
      nCol = math.ceil(N/nRow)
    end
    if nCol*nRow > N then
      local pad = torch.Tensor(nCol*nRow - N, 1):fill(0/0)
      tnsr = torch.cat(tnsr, pad):resize(nRow, nCol)
    else
      tnsr = tnsr:reshape(nRow, nCol)
    end
  elseif self.shape(tnsr):max() == N and vertical then
    tnsr = tnsr:t()
  end

  ---- Construct string representation
  local count, str = 0, ''
  for row = 1, nRow do
    for col = 1, nCol do  
      if count < N then
        str = str .. string.format(config.format, tnsr[row][col])
        if col < nCol then str = str .. config.delim .. ' ' end
      end
     count = count + 1
    end
    if row < nRow then str = str .. '\n' end
  end
  return str
end

--------------------------------
--        Safer append to tensor
--------------------------------
function self.append(tnsr, subtnsr, axis)
  local axis = axis or 1
  if torch.isTensor(tnsr) and tnsr:dim() > 0 then
    tnsr = torch.cat(tnsr, subtnsr, axis)
  else
    tnsr = subtnsr:clone()
    if axis == 1 and (tnsr:dim() == 1 or tnsr:size(1) == tnsr:nElement()) then
      tnsr:resize(1, tnsr:nElement())
    end
  end
  return tnsr
end

--------------------------------
--     Remove slices from tensor
--------------------------------
function self.remove(tnsr, idx, axis)
  -------- Determine elements maintained by tnsr
  local axis  = axis or 1
  local Ns    = tnsr:size(axis)
  local keep  = torch.range(1, Ns, 'torch.LongTensor')
        keep:indexFill(1, idx, 0)
        keep  = keep:nonzero()
  if keep:dim() > 0 then
    return tnsr:index(axis, keep:select(2,1))
  else
    return nil
  end
end

--------------------------------
-- Move slice(s) from src to res
--------------------------------
function self.steal(res, src, idx, axis_r, axis_s)
  local res, src, idx = res, src, idx

  local axis_r = axis_r or 1
  local axis_s = axis_s or 1

  if idx:dim() > 1 then 
    idx = idx:resize(idx:nElement()) 
  end
  
  -------- Append slices to res
  res = self.append(res, src:index(axis_s, idx), axis_r)

  -------- Remove slices from src
  src = self.remove(src, idx, axis_s)

  collectgarbage()
  return res, src
end


--------------------------------
--              Tensor Transpose
--------------------------------
function  self.transpose(tnsr, inplace)
  local nDim = tnsr:dim()

  if inplace then res = tnsr
  else res = tnsr:clone() end

  for d = 1, math.floor(nDim/2) do
    res = res:transpose(d, nDim-d+1):clone()
  end
  collectgarbage()
  return res
end

--------------------------------
--               Tensor Reversal
--------------------------------
function self.flip(tnsr, axis, idx0, idx1, inplace)
  local axis = axis or 1
  local idx0 = idx0 or 1
  local idx1 = idx1 or tnsr:size(axis)
  if inplace then
    tnsr = tnsr:index(axis, torch.range(-idx1, -idx0, 'torch.LongTensor'):mul(-1))
    return tnsr
  else
    return tnsr:index(axis, torch.range(-idx1, -idx0, 'torch.LongTensor'):mul(-1))
  end
end

--------------------------------
--               Linear Indexing
--------------------------------
function self.linear_index(shape, coords, order)
  local nDim   = shape:nElement()
  local order  = order or 'C'

  local coords = coords
  if (coords:dim() == 1) then
    coords = coords:reshape(coords:nElement(), 1)
  end

  local offset
  if order == 'C' then
    offset = self.flip(torch.cat(torch.ones(1, 'torch.LongTensor'),
                        self.flip(shape, 1, 2):cumprod(), 1))
  elseif order == 'F' then
    offset = torch.cat(torch.ones(1, 'torch.LongTensor'), shape:sub(1, nDim-1), 1):cumprod()
  end
  return torch.mv(coords - 1, offset):add(1)
end

--------------------------------
--   Linear indices of mat diag
--------------------------------
function self.diag_indices(N)
  return torch.range(1, N^2, N+1, 'torch.LongTensor')
end

--------------------------------
--   Linear indices of lower tri
--------------------------------
function self.tril_indices(N)
  local idx   = torch.LongTensor((N^2+N)/2)
  local count = 1
  for row = 1,N do
    idx[{{count, count+row-1}}] = torch.range(1, row):add(N*row-N)
    count = count + row
  end
  return idx
end

--------------------------------
--   Linear indices of upper tri
--------------------------------
function self.triu_indices(N)
  local idx   = torch.LongTensor((N^2+N)/2)
  local count = 1
  for row = 0,N-1 do
    idx[{{count, count+N-row-1}}] = torch.range(1+row, N):add(N*row)
    count = count + N - row
  end
  return idx
end

--------------------------------
--          Tensor Vectorization
--------------------------------
function self.vect(tnsr, order, inplace)
  local N, nDim = tnsr:nElement(), tnsr:dim()
  local order   = order or 'F'
  local shape   = self.shape(tnsr)

  -------- noop, already vectorized
  if (shape:max() == N) then return tnsr end

  -------- Vectorize in-place
  if inplace then res = tnsr
  else res = tnsr:clone() end

  -------- Vectorize using C indexing convention 
  -- Row-major order: Indices of last dimension change first.
  -- Tensors are natively row-major; so, just call resize().
  if order == 'C' then
    res:resize(N,1)

  -------- Vectorize using Fortran indexing convention
  -- Column-major order: Indices of leading dimension change first
  elseif order == 'F' then
    self.transpose(res, true)
    res:resize(N,1)
  end

  collectgarbage()
  return res
end

--------------------------------
--  Wrapper for tensor.indexCopy
--------------------------------
function self.indexCopy(res, src, idx, axis_r, axis_s)
  local axis_r = axis_r or res:dim()
  local axis_s = axis_s or src:dim()
  return res:indexCopy(axis_r, idx, src:index(axis_s, idx))
end

return self