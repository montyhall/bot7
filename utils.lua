------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Utility methods for bot7.

Authored: 2015-09-12 (jwilson)
Modified: 2015-10-28
--]]

---------------- External Dependencies
local math = require('math')

---------------- Constants
local sqrt2_inv   = 1/math.sqrt(2)
local sqrt2pi_inv = 1/math.sqrt(2*math.pi)
local log2pi      = math.log(2*math.pi)

------------------------------------------------
--                                         utils
------------------------------------------------
local utils = {}

--------------------------------
--         Imports from penlight
--------------------------------
utils.find     = require('pl.tablex').find
utils.deepcopy = require('pl.tablex').deepcopy


--------------------------------
--                    Table size
--------------------------------
function utils.tbl_size(tbl, recurse, max_depth, depth)
  local size = tablex.size
  local max_depth = max_depth or math.huge
  local depth = depth or 0
  if not recurse or depth > max_depth then
    return size(tbl)
  else
    local N = 0
    for _, val in pairs(tbl) do
      if type(val) == 'table' then
        N = N + utils.tbl_size(val, true, max_depth, depth+1)
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
function utils.tbl_update(res, src, keep)
  local res, src = res or {}, src or {}
  for key, val in pairs(src) do
    if type(res[key] == 'table') and type(val) == 'table' then
      res[key] = utils.tbl_update(res[key], val, keep)
    else
      if not keep or not res[key] then
        res[key] = val
      end
    end
  end
  return res
end

--------------------------------
--              Tensor to String
--------------------------------
function utils.tnsr2str(tnsr, config)
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
  if nDims == 1 or utils.shape(tnsr):max() == N then
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
  elseif utils.shape(tnsr):max() == N and vertical then
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
--          Print section header
--------------------------------
function utils.printSection(str, config)
  if not os then require 'os' end
  local config   = config or {}

  local str     = str   or ''
  local width   = config.width or 48
  local tstamp  = config.timestamp or config.time
  local rdelim  = config.ldelim or config.delim or ''
  local ldelim  = config.rdelim or config.delim or ''
  local ljust   = config.ljust or false
  local newline = config.newline or config.newline == nil
  local bar     = config.bar or '='

  local msg  = ldelim
  local len  = ldelim:len() + rdelim:len() + str:len()
  local time = nil
  if tstamp == nil then tstamp = 'auto' end
  if tstamp then
    time = os.date("%d/%m - %H:%M%p ")
    local tlen = time:len()
    if tstamp == true or (tstamp == 'auto' and len+tlen <= width) then
      len = len + tlen
    end
  end

  local reps = width/bar:len()
  bar = string.rep(bar, reps)

  if ljust then
    msg = '\n' .. msg .. str .. string.rep(' ', width - len)
    if time then msg = msg .. time end
    msg = msg .. rdelim
  else
    if time then msg = msg .. time end
    msg = '\n' .. msg .. string.rep(' ', width - len) .. str .. rdelim
  end
  
  if newline then
    msg = '\n' .. bar .. msg .. '\n' .. bar
  else
    msg = bar .. msg .. '\n' .. bar
  end

  print(msg)
end

--------------------------------
--   Print command line argument
--------------------------------
function utils.printArgs()
  local nArgs = #arg
  if nArgs == 0 then return end
  utils.printSection('Command Line Arguments:', 32)
  local count,k = 0, 0
  while k < nArgs do
    count, k = count+1, k+1
    if arg[k] ~= '-verbose' then
      if k < #arg and arg[k+1]:sub(1,1) ~= '-' then
        print(string.format('[%d]  %s \t %s',count, arg[k],arg[k+1]))
        k = k+1
      else
        print(string.format('[%d]  %s',count, arg[k]))
      end
    end
  end
  print(' ')
end

--------------------------------
--        Return target as value
--------------------------------
-- NEEDS WORK
function utils.as_val(x, idx)
  local idx = idx or 1
  if torch.isTensor(x) then
    return x:clone():storage()[idx] -- sloppy
  else
    return x
  end
end

--------------------------------
--           Return shape tensor
--------------------------------
function utils.shape(tnsr)
   local shape = torch.LongTensor(tnsr:dim())
         shape:storage():copy(tnsr:size())
  return shape
end

--------------------------------
--        Safer append to tensor
--------------------------------
function utils.append(tnsr, subtnsr, axis)
  local axis = axis or 1
  if torch.isTensor(tnsr) and tnsr:dim() > 0 then
    tnsr = torch.cat(tnsr, subtnsr, axis)
  else
    tnsr = subtnsr:clone()
    if axis == 1 and (tnsr:dim() == 1 or tnsr:size(1) == tnsr:nElement()) then
      tnsr:resize(1, tnsr:nElement())  -- hack to get desired shape
    end
  end
  return tnsr
end

--------------------------------
--     Remove slices from tensor
--------------------------------
function utils.remove(tnsr, idx, axis)
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
function utils.steal(res, src, idx, axis_r, axis_s)
  local res, src, idx = res, src, idx

  local axis_r = axis_r or 1
  local axis_s = axis_s or 1

  if idx:dim() > 1 then 
    idx = idx:resize(idx:nElement()) 
  end
  
  -------- Append slices to res
  res = utils.append(res, src:index(axis_s, idx), axis_r)

  -------- Remove slices from src
  src = utils.remove(src, idx, axis_s)

  collectgarbage()
  return res, src
end

--------------------------------
--  Wrapper for tensor.indexCopy
--------------------------------
function utils.indexCopy(res, src, idx, axis_r, axis_s)
  local axis_r = axis_r or res:dim()
  local axis_s = axis_s or src:dim()
  return res:indexCopy(axis_r, idx, src:index(axis_s, idx))
end

--------------------------------
--             Kronecker Product
--------------------------------
function utils.kron(X, Z, buffer)
  assert(X:dim() == 2 and Z:dim() == 2) -- should generalize this
  local N, M = X:size(1), X:size(2)
  local P, Q = Z:size(1), Z:size(2)
  local K    = buffer or torch.Tensor(N*P, M*Q)
  for row = 1,N do
    for col = 1,M do
      K[{{(row - 1)*P + 1, row*P},{(col - 1)*Q + 1, col*Q}}]
          = torch.mul(Z, X[row][col])
    end
  end
  return K
end

--------------------------------
--               Modulo Operator
--------------------------------
function utils.modulus(val, base)
  local typ = val:type()
  return torch.add(val:double(), -base, 
    torch.floor(torch.mul(val:double(), 1/base))):type(typ)
end

--------------------------------
--                     Factorial
--------------------------------
function utils.factorial(x)
  local res = utils.as_val(x)
  local val = res - 1
  while val > 0 do
    res = res * val
  end
  return res
end

--------------------------------
--              Tensor Transpose
--------------------------------
function  utils.transpose(tnsr, inplace)
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
function utils.flip(tnsr, axis, idx0, idx1, inplace)
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
function utils.linear_index(shape, coords, order)
  local nDim   = shape:nElement()
  local order  = order or 'C'

  local coords = coords
  if (coords:dim() == 1) then
    coords = coords:reshape(coords:nElement(), 1)
  end

  local offset
  if order == 'C' then
    offset = utils.flip(torch.cat(torch.ones(1, 'torch.LongTensor'),
                        utils.flip(shape, 1, 2):cumprod(), 1))
  elseif order == 'F' then
    offset = torch.cat(torch.ones(1, 'torch.LongTensor'), shape:sub(1, nDim-1), 1):cumprod()
  end
  return torch.mv(coords - 1, offset):add(1)
end

--------------------------------
--   Linear indices of mat diag
--------------------------------
function utils.diag_indices(N)
  return torch.range(1, N^2, N+1, 'torch.LongTensor')
end

--------------------------------
--   Linear indices of lower tri
--------------------------------
function utils.tril_indices(N)
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
function utils.triu_indices(N)
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
function utils.vect(tnsr, order, inplace)
  local N, nDim = tnsr:nElement(), tnsr:dim()
  local order   = order or 'F'
  local shape   = utils.shape(tnsr)

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
    utils.transpose(res, true)
    res:resize(N,1)
  end

  collectgarbage()
  return res
end

--------------------------------
--             Pairwise Distance
--------------------------------
---- Can this be sped up / improved upon?
function utils.pdist(X, Z, p, lenscale, w_root)
  local p    = p or 2
  local dist = nil

  -------- Compute pairwise lp distance (w/o root)
  if lenscale then
    ---- Inverse Lengthscales
    local inv_ls = torch.ones(lenscale:size()):cdiv(lenscale)
    if (inv_ls:dim() == 1) then
      inv_ls:resize(inv_ls:size(1), 1)
    end
    if Z then
      local M, N = X:size(1), Z:size(1)
      local X_ss = torch.mm(X:clone():pow(p), inv_ls):repeatTensor(1, N)
      local Z_ss = torch.mm(Z:clone():pow(p), inv_ls):repeatTensor(1, M)
      dist = Z:clone():t()
      dist:cmul(inv_ls:expandAs(dist))
      dist = torch.zeros(M, N):mm(X, dist):mul(-2.0):add(X_ss):add(Z_ss:t())
    else
      local N    = X:size(1)
      local X_ss = torch.mm(X:clone():pow(p), inv_ls):repeatTensor(1, N)
      dist = X:clone():t()
      dist:cmul(inv_ls:expandAs(dist))
      dist = torch.zeros(N, N):mm(X, dist):mul(-2.0):add(X_ss):add(X_ss:t())
    end
  else
    if Z then
      local M, N = X:size(1), Z:size(1)
      local X_ss = torch.sum(X:clone():pow(p), 2):repeatTensor(1, N)
      local Z_ss = torch.sum(Z:clone():pow(p), 2):repeatTensor(1, M)
      dist = torch.zeros(M, N):mm(X, Z:t()):mul(-2.0):add(X_ss):add(Z_ss:t())
    else
      local N    = X:size(1)
      local X_ss = torch.sum(X:clone():pow(p), 2):repeatTensor(1, N)
      dist = torch.zeros(N, N):mm(X, X:t()):mul(-2.0):add(X_ss):add(X_ss:t())
    end
  end

  -------- Restrict to be non-negative (numerical stability hack)
  dist:clamp(0.0, math.huge)
  collectgarbage()

  if w_root then return dist:pow(1.0/p)
  else           return dist end
end

--------------------------------
--           Tensor NaN Operator
--------------------------------
function utils.nanop(op, tnsr, axis, res)
  local axis = axis or 0

  -------- Special Case: Operate over all axis
  if axis == 0 then
    tnsr = tnsr:reshape(tnsr:nElement())
  end
  local nDims = tnsr:dim()

  -------- Recurse down to 1d case
  if nDims > 1 then
    -------- Swap axis to with last axis
    local tnsr  = tnsr:transpose(axis, nDims)
    local shape = tnsr:size(); shape[nDims] = 1
    local res   = res or torch.Tensor(shape)
    for k = 1, shape[1] do
      res[k] = utils.nanop(op, tnsr:select(1, k), nDims-1, res[k])
    end
    ------ Reshape tensor
    if res:dim() == 1 and res:nElement() == 1 then
      return res:storage()[1]
    else
      return res:transpose(axis, nDims)
    end
  end

  -------- Base case
  return op(tnsr:index(1, tnsr:eq(tnsr):nonzero():squeeze()))
end


--------------------------------
--               Jitter Cholesky
--------------------------------
function utils.chol(src, uplo, config, res)
  ---- Local Variables
  local uplo = uplo or 'L'
  local res  = res or torch.Tensor(src:size()):typeAs(src)
  
  local chol = function(src)
    torch.potrf(res, src, uplo)
  end

  local status, err = pcall(chol, src)

  ---- Jitter Routine
  if status == false then
    local config  = config or {}
    local verbose = config.verbose or true
    local max_eps = config.max_eps or src:norm()
    local eps     = config.eps or 1e-8
    local growth  = config.growth or 1.1
    local I, itr  = torch.eye(src:size(1)), 0
    local mask    = I:byte()

    while(status == false) do
      itr = itr + 1
      if eps > max_eps then
        I:maskedFill(mask, 1.0)
        status, err = pcall(chol, I)
      else
        eps = eps * growth
        I:maskedFill(mask, eps)
        status, err = pcall(chol, torch.add(src, I))
      end
    end

    if verbose then
      local msg
      if eps > max_eps then
        msg = 'Warning: utils.chol failed to find a PSD version\n'..
              'of the input matrix; returning chol(I).'
      else
        msg = string.format('Warning: utils.chol succeeded '..
              'in factorizing the\ninput matrix after applying '..
              'a jitter of %.2e', eps)
      end
      print(msg)
    end
  end
  return res
end

--------------------------------
--             Tensor Covariance
--------------------------------
function utils.cov(X, axis)
  local nDims = X:dim()
  assert(nDims == 2) -- temp hack
  local axis = axis or 2
  local dim  = 1 + (axis % 2)
  local cov  = X - X:mean(dim):expandAs(X)
  if (axis == 2) then
    cov = torch.mm(cov:t(), cov)
  else
    cov = torch.mm(cov, cov:t())
  end
  return cov:div(X:size(dim))
end

--------------------------------
--       Tensor Cross Covariance
--------------------------------
function utils.cross_cov(X, Z, axis)
  if not Z then
    return utils.cov(X, axis)
  else
    assert(X:dim() == 2) -- temp hack
    assert(Z:dim() == 2)
    local axis = axis or 2
    local dim  = 1 + (axis % 2)
    local cov
    if (axis == 2) then
      cov = torch.mm((X - X:mean(dim):expandAs(X)):t(), Z - Z:mean(dim):expandAs(Z))
     else
      cov = torch.mm(X - X:mean(dim):expandAs(X), (Z - Z:mean(dim):expandAs(Z)):t())
    end
  end
  return cov:div(X:size(dim))
end

--------------------------------
--      Error Function (Approx.)
--------------------------------
function utils.erf(x)
   -------- Constants
  local c1, c2 = 0.254829592, -0.284496736
  local c3, c4 = 1.421413741, -1.453152027
  local c5, p  = 1.061405429,  0.3275911

  -------- Argument Parsing
  if not torch.isTensor(x) then
    local t = type(x)
    if t == 'table' then
      x = torch.Tensor(x)
    elseif t == 'number' then
      x = torch.Tensor{x}
    end
  end

  -------- Error Function
  local sign = x:ge(0.0):type(x:type()):mul(2):add(-1)
  local x    = torch.abs(x)
  local t    = x:clone():mul(p):add(1):pow(-1)
  local erf  = t:clone():mul(c5):add(c4):cmul(t):add(c3)
                :cmul(t):add(c2):cmul(t):add(c1):cmul(t)
                :cmul(x:pow(2):mul(-1.0):exp())
                :mul(-1):add(1):cmul(sign)
  return erf
end

--------------------------------
--           Standard Normal PDF
--------------------------------
function utils.norm_pdf(x)
  return torch.exp(torch.pow(x, 2):mul(-0.5)):mul(sqrt2pi_inv)
end

--------------------------------
--           Standard Normal CDF
--------------------------------
function utils.norm_cdf(x)
  return utils.erf(torch.mul(x, sqrt2_inv)):add(1):mul(0.5)
end

--------------------------------
--       Standard Normal Log-PDF
--------------------------------
function utils.norm_logpdf(x)
  return torch.pow(x, 2):add(log2pi):mul(-0.5)
end

--------------------------------
--                Log-Normal PDF
--------------------------------
function utils.lognorm_pdf(x)
  return torch.exp(torch.log(x):pow(2):mul(-0.5)):cdiv(x):mul(sqrt2pi_inv)
end

--------------------------------
--                Log-Normal CDF
--------------------------------
function utils.lognorm_cdf(x)
  return utils.erf(torch.log(x):mul(sqrt2_inv)):mul(0.5):add(0.5)
end

--------------------------------
--            Log-Normal Log-PDF
--------------------------------
function utils.lognorm_logpdf(x)
  return torch.log(x):pow(2):mul(-0.5):add(x:clone():mul(-sqrt2pi_inv))
end

return utils