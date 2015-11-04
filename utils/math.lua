------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Math utility method for bot7.


Authored: 2015-10-30 (jwilson)
Modified: 2015-11-04
--]]

---------------- Constants
local sqrt2_inv   = 1/math.sqrt(2)
local sqrt2pi     = math.sqrt(2*math.pi)
local sqrt2pi_inv = 1/math.sqrt(2*math.pi)
local log2pi      = math.log(2*math.pi)

------------------------------------------------
--                                          math
------------------------------------------------
local utils = require('bot7.utils.tensor')
local self  = {}

--------------------------------
--             Kronecker Product
--------------------------------
function self.kron(X, Z, buffer)
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
function self.modulus(val, base)
  local typ = val:type()
  return torch.add(val:double(), -base, 
    torch.floor(torch.mul(val:double(), 1/base))):type(typ)
end

--------------------------------
--                     Factorial
--------------------------------
function self.factorial(x)
  local res = utils.number(x)
  local val = res - 1
  while val > 0 do
    res = res * val
  end
  return res
end

--------------------------------
--             Pairwise Distance
--------------------------------
function self.pdist(X, Z, lenscale, w_root)
  local p    = 2
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
      local X_ss = torch.mm(torch.pow(X,p ), inv_ls):repeatTensor(1, N)
      dist = X:clone():t()
      dist:cmul(inv_ls:expandAs(dist))
      dist = torch.zeros(N, N):mm(X, dist):mul(-2.0):add(X_ss):add(X_ss:t())
    end
  else
    if Z then
      local M, N = X:size(1), Z:size(1)
      local X_ss = torch.sum(torch.pow(X, p), 2):repeatTensor(1, N)
      local Z_ss = torch.sum(torch.pow(Z, p), 2):repeatTensor(1, M)
      dist = torch.zeros(M, N):mm(X, Z:t()):mul(-2.0):add(X_ss):add(Z_ss:t())
    else
      local N    = X:size(1)
      local X_ss = torch.sum(torch.pow(X, p), 2):repeatTensor(1, N)
      dist = torch.zeros(N, N):mm(X, X:t()):mul(-2.0):add(X_ss):add(X_ss:t())
    end
  end

  if p == 1 then dist:abs() end

  -------- Restrict to be non-negative (numerical stability hack)
  dist:clamp(0.0, math.huge)
  collectgarbage()

  if w_root then return dist:pow(1.0/p)
  else           return dist end
end

--------------------------------
--           Tensor NaN Operator
--------------------------------
function self.nanop(op, tnsr, axis, res)
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
      res[k] = self.nanop(op, tnsr:select(1, k), nDims-1, res[k])
    end
    ------ Reshape tensor
    if res:dim() == 1 and res:nElement() == 1 then
      return res:storage()[1]
    else
      return res:transpose(axis, nDims)
    end
  end

  
  -------- Base case
  local idx = tnsr:eq(tnsr):nonzero()
  local dim = idx:dim()
  if dim == 0 then
    return torch.Tensor(1):fill(0/0)
  elseif dim > 1 then
    idx:resize(idx:nElement())
  end
  return op(tnsr:index(1, idx))
end

--------------------------------
--               Jitter Cholesky
--------------------------------
function self.chol(src, uplo, config, res)
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
    local verbose = config.verbose or 1
    local max_eps = config.max_eps or src:norm()
    local eps     = config.eps or 1e-8
    local growth  = config.growth or 1.1
    local I, itr  = torch.eye(src:size(1)), 0
    local mask    = I:byte()
    local cleanup_freq = config.cleanup_freq or 1e2
    local msg_freq     = config.msg_freq or 1e2

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

      ---- Report status to user
      if verbose > 0 and (itr-1) % msg_freq then
        print(string.format('Iteration %d of jitter routine', itr))
      end

      ---- Cleanup periodically
      if itr % cleanup_freq == 0 then
        collectgarbage()
      end
    end

    if verbose > 0 then
      local msg
      if eps > max_eps then
        msg = 'Warning: utils.math.chol failed to find a PSD version\n'..
              'of the input matrix; returning chol(I).'
      else
        msg = string.format('Warning: utils.math.chol succeeded '..
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
function self.cov(X, axis)
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
function self.cross_cov(X, Z, axis)
  if not Z then
    return self.cov(X, axis)
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
function self.erf(x)
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
function self.norm_pdf(x)
  return torch.exp(torch.pow(x, 2):mul(-0.5)):mul(sqrt2pi_inv)
end

--------------------------------
--           Standard Normal CDF
--------------------------------
function self.norm_cdf(x)
  return self.erf(torch.mul(x, sqrt2_inv)):add(1):mul(0.5)
end

--------------------------------
--       Standard Normal Log-PDF
--------------------------------
function self.norm_logpdf(x)
  return torch.pow(x, 2):add(log2pi):mul(-0.5)
end

--------------------------------
--                Log-Normal PDF
--------------------------------
function self.lognorm_pdf(x)
  return torch.exp(torch.log(x):pow(2):mul(-0.5)):cdiv(x):mul(sqrt2pi_inv)
end

--------------------------------
--                Log-Normal CDF
--------------------------------
function self.lognorm_cdf(x)
  return self.erf(torch.log(x):mul(sqrt2_inv)):mul(0.5):add(0.5)
end

--------------------------------
--            Log-Normal Log-PDF
--------------------------------
function self.lognorm_logpdf(x)
  return torch.log(x):pow(2):mul(-0.5):add(-torch.mul(x, sqrt2pi):log())
end

return self
