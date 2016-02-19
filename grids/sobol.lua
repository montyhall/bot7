------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Sobol sequence grid class. Implementation of
Bennett Fox's FORTRAN77 i4_sobol methods.

For further information, please see
http://people.sc.fsu.edu/~jburkardt/py_src/
sobol/sobol.html

Authored: 2015-10-02 (jwilson)
Modified: 2016-02-24
--]]

---------------- External Dependencies
local math  = require('math')
local utils = require('bot7.utils.bits')

------------------------------------------------
--                                         sobol
------------------------------------------------
local title  = 'bot7.grids.sobol'
local parent = 'bot7.grids.abstract'
local grid, parent = torch.class(title, parent)

function grid:__init(config)
  parent.__init(self)

  local C = config or {}
  C.max_dims    = C.max_dims or 40
  C.log_max     = C.log_max or 30
  C.bit_precis  = C.bit_precis or 32
  C.initialized = False
  assert(C.size)
  assert(C.dims and C.dims < C.max_dims)
  self.config   = C

  self.seed     = -1
  self.recipd   = nil
  self.includ   = nil
  self.atmost   = nil
  self.maxcol   = nil
  self.lastq    = nil

  self.poly = torch.Tensor
  {
    001, 003, 007, 011, 013, 019, 025, 037, 059, 047,
    061, 055, 041, 067, 097, 091, 109, 103, 115, 131,
    193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
    213, 191, 253, 203, 211, 239, 247, 285, 369, 299,
  }

  self.bank = torch.Tensor()
  self:create_bank()
end

function grid:generate(config)
--[[
Generator function for Sobol sequences

Parameters:
  config.dims, integer, the spatial dimension.
  config.size, integer, the number of points to generate.
  config.skip, integer, the number of initial points to skip.

Output: real R(config.size, config.dims), the points.
--]]
  local config = config or self.config
  local skip = config.skip or 1
  local grid = torch.Tensor(config.size, config.dims):fill(0/0)
  local seed = nil

  for j = 1, config.size do
    seed = j + skip - 1
    grid[{{j},{}}], seed = self:i4_sobol(config.dims, seed)
  end

  if config.mins and config.maxes then
    grid:cmul(grid, torch.add(config.maxes, -config.mins):expandAs(grid))
        :add(config.mins:expandAs(grid))
  elseif config.mins then
    grid:add(torch.add(config.mins, grid:min(1)[1]):expandAs(grid))
  elseif config.maxes then 
    grid:cmul(torch.cdiv(config.maxes, grid:max(1)[1]):expandAs(grid))
  end

  collectgarbage()
  return grid
end

function grid:i4_bit_hi1(n)
--[[
i4_bit_hi1 returns the position of the high
1  bit base 2 in an integer.

Example:
  +------+-------------+-----
  |    N |      Binary | BIT
  +------|-------------+-----
  |    0 |           0 |   0
  |    1 |           1 |   1
  |    2 |          10 |   2
  |    3 |          11 |   2
  |    4 |         100 |   3
  |    5 |         101 |   3
  |    6 |         110 |   3
  |    7 |         111 |   3
  |    8 |        1000 |   4
  |    9 |        1001 |   4
  |   10 |        1010 |   4
  |   11 |        1011 |   4
  |   12 |        1100 |   4
  |   13 |        1101 |   4
  |   14 |        1110 |   4
  |   15 |        1111 |   4
  |   16 |       10000 |   5
  |   17 |       10001 |   5
  | 1023 |  1111111111 |  10
  | 1024 | 10000000000 |  11
  | 1025 | 10000000001 |  11

Parameters:
  Input, integer N, the integer to be
  measured. N should be nonnegative. 
  If N is nonpositive, the value will 
  always be 0.

  Output, integer BIT, the number of
  bits base 2.
--]]

  local i, bit = math.floor(n), 0
  while i > 0 do
    bit = bit + 1
    i   = math.floor(i/2)
  end
  return bit
end

function grid:i4_bit_lo0(n)
--[[
I4_BIT_LO0 returns the position of the low
0 bit base 2 in an integer.

Example:
  +------+------------+----
  |    N |     Binary | BIT
  +------+------------+----
  |    0 |          0 |   1
  |    1 |          1 |   2
  |    2 |         10 |   1
  |    3 |         11 |   3
  |    4 |        100 |   1
  |    5 |        101 |   2
  |    6 |        110 |   1
  |    7 |        111 |   4
  |    8 |       1000 |   1
  |    9 |       1001 |   2
  |   10 |       1010 |   1
  |   11 |       1011 |   3
  |   12 |       1100 |   1
  |   13 |       1101 |   2
  |   14 |       1110 |   1
  |   15 |       1111 |   5
  |   16 |      10000 |   1
  |   17 |      10001 |   2
  | 1023 | 1111111111 |   1
  | 1024 | 0000000000 |   1
  | 1025 | 0000000001 |   1

Parameters:
  Input, integer N, the integer to be
  measured. N should be nonnegative.

  Output, integer BIT, the position 
  of the low 1 bit.
]]--
    
  local bit = 1  
  local i   = math.floor(n)
  local i2  = math.floor(i/2)
  while (i ~= 2*i2) do
    bit = bit + 1
    i   = i2
    i2  = math.floor(i/2)
  end
  return bit
end


function grid:i4_sobol_generate(dims, n, skip)
--[[
i4_sobol_generate generates a Sobol dataset.

Parameters:
  Input, integer dims, the spatial dimension.
  Input, integer N, the number of points to generate.
  Input, integer SKIP, the number of initial points to skip.

Output: real R(M,N), the points.
--]]

  local skip = skip or 1
  local grid = torch.Tensor(n, dims):fill(0/0)
  local seed, next_seed = nil, nil

  for j = 1, n do
    seed = j + skip - 1
    grid[{{j},{}}], seed = self:i4_sobol(dims, seed)
  end

  return grid
end

function grid:i4_sobol(dims, seed)
--[[
    i4_sobol generates a new quasirandom Sobol vector with each call.

    Parameters:
      Input, integer DIM_NUM, the number of spatial dimensions.
      DIM_NUM must satisfy 1 <= DIM_NUM <= 40.
      Input/output, integer SEED, the "seed" for the sequence.
      This is essentially the index in the sequence of the quasirandom
      value to be generated.  On output, SEED has been set to the
      appropriate next value, usually simply SEED+1.
      If SEED is less than 0 on input, it is treated as though it were 0.
      An input value of 0 requests the first (0-th) element of the sequence.

      Output, real QUASI(DIM_NUM), the next quasirandom vector.
--]]
  local config  = self.config
  config.atmost = 2^config.log_max - 1 -- largest int val
  config.maxcol = self:i4_bit_hi1(config.atmost) -- Find the number of bits in ATMOST.

  -- Initialize
  if not config.initialized then
    config.initialized = true
    self.bank[{{1},{1, config.maxcol}}]:fill(1)
    config.dims = nil -- clear config.dims to ensure initialization occurs
  end

  if dims ~= config.dims then -- Things to do only if the dimension changed.
    config.dims = dims

    -- Initialize the remaining rows of self.bank for i in range(2, dims):
    local j, m, j2, v, l
    for i = 1, dims do
      -- The bits of the integer POLY(I) gives the form of polynomial I.
      -- Find the degree of polynomial I from binary encoding.
      j, m = math.floor(self.poly[i]/2), 0
      while j > 0 do
        m = m + 1
        j = math.floor(j/2)
      end

      -- Expand this bit pattern to separate components of the 
      -- logical array INCLUD.
      j, self.includ = self.poly[i], torch.zeros(m)
      for k = m, 1, -1 do
        j2 = math.floor(j/2)
        if j ~= 2*j2 then self.includ[k] = 1 end
        j = j2
      end

      -- Calculate the remaining elements of row I as explained
      -- in Bratley and Fox, section 2.
      for j = m+1, config.maxcol do
        v, l = self.bank[{i, j-m}], 1
        for k = 1, m do
          l = 2 * l
          if self.includ[k] == 1 then
            v = utils.bitwise_xor(v, l*self.bank[{i, j-k}])
          end
        end
        self.bank[{i, j}] = v
      end
    end

    -- Multiply columns of bank by appropriate power of 2.
    l = 1
    for j = config.maxcol-1, 1, -1 do
      l = l * 2
      self.bank[{{1, dims}, {j}}]:mul(l)
    end

    self.recipd = 0.5/l -- 1/(common denominator of the elements in bank)
    self.lastq  = torch.zeros(dims)
  end

  seed = math.max(0, math.floor(seed)) -- seed should be of type int?

  if seed == 0 then
    l, self.lastq = 1, torch.zeros(dims)
  elseif seed == self.seed + 1 then
    l = self:i4_bit_lo0(seed)
  elseif seed <= self.seed then
    self.seed, l, self.lastq = 0, 1, torch.zeros(dims)
    for seed_temp = self.seed, seed-1 do
      l = self:i4_bit_lo0(seed_temp)
      for i = 1, dims do
        self.lastq[i] = utils.bitwise_xor(self.lastq[i], self.bank[{i, l}])
      end
    end
    l = self:i4_bit_lo0(seed)

  elseif self.seed + 1 < seed then
    for seed_temp = self.seed+1, seed-1 do
      l = self:i4_bit_lo0(seed_temp)
      for i = 1, dims do
        self.lastq[i] = utils.bitwise_xor(self.lastq[i], self.bank[{i, l}])
      end
    end
    l = self:i4_bit_lo0(seed)
  end

  -- Check that the user is not calling too many times!
  if config.maxcol < l then
    print('I4_SOBOL - Fatal error!')
    print('  Too many calls!')
    print(string.format('  MAXCOL = %d\n', config.maxcol))
    print(string.format('  L =      %d\n', l))
    return
  end

  -- Calculate the new components of QUASI.
  quasi = torch.zeros(dims)
  for i = 1, dims do
    quasi[i] = self.lastq[i] * self.recipd
    self.lastq[i] = utils.bitwise_xor(self.lastq[i], self.bank[{i, l}])
  end
  self.seed = seed
  seed = seed + 1
  return quasi, seed
end


function grid:create_bank()
  local bank = self.bank:resize(self.config.max_dims,
                    self.config.log_max)

  bank:select(2, 1):fill(1.0)

  bank[{{3, 40}, 2}] = torch.Tensor
  {
          1, 3, 1, 3, 1, 3, 3, 1,
    3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
    1, 3, 1, 3, 3, 1, 3, 1, 3, 1,
    3, 1, 1, 3, 1, 3, 1, 3, 1, 3,
  }

  bank[{{4, 40}, 3}] = torch.Tensor
  {
             7, 5, 1, 3, 3, 7, 5,
    5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
    5, 3, 3, 1, 7, 5, 1, 3, 3, 7,
    5, 1, 1, 5, 7, 7, 5, 1, 3, 3,
  }

  bank[{{6, 40}, 4}]= torch.Tensor
  {
                        01, 07, 09, 13, 11,
    01, 03, 07, 09, 05, 13, 13, 11, 03, 15,
    05, 03, 15, 07, 09, 13, 09, 01, 11, 07,
    05, 15, 01, 15, 11, 05, 03, 01, 07, 09,
  }

  bank[{{8, 40}, 5}] = torch.Tensor
  {
                                09, 03, 27,
    15, 29, 21, 23, 19, 11, 25, 07, 13, 17,
    01, 25, 29, 03, 31, 11, 05, 23, 27, 19,
    21, 05, 01, 17, 13, 07, 15, 09, 31, 09,
  }

  bank[{{14, 40}, 6}] = torch.Tensor
  {
                37, 33, 07, 05, 11, 39, 63,
    27, 17, 15, 23, 29, 03, 21, 13, 31, 25,
    09, 49, 33, 19, 29, 11, 19, 27, 15, 25,
  }

  bank[{{20, 40}, 7}] = torch.Tensor
  {
                                                 013,
    033, 115, 041, 079, 017, 029, 119, 075, 073, 105,
    007, 059, 065, 021, 003, 113, 061, 089, 045, 107,
  }

  bank[{{38, 40}, 8}] = torch.Tensor{7, 23, 39}
end
