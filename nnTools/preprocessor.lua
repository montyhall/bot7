------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Convenience module for application of common 
data preprocessing techniques.


Expects data to be passed in as:
  ------------------------------
  | Field | Content            |
  ------------------------------
  |  xr   | Training Inputs    |
  |  yr   | Training Targets   |
  |  xe   | Test Inputs        |
  |  ye   | Test Targets       |
  |  xv   | Validation Inputs  |
  |  yv   | Validation Targets |
  ------------------------------

Authored: 2015-11-12 (jwilson)
Modified: 2015-11-12
--]]

---------------- External Dependencies
local nn = require('nn')
local image = require('image')

------------------------------------------------
--                                  preprocessor
------------------------------------------------
local preprocessor = function(data)

  local suffix = {'r', 'e', 'v'}
  local counts = {r=0, e=0, v=0}
  local normalizer = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  local set, yuv, mu, sd

  for enum = 1, #suffix do
    set = suffix[enum]

    if data['x'..set] and data['y'..set] then
      counts[set] = data['x'..set]:size(1)
      assert(counts[set] == data['y'..set]:size(1))
    end

    -- Need training data in order to calculate statistics
    if enum == 1 then assert(counts[set] > 0) end

    if counts[set] > 0 then
      for idx = 1, counts[set] do
        xlua.progress(idx, counts[set])
        yuv    = image.rgb2yuv(data['x'..set][idx])
        yuv[1] = normalizer(yuv[{{1}}]) -- normalize 'y' locally
        data['x'..set][idx] = yuv
      end
    end

    -- Need training data to calculate statistics
    if enum == 1 then 
      assert(counts[set] > 0)
      mu = {u = data['x'..set]:select(2,2):mean(),
            v = data['x'..set]:select(2,3):mean()}
      sd = {u = data['x'..set]:select(2,2):std(),
            v = data['x'..set]:select(2,3):std()}
    end

    -- Normalize 'u' and 'v' globally
    if counts[set] > 0 then
      data['x'..set]:select(2,2):add(-mu.u);
      data['x'..set]:select(2,2):div(sd.u);
      data['x'..set]:select(2,3):add(-mu.v);
      data['x'..set]:select(2,3):div(sd.v);
    end
  end
end

return preprocessor