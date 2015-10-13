[![Join the chat at https://gitter.im/torch/torch7](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/j-wilson/bot7?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

__Notice__: Please be sure to update your copy of `Torch7` prior to work with `bot7`.

<a name="bot7.intro.dok"/>
# bot7 #
__bot7__ is a framework for Bayesian Optimization implemented in Torch7. For it's companion package on Gaussian Processes, please refer to [gpTorch7](https://github.com/j-wilson/gpTorch7). This package is currently still in an alpha-build stage; so, please feel free to pass along suggestions and/or feedback.
<a name="bot7.content.dok"/>
## Package Content ##

Directory    | Content 
:-------------:|:----------------------
bots       | Automated experiment runners
models     | Target function models
scores     | Acquisition functions (incl. EI, UCB)
grids      | Candidate grids
samplers   | Sampling methods (e.g. slice sampling)
examples   | Demos / Benchmark Functions
<a name="bot7.dev.dok"/>

## Installation and Usage ##

You can install `bot7` via `luarocks` provided by the `torch` distribution.
```bash
git clone https://github.com/j-wilson/bot7
cd bot7
luarocks make
```

After installing `bot7`, you can load the package inside Torch via:
```lua
require('bot7');
th> bot7.
bot7.grids.       bot7.models.      bot7.samplers.    
bot7.bots.        bot7.scores.      bot7.utils.
```

## Developers' Notes ##
Further documentation coming soon.
