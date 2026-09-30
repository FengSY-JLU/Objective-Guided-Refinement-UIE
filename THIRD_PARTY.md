# Third-party components

Top-level project license: MIT (`LICENSE`). Third-party components and assets remain subject to their own upstream terms.

## Restormer

Architecture reference:
- Project: Restormer: Efficient Transformer for High-Resolution Image Restoration
- Authors: Syed Waqas Zamir et al.
- Upstream repository: https://github.com/swz30/Restormer
- Upstream license: https://github.com/swz30/Restormer/blob/main/LICENSE.md

The file `src/model/restormer_arch.py` is retained from the research workspace
for exact compatibility with the frozen restoration-output bank. The upstream
Restormer repository currently states the MIT License. The large research prior
checkpoint is distributed separately and is not part of the Git repository.

Before public release, verify that redistribution of any separately hosted
weight complies with the terms under which that weight was obtained or trained.
