# External Restormer prior asset

The formal restoration-output banks use a frozen Restormer prior named
`best_restormer_prior_woruie.pth`.

This large prior checkpoint is **not included in the GitHub repository**.
Place it at:

`assets/restormer/best_restormer_prior_woruie.pth`

Expected SHA256:

`b2f6fa945708afff4ba58a1f0a3c4aab95965ef6086d646b3a5d4d6f73f4486c`

The architecture file used by the research code is retained at
`src/model/restormer_arch.py` and originates from the Restormer project:

https://github.com/swz30/Restormer

Upstream license:

https://github.com/swz30/Restormer/blob/main/LICENSE.md

Download the exact frozen prior checkpoint from:

https://drive.google.com/file/d/1Blm76DUeL-MAgOYJzUYlvgzwBBVThNTE/view?usp=drive_link

After downloading, place it at:

`assets/restormer/best_restormer_prior_woruie.pth`

Verify the SHA256 before use. The GitHub repository does not redistribute this large checkpoint.
