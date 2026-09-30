# Dataset layout

The repository does not redistribute benchmark images.

For inference, `--input-dir` may be any directory of RGB PNG/JPG/BMP images.

For formal retraining of UFO120/EUVP/RUIE/OceanDark, `--data-root` should point
to the directory containing the source images named by the released manifest.
The formal self-supervised trainer uses the same source image as input and
training target; no clean/GT image is used for optimization.

UIEBD split manifests are released under `protocols/splits/uiebd/`.
