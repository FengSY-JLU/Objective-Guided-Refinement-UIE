# Public-release sanitization

The public repository preserves scientific sample membership, image hashes,
frozen model weights, and formal method parameters while removing machine-local
filesystem locations.

Specifically:
- UIEBD split CSV `input_path` and `target_path` fields were rewritten from
  server-absolute paths to portable relative paths. Source IDs, split membership,
  image SHA256 values, pair signatures, and duplicate-exclusion records are
  unchanged.
- Machine-local completion cards and preflight logs were omitted because they
  are not required to reproduce the released method.
- Path-only metadata in the released source-wise mechanism report was replaced
  by repository-relative references.
- The checksum-frozen formal multidataset trainer is retained byte-for-byte for
  provenance. It contains legacy development-machine default roots, but the
  public `tools/train.py` wrapper overrides those roots before training.

No learned tensor, objective definition, routing rule, or paper-facing
evaluation value was modified by this sanitization.
