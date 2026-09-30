# Terminology and frozen internal symbols

The manuscript uses **restoration output**, **restoration-output bank**,
**synthesis anchor**, **post-synthesis correction evidence**, and **OMRR**.

Some frozen source files retain historical internal symbol names such as
`ContributorBank` and `StateObjectiveReliabilityRouterV1`. They are preserved
to maintain checkpoint/source provenance. In the formal paper path:

- a "contributor" in these internal symbols means a complete restoration output;
- `router_mode="direct_objective"` is used;
- the first six learned-branch channels are zeros, so the learned reliability
  branch receives `[0, 0, V_k]`, not image state;
- the current paper does not make a contributor-conflict or manifold claim.

Do not interpret historical class names as manuscript claims.
