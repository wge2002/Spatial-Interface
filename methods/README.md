# Method identity

`si-r1` is the initial Spatial Interface release. It preserves the imported
DG-r5 robot method while recording the namespace, portable runtime and startup
changes as a new source identity. `si-codex-r1` and `si-qwen-r1` identify client
source separately; model, provider, effort, task, seed and budgets are recorded
in each experiment manifest.

The initial registry pins SHA256 contents and the repository's **unique root
commit**. This permits a single initial Git commit without a self-referential
commit hash in a file. At runtime the root resolves to its full immutable commit
hash, the original registry is read from that commit, and every declared file
and LIBERO dependency is checked. Snapshots also record the actual full HEAD.
Dirty method files, a modified initial registry, unknown IDs and mismatched
dependencies are rejected. Existing snapshot files are not overwritten.

```bash
python scripts/methods.py list
python scripts/methods.py validate
python scripts/methods.py snapshot --release si-r1 --profile si-codex-r1 \
  --out data/identity/example.json
```

An identity sidecar supplements the complete experiment manifest. It does not
prove that an experiment ran or that an untested interface/model combination
works. The initial tool supports the root release only: when changing method
behavior, add a new release and an explicit commit-anchor implementation before
launching a new condition. Never edit si-r1 to silently follow new source.
Do not recreate old VIA method tags on new commits or rewrite historical results.
