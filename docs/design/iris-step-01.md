# IRIS, step 1: register a development scaffold

This lesson describes the registration changes implemented on `codex/iris`.
Use it to review this branch or reproduce the changes later on your own `iris`
branch. The [reference baseline](iris.md) and its
[harness README](../../scripts/iris/README.md) are already available.

## Before editing

Use this worktree, not the original Dreamer checkout:

```bash
cd /Users/ashutoshpanda/.codex/worktrees/d8dc/mstar
git branch --show-current
source .venv/bin/activate
```

This implementation worktree should be on `codex/iris`. Your original checkout
remains on `dreamer_v3`; an independent learning branch can be named `iris`.
Read the baseline's scope and tensor-contract sections. Then inspect these
upstream files in `.ci-results/iris/reference/src/`:

- `agent.py`, `Agent.act`: see why acting alone does not call the Transformer.
- `models/world_model.py`, `WorldModel.forward`: observation/action embeddings,
  positions, Transformer, and observation/reward/ending heads.
- `envs/world_model_env.py`, `reset_from_initial_observations` and `step`: see
  priming, token generation, decoding, and the context refresh.

Run one short harness example from the README and inspect `initial/tokens` and
`step/0000/cache_size` in its NPZ trace. Explain why the values are 16 tokens
per image and 33 cached tokens after the first predicted frame.

## What step 1 accomplishes

The public name `iris` should resolve to its default deployment YAML and then
to the `IrisModel` class. It does not yet construct a working inference model.

Read these M* paths:

- `mstar/model/registry.py`: `MODEL_REGISTRY`, `HF_MODELS`, `get_model_class`,
  `model_init_kwargs`.
- `mstar/cli/main.py`: `DEFAULT_CONFIGS` and `_resolve_config`.
- `test/modular/test_model_registration.py`: what the registration checks prove.

## Exact edits for you to make

1. Create an empty `mstar/model/iris/__init__.py`.

2. Create `mstar/model/iris/iris_model.py` with an abstract scaffold:

   ```python
   from mstar.model.base import Model


   class IrisModel(Model):
       """IRIS world-model integration scaffold; inference is not implemented yet."""
   ```

   The class stays abstract until we implement its Model hooks. Importing it
   must not load a checkpoint or import the external reference repository.

3. Add this to `MODEL_REGISTRY` in `mstar/model/registry.py`:

   ```python
   "iris": ("mstar.model.iris.iris_model", "IrisModel"),
   ```

   Add this to `HF_MODELS` in the same file:

   ```python
   "iris": {"model_path_hf": "eloialonso/iris"},
   ```

   Unlike our Dreamer baseline, IRIS has a selected published HF artifact.
   This is its repository ID; the exact `pretrained_models/Breakout.pt` path,
   revision and checksum remain pinned in the baseline. Step 4's loader must
   select that file deliberately, rather than expecting a Transformers-format
   `config.json` or downloading every game's weights.

4. Create `configs/iris.yaml` as a draft:

   ```yaml
   # Development scaffold: real inference is not implemented yet.
   model: "iris"
   model_kwargs:
     game: "Breakout"
   node_groups:
     - node_names: [image_encoder, world_model, image_decoder]
       ranks: [0]
   ```

   These are proposed logical component names; the graph will use them when
   implemented. The `game` option will become part of the constructor contract
   in step 2. Do not copy an LLM's resource-capacity settings before we declare
   and size IRIS's actual resources.

5. In `mstar/cli/main.py`, add to `DEFAULT_CONFIGS`:

   ```python
   "iris": "iris.yaml",
   ```

   Read `_next_steps` too, but defer a runnable client snippet until the API
   contract exists. A text-chat example would not describe this workload.

6. In the first model list-table in `docs/models.rst`, add an accurate draft row:

   ```rst
      * - ``iris`` *(Development scaffold)*
        - ``eloialonso/iris`` (``pretrained_models/Breakout.pt``)
        - IRIS Transformer world-model integration under development; inference is not implemented yet.
   ```

Keep the two blank lines before the class and terminate the YAML with a newline.
Do not copy upstream GPL source files into the model package at this step.
The code-reuse approach is a maintainer design decision before the later port.
No new engine code, dependency extra, OpenAI adapter or runtime implementation
is needed for discovery.

## What to learn while implementing

**Python:** module paths, lazy imports, class lookup and abstract base classes.
Registry tuples name code without loading its weights.

**M*:** a registered name selects a Model contract; the graph declares logical
nodes, and YAML places them on hardware. Node names must eventually agree across
the graph, resources, loader dispatch and placement config.

**Systems:** discovery/configuration and computation are separate responsibilities.
Large dependencies and weight allocation should not happen in every process just
because a registry is imported.

## Acceptance check and review

From this worktree's repository root:

```bash
.venv/bin/ruff check .
OMP_NUM_THREADS=2 .venv/bin/python -m pytest test/modular/test_model_registration.py -q
git diff -- mstar/model/registry.py mstar/cli/main.py docs/models.rst
git status --short
```

The registration test checks files, class declarations, CLI defaults and model
documentation. It does not instantiate the abstract class or prove inference
works. Rerun it after reproducing these edits on your learning branch.

Before step 2, explain:

- Why does `Agent.act` alone miss our Transformer-serving learning goal?
- Why does the model registry store strings instead of an imported IrisModel?
- How does the HF repository ID differ from the exact checkpoint file/revision?
- Why can registration pass while the Model class is still abstract?
- Why are node names and GPU ranks different concepts?

Then show your diff for review. A suitable development commit is:
`Add IRIS registration and model scaffold`.

## Implementation acceptance record

On `codex/iris`, the model package, lazy registry entry, HF default, draft YAML,
CLI default and development documentation row are implemented.

- `ruff check .`: passed.
- Registration tests: 4 passed; reference-harness utility tests: 5 passed
  (`OMP_NUM_THREADS=2`, combined run: 9 passed).
- CLI `_resolve_config("iris", None)` and parsing the selected YAML: passed;
  the model key, Breakout option and proposed node placement matched.
- `git diff --check`: passed.

An additional attempt to import the real Model base outside pytest failed
because this macOS environment lacks `triton`, imported by the existing
`mstar/engine/resources/sampler/utils.py`. The modular test environment stubs
optional accelerator dependencies; these passing checks establish registration,
not a production import or server launch. No GPU inference or benchmark was run.
The six registration/scaffold files were committed as `69f05b7f`
(`Add IRIS registration and model scaffold`) after rebasing onto `a9e81ed8`,
which matched the fetched upstream and fork main branches. That commit was pushed
to `origin/codex/iris`. The reference tools and learning notes were included with
the subsequent contract and benchmark-plan commit.
The subsequent [Step 2 contract](iris-step-02.md) extends the original abstract
scaffold; the code examples above describe the Step 1 checkpoint.
