# Changelog

This file lists every change to what a developer can rely on: the calls of the browser runtime,
the names and paths in the folder that `cifarium build` writes, the manifest `cifarium.json`, the
keys of the settings file, the exit codes and `report.json` (ARCHITECTURE.md, section 14).

The rule of compatibility:

- Adding a field, an error code, a reason or a value breaks nothing, and may come in any version.
  Whatever reads cifarium output skips the fields and codes it does not know.
- Removing, renaming or giving a new meaning to anything in the contract breaks someone. While
  the version is 0.x it is done only in a minor version (0.2.0, not 0.1.1) and has a line here
  under "Changed" or "Removed".
- The runtime reads every manifest format up to its own. A build made by an older cifarium keeps
  working with a newer `cifarium.js`; the other way round, the runtime says `unsupported`.

The tests in `tests/test_contracts.py`, `tests/test_bundle.py` and `tests/test_runtime_e2e.py` hold
each contract as a list. Changing one of the lists fails a test, and that change goes with a line
below.

## 0.1.0 (not released yet)

The first contracts. Nothing was published before, so nothing is a break.

- Runtime `cifarium.js`: `createCifarium(manifestUrl)`, `load({ onProgress, device })`,
  `run(text)`, `info()`, `dispose()`; the exports `CifariumError`, `VERSION`, `MANIFEST_FORMAT`,
  `ERROR_CODES`, `FALLBACK_REASONS`, `PROGRESS_PHASES`. The name `signal` is reserved for a later
  version.
- `run` of a classification answers `{ label, score, scores }`; `scores` lists every label, the
  best first.
- `info().fallback` is `{ reason, message }` or `null`; the labels, with their descriptions, are
  in `info()` from `createCifarium` on.
- Errors carry `code` and, when there was one, `cause`. Codes: `manifest`, `unsupported`, `input`,
  `not-loaded`, `network`, `load`, `run`, `disposed`.
- Progress events carry `phase`: `download` or `prepare`.
- Manifest, format 1: every key in camel case, the labels in the manifest, `data` holds the files
  of the recipe.
- The folder: `index.html`, `cifarium.js`, `cifarium-worker.js`, `cifarium.json`, `LICENSES.txt`,
  `vendor/`, `models/`, `data/`, `hosting/`. Everything in it may be published: no file holds the
  text of an example. The same input gives the same bytes.
- Headers go on `/`, `/index.html`, `/cifarium-worker.js` and `/vendor/*`; `hosting/_headers` is
  read by Netlify and by Cloudflare Pages.
- Settings (`cifarium.yaml`): the keys `task`, `examples`, `labels`, `pass_rate`, `limits`
  (`max_size_mb`, `max_latency_ms`), `license`, `host`, `output`. `license` is `permissive`
  (Apache-2.0, MIT, BSD: the default) or `any`; the value `commercial` of the development builds is
  refused with a hint to write `permissive`, because Gemma allows commercial use under terms of its
  own and `commercial` hid it for that reason. The keys of tasks that are not released
  (`documents`, `similarity`, `limits.min_tokens_per_sec`) answer "reserved in this version" whatever
  their value. The defaults are part of the contract: `pass_rate` 0.85, `limits.max_size_mb` 300,
  no `limits.max_latency_ms`, `license` permissive, `host` bundle, `output` `./cifarium-out`.
- Exit codes of `build` and `init`: 0 done, 1 no model fits, 2 the settings, the examples or the
  command line are wrong (`init`: a file is already there), 3 the environment failed (Chrome, the
  network, the cache, the disk, the installation; `init`: the folder cannot be written; also an
  error cifarium did not expect, with its traceback). `catalog-update` keeps its own codes.
- One weight of a first visit: the engine, the data of the recipe and the files of the processor's
  variant (`bundle.first_load`). The model selection, `limits.max_size_mb`, the line "a first visit
  downloads" of `build` and `info().bytes` of the runtime all give this number; the selection used to
  leave out the data of the labels and count the heaviest variant, so a model close to the limit may
  now be taken or refused the other way.
- `LICENSES.txt` holds the full texts of Apache-2.0 (Transformers.js and the cifarium runtime) and
  MIT (ONNX Runtime) and, for a model whose license is not permissive, the notice its terms ask for.
- `report.md` for people and `report.json` for programs, in the folder `build` writes and, when no
  model fits (exit code 1), alone in `output`. `report.json`, format 1: `format` and `cifarium` come
  first, every key is in camel case. It has `outcome` (`built` or `no-model`), `exitCode`, the
  `settings` and `examples` it ran on, the `device` (Chrome version, adapter kind, threads,
  isolation), every candidate in the order of the walk (`candidates[]`: `verdict`, one of `passed`,
  `latency`, `license`, `size`, `language`, `recipe`, `accuracy`, `vectors`, `bench`, `not-tried`; a
  `reason` for people; the weight of the first load; the share, the `margin` over the pass rate,
  which is negative for a model that fell short, and the misses by line of the examples file; the
  times) and, for the chosen model, the weight of the first load by parts and its one-thread run.
  The keys of `report.md` are not a contract. Neither file holds the text of an example.
