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
- Exit codes of `build`, `check` and `init`: 0 done, 1 no model fits (`check`: the folder did not
  pass), 2 the settings, the examples or the command line are wrong (`init`: a file is already
  there; `check`: no folder, manifest or report, a format newer than this cifarium reads, examples
  that do not fit the labels of the folder), 3 the environment failed (Chrome, the network, the
  cache, the disk, the installation; `init`: the folder cannot be written; also an error cifarium
  did not expect, with its traceback). `catalog-update` keeps its own codes.
- `cifarium check [folder] [--examples FILE] [--hosting netlify|vercel|nginx] [--headed]`: serves the
  folder with the headers of one of its hosting files, loads the model on a page through
  `cifarium.js` as a site would and answers the examples; it fails (code 1) when the share is under
  `settings.passRate` of `report.json`, when the page or the worker is not isolated or has fewer
  threads than ONNX Runtime takes for the cores, when the usual answer is over twice
  `settings.limits.maxLatencyMs` (over once is a warning), or when `info().bytes` differs from
  `chosen.firstLoad.bytes`. It reads `cifarium.json` and `report.json` and writes nothing, the
  folder included. Without a folder and `--examples` it takes `output` and `examples` from
  `cifarium.yaml` in the current folder. The lines it prints are not a contract.
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
- `build` keeps the vectors of the examples between runs, in the `results` folder of the cifarium
  cache (`%LOCALAPPDATA%\cifarium` on Windows), so a second build on the same examples computes none
  of them and answers the same: the same choice, the same table, the same bytes of the folder, the
  `report.json` times aside. The times are not kept: the model it chooses is measured again in
  every build. To start from nothing, delete that folder; `build` prints its path. The folder and
  its files are not a contract, and `build` has no flag for it.
- The recipe `zero-shot` ("does the label fit", the NLI models of the catalog) takes part in the
  model selection: it runs on the bench, is scored on every example (the label with the highest
  score; the examples teach it nothing, so none is left out) and gets its share and its `verdict`
  in the table and in `report.json` like any other candidate. Its time per example is the time of
  one text with all of its labels. `vectors` in its row means that the scores of the labels the
  bench returned cannot be read; no new verdict word.
- The folder of the recipe `zero-shot`, which `build` writes with code 0 when it chooses that recipe:
  `recipe` is `"zero-shot"` in the manifest, `params.hypotheses` lists one finished sentence for each
  label in the order of `labels` (the runtime puts none together, and it asks the model what the bench
  asked), `data` is empty and there is no `data/labels.bin`, `params` has no `pooling` or `prefix`.
  The first load is the engine and the files of the model. The manifest of `embed-centroid` is as it
  was. `cifarium.js` runs both recipes: `run` answers `{ label, score, scores }` for each, but the scale
  of `score` is the recipe's: from -1 to 1 for `embed-centroid`, from 0 to 1 with all the scores
  adding up to 1 for `zero-shot` (`info().recipe` says which; ARCHITECTURE.md, section 8). `check`
  judges such a folder with the same four checks.
- `host: hub` works: `build` writes the folder without `models/` and ends with code 0, and the page
  downloads the model's files from Hugging Face at the revision the catalog pins
  (`https://huggingface.co/<repo>/resolve/<revision>/…`, which hands the large files on to its CDN
  under `*.hf.co`). The manifest gets `model.host`: `"bundle"` or `"hub"`; with `"hub"`,
  `model.path` is `null`. A manifest without `model.host` is read as `"bundle"`, and the runtime
  refuses a value it does not know with `unsupported`. Everything else in the manifest, the sizes of
  the files and the weight of a first visit included, is the same for both hosts. For a model's file
  of such a build, `file` of a progress event is the file's full address on Hugging Face. The
  isolation headers of `hosting/` need no change: Hugging Face answers with CORS, which
  `Cross-Origin-Embedder-Policy: require-corp` lets through. `check` on such a folder needs the
  network and says so; a download from Hugging Face that a new try may fix ends it with code 3, not 1.
- The runtime clears the files of earlier builds of its folder out of the visitor's browser store
  (Cache API, `transformers-cache`) when it loads: a rebuild with another revision, another engine
  or another host no longer leaves the old files there. It keeps the list of what its folder stored
  under the address of the manifest with `?cifarium-cached-files`, and touches nothing else in the
  store.