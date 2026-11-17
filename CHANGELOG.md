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
