/**
 * Background thread of the cifarium runtime. cifarium.js starts it as a module worker and sends
 * numbered requests: one "load", then any number of "run". Each request waits for the one before
 * it, so answers come in the order the page asked (ARCHITECTURE.md, sections 1.5, 1.7 and 8).
 *
 * Replies: {id, type: "progress", progress}, {id, type: "done", value},
 * {id, type: "fail", error: {code, message, cause}}, or, to a load whose graphics card failed,
 * {id, type: "retry", fallback, carry}: the page ends this thread and loads on the processor
 * in a new one. `cause` is the original error as plain fields: {name, message, stack}.
 */

/** @typedef {import('./cifarium.js').Manifest} Manifest */
/** @typedef {import('./cifarium.js').Variant} Variant */
/** @typedef {import('./cifarium.js').Progress} Progress */
/** @typedef {import('./cifarium.js').Fallback} Fallback */
/** @typedef {import('./cifarium.js').Result} Result */
/** @typedef {import('./cifarium.js').Score} Score */

/** Weights that need the adapter's shader-f16. The library checks it only for fp16. */
const HALF_PRECISION = new Set(['fp16', 'q4f16']);

/** How far a centre's length may be from 1; Python writes them in float32. */
const UNIT_TOLERANCE = 1e-3;

/** The library's pipeline of each recipe (the operations of ARCHITECTURE.md, section 2). */
const PIPELINES = { 'embed-centroid': 'feature-extraction', 'zero-shot': 'zero-shot-classification' };

/** Short text for the trial run that load() makes on the chosen device. */
const TRIAL_TEXT = 'cifarium';

/**
 * Where a build with `model.host` "hub" takes the model's files from: the one address outside the
 * site that the runtime ever asks, and only for such a build (bundle.HUB_ADDRESS in Python).
 */
const HUB = 'https://huggingface.co/';

/**
 * The query that turns the manifest's address into the key of this folder's list in the browser's
 * store: which files its last load put there (see `forgetEarlierFiles`).
 */
const LIST_QUERY = '?cifarium-cached-files';

class Failure extends Error {
  /**
   * @param {string} code One of ERROR_CODES in cifarium.js.
   * @param {string} message
   * @param {unknown} [cause] The error that led to this one.
   */
  constructor(code, message, cause) {
    super(message, cause === undefined ? undefined : { cause });
    this.code = code;
  }
}

/**
 * The graphics card's variant failed and the processor's remains. It is tried in a new thread:
 * after a failed session ONNX Runtime does not create even a sound one in the same thread
 * ("protobuf parsing failed" on an intact file, found 17.11.2026).
 */
class Retry extends Error {
  /**
   * @param {Fallback} fallback Why the card's variant is not in use.
   * @param {[string, {loaded: number, total: number}][]} carry The bytes counted so far.
   */
  constructor(fallback, carry) {
    super(fallback.message);
    this.fallback = fallback;
    this.carry = carry;
  }
}

/**
 * What run() needs, set by a successful load: one of the two recipes.
 * @typedef {Object} VectorModel The recipe "embed-centroid".
 * @property {"embed-centroid"} recipe
 * @property {any} extractor A "feature-extraction" pipeline.
 * @property {string[]} labels
 * @property {Float32Array} centres labels × dims, one unit-length row per label.
 * @property {number} dims
 * @property {"mean"|"cls"|"none"} pooling
 * @property {string} prefix
 *
 * @typedef {Object} EntailmentModel The recipe "zero-shot".
 * @property {"zero-shot"} recipe
 * @property {any} classifier A "zero-shot-classification" pipeline.
 * @property {string[]} labels
 * @property {string[]} sentences The hypotheses of the manifest, ready to hand to the library: every
 *   "$" doubled (see `passable`).
 *
 * @typedef {VectorModel|EntailmentModel} Model
 */

/** @type {Model|null} */
let model = null;

/**
 * The library puts each label into `hypothesis_template` with `String.prototype.replace`, which
 * reads `$&`, `$1` and the like in what it inserts. With the template "{}" and every "$" doubled,
 * what it checks is the hypothesis as the build wrote it. The same as the bench page does (web/bench/bench.js),
 * so that a share counted on the bench is the share a site gets.
 */
const WHOLE = '{}';
let queue = Promise.resolve();

self.onmessage = (event) => {
  const request = event.data;
  queue = queue.then(() => handle(request));
};

/** @param {any} request */
async function handle(request) {
  try {
    let value;
    if (request.type === 'load') {
      value = await load(request);
    } else if (request.type === 'run') {
      value = await run(request.text);
    } else {
      throw new Failure('input', `Unknown request ${JSON.stringify(request.type)}.`);
    }
    self.postMessage({ id: request.id, type: 'done', value });
  } catch (error) {
    if (error instanceof Retry) {
      self.postMessage({ id: request.id, type: 'retry', fallback: error.fallback, carry: error.carry });
      return;
    }
    const known = error instanceof Failure;
    const code = known ? error.code : request.type === 'load' ? 'load' : 'run';
    self.postMessage({
      id: request.id,
      type: 'fail',
      error: { code, message: describe(error), cause: plain(known ? error.cause : error) },
    });
  }
}

/**
 * Counts the bytes of every file of one load and reports them to the page.
 */
class Tally {
  /**
   * @param {number} id Number of the load request.
   * @param {[string, {loaded: number, total: number}][]} [carry] What a failed thread counted.
   */
  constructor(id, carry = []) {
    this.id = id;
    /** @type {Map<string, {loaded: number, total: number}>} */
    this.files = new Map(carry);
    /** Whether the "prepare" event of this thread has been sent. */
    this.preparing = false;
  }

  /**
   * @param {string} file
   * @param {number} total
   */
  expect(file, total) {
    if (!this.files.has(file)) {
      this.files.set(file, { loaded: 0, total });
    }
  }

  /**
   * @param {string} file
   * @param {number} loaded
   * @param {number} [total] Used only for a file the manifest did not name.
   */
  update(file, loaded, total = 0) {
    this.expect(file, Math.max(total, loaded));
    const entry = /** @type {{loaded: number, total: number}} */ (this.files.get(file));
    const value = Math.min(Math.max(loaded, entry.loaded), entry.total);
    if (value === entry.loaded && value !== 0) {
      return;
    }
    entry.loaded = value;
    this.send(file, entry);
  }

  /** @param {string} file */
  finish(file) {
    const entry = this.files.get(file);
    if (entry) {
      this.update(file, entry.total);
    }
  }

  /**
   * Leaves only what was really downloaded of a file that is no longer needed.
   * @param {string} file
   */
  drop(file) {
    const entry = this.files.get(file);
    if (entry && entry.loaded === 0) {
      this.files.delete(file);
    } else if (entry) {
      entry.total = entry.loaded;
    }
  }

  finishAll() {
    for (const [file, entry] of this.files) {
      if (entry.loaded < entry.total) {
        this.finish(file);
      }
    }
  }

  /**
   * One "download" event for the step; the one that brings the last byte is followed by the
   * "prepare" event, once per thread.
   * @param {string} file
   * @param {{loaded: number, total: number}} entry
   */
  send(file, entry) {
    let allLoaded = 0;
    let allTotal = 0;
    for (const item of this.files.values()) {
      allLoaded += item.loaded;
      allTotal += item.total;
    }
    const post = (/** @type {Progress["phase"]} */ phase) => {
      /** @type {Progress} */
      const progress = {
        phase,
        file,
        loaded: entry.loaded,
        total: entry.total,
        allLoaded,
        allTotal,
        percent: allTotal > 0 ? (allLoaded / allTotal) * 100 : 100,
      };
      self.postMessage({ id: this.id, type: 'progress', progress });
    };
    post('download');
    if (!this.preparing && allTotal > 0 && allLoaded === allTotal) {
      this.preparing = true;
      post('prepare');
    }
  }
}

/**
 * Watches the downloads the library makes, so a failed model can be told apart: a download that
 * did not happen (the "network" code) from a model that did not start (the "load" code). The
 * library hides a failed `fetch` behind its own error, so the runtime sees only what passes
 * through `fetch`: a rejection, or an answer that is worth another try.
 * @param {any} env The library's settings.
 * @returns {{problem: Error|null}} `problem` is the last such failure since it was cleared.
 */
function watchFetch(env) {
  const watch = { problem: /** @type {Error|null} */ (null) };
  const wrap = (/** @type {Function} */ real) => async (/** @type {any[]} */ ...args) => {
    let response;
    try {
      response = await real(...args);
    } catch (error) {
      watch.problem = error instanceof Error ? error : new Error(String(error));
      throw error;
    }
    if (retryable(response.status)) {
      watch.problem = new Error(`HTTP ${response.status} for ${response.url}`);
    }
    return response;
  };
  // The library takes model files through env.fetch; ONNX Runtime takes the data files of a
  // big graph through the thread's own fetch.
  if (typeof env.fetch === 'function') {
    env.fetch = wrap(env.fetch);
  }
  const own = self.fetch.bind(self);
  self.fetch = wrap(own);
  return watch;
}

/**
 * Loads the model on the first variant that works in this thread. When the graphics card's
 * variant fails and the processor's remains, answers with a Retry: the page then starts a new
 * thread with `retry` set, which tries only the processor and goes on with the same count.
 * @param {{id: number, manifestUrl: string, manifest: Manifest, device: "auto"|"wasm"|"webgpu",
 *   retry?: {fallback: Fallback, carry: [string, {loaded: number, total: number}][]}}} request
 */
async function load({ id, manifestUrl, manifest, device, retry }) {
  if (model) {
    throw new Failure('load', 'This background thread has already loaded a model.');
  }
  const address = (/** @type {string} */ path) => new URL(path, manifestUrl).href;
  const { engine, data } = manifest;
  const zeroShot = manifest.recipe === 'zero-shot';
  // The files of the recipe: the centres for "embed-centroid", none for "zero-shot".
  const dataFiles = Object.values(data);
  const tally = new Tally(id, retry?.carry);
  const onHub = manifest.model.host === 'hub';
  // What goes in front of a model's file: a path relative to the manifest, or the file's address on
  // Hugging Face, which is also the key the library stores it under.
  const modelFolder = onHub
    ? `${HUB}${manifest.model.repo}/resolve/${encodeURIComponent(manifest.model.revision)}/`
    : `${manifest.model.path}${manifest.model.repo}/`;
  // Before anything is downloaded: a sentence the library would change is a failed load, not a
  // build that answers about a sentence nobody wrote.
  const sentences = zeroShot ? passable(/** @type {{hypotheses: string[]}} */ (manifest.params).hypotheses) : [];

  // The variant is chosen first, so that every file of this load is counted before the first
  // byte arrives: the "prepare" phase starts when the count is complete, and a count that
  // learned of the model's files late would be complete too soon.
  const { order, fallback } = retry && manifest.variants.cpu
    ? { order: [{ name: /** @type {"cpu"} */ ('cpu'), variant: manifest.variants.cpu }], fallback: retry.fallback }
    : await plan(manifest.variants, device);
  if (order.length === 0) {
    throw new Failure('load', `The model did not start: no variant could be tried${fallback ? `; ${fallback.message}` : ''}.`);
  }
  for (const file of [engine.library, engine.loader, engine.executor, ...dataFiles]) {
    tally.expect(file.path, file.bytes);
  }
  for (const [file, bytes] of Object.entries(order[0].variant.files)) {
    tally.expect(modelFolder + file, bytes);
  }

  const library = await importModule(address(engine.library.path), 'Transformers.js');
  tally.finish(engine.library.path);
  // ONNX Runtime imports the loader itself when it makes the first session; importing it here
  // counts its bytes with the others and lets a failure be named now. A module that was
  // imported once is not fetched again.
  await importModule(address(engine.loader.path), 'the ONNX Runtime loader');
  tally.finish(engine.loader.path);
  const { env, pipeline } = library;
  // Before anything big is downloaded, so that the space of an earlier build is free first.
  await forgetEarlierFiles(env.cacheKey, manifestUrl, storable(manifest, manifestUrl, modelFolder));

  const { labels, centres, dims } = zeroShot
    ? { labels: manifest.labels.map((label) => label.name), centres: new Float32Array(0), dims: 0 }
    : await readCentres(manifest, address, tally);

  if (onHub) {
    // The model's files come from Hugging Face, everything else from this build. The revision goes
    // into the template of the path itself, not into the `revision` option: Transformers.js 4.3.2
    // drops that option where it lists the files and asks for their sizes, and would ask for them
    // at `main`, whatever the repository holds there today (ARCHITECTURE.md, section 1.6).
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.remoteHost = HUB;
    env.remotePathTemplate = `{model}/resolve/${encodeURIComponent(manifest.model.revision)}/`;
  } else {
    // Every file comes from this build: nothing from Hugging Face or a CDN. The models' folder
    // must be a path, not a full address: the library checks that a local file exists (it decides
    // so whether to load the tokenizer) only for paths, and skips addresses that start with http.
    const models = new URL(/** @type {string} */ (manifest.model.path), manifestUrl);
    if (models.origin !== self.location.origin) {
      throw new Failure('load', `The model lies at ${models.origin}, cifarium-worker.js at ${self.location.origin}; they must be on the same site.`);
    }
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = models.pathname;
  }
  // The runtime downloads the executor itself, to report its progress (the library does not)
  // and to keep it in the same store. With wasmBinary set and no "wasm" in wasmPaths, the
  // library leaves the executor alone and ONNX Runtime takes the buffer.
  env.backends.onnx.wasm.wasmBinary = await readExecutor(address(engine.executor.path), engine.executor.path, tally, env.cacheKey);
  env.backends.onnx.wasm.wasmPaths = { mjs: address(engine.loader.path) };
  const watch = watchFetch(env);

  for (const [index, { name, variant }] of order.entries()) {
    for (const [file, bytes] of Object.entries(variant.files)) {
      tally.expect(modelFolder + file, bytes);
    }
    let pipe = null;
    watch.problem = null;
    try {
      pipe = await pipeline(PIPELINES[manifest.recipe], manifest.model.repo, {
        device: variant.device,
        dtype: variant.dtype,
        model_file_name: variant.graph,
        ...(variant.externalData > 0 ? { use_external_data_format: variant.externalData } : {}),
        progress_callback: (/** @type {any} */ event) => {
          if (!event.file) {
            return;
          }
          if (event.status === 'progress') {
            tally.update(modelFolder + event.file, event.loaded ?? 0, event.total ?? 0);
          } else if (event.status === 'done') {
            tally.finish(modelFolder + event.file);
          }
        },
      });
      // A card can fail on the first computation rather than on loading, or return NaN.
      if (zeroShot) {
        entailmentOf(pipe);
        await scoreLabels(pipe, sentences, TRIAL_TEXT);
      } else {
        const { pooling, prefix } = /** @type {{pooling: "mean"|"cls"|"none", prefix: string}} */ (manifest.params);
        const trial = await toVector(pipe, prefix + TRIAL_TEXT, pooling);
        scaleToUnit(trial);
        if (trial.length !== dims) {
          throw new Failure('load', `The model gives vectors of ${trial.length} numbers, the centres have ${dims}.`);
        }
      }
    } catch (error) {
      // A file that did not download is not a reason to try another variant: the next one
      // needs files from the same place. Say so and let the page try the whole load again.
      const missing = error instanceof Failure ? null : watch.problem ?? (isTransportError(error) ? error : null);
      if (missing) {
        throw new Failure('network', `A file of the model did not download: ${describe(missing)}`, missing);
      }
      // Any other failure of the card's variant, the trial run included, hands over to the processor.
      const next = order[index + 1]?.variant.files;
      if (!next) {
        const before = fallback && name === 'cpu' ? `; before that, ${fallback.message}` : '';
        throw new Failure('load', `The model did not start: ${name} (${variant.device}, ${variant.dtype}): ${describe(error)}${before}.`, error);
      }
      for (const file of Object.keys(variant.files)) {
        if (!(file in next)) {
          tally.drop(modelFolder + file);
        }
      }
      /** @type {Fallback} */
      const failed = { reason: 'failed', message: `the graphics card failed: ${describe(error)}` };
      console.warn(`cifarium: ${failed.message}; trying the processor in a new thread.`);
      throw new Retry(failed, [...tally.files]);
    }

    if (zeroShot) {
      model = { recipe: 'zero-shot', classifier: pipe, labels, sentences };
    } else {
      const { pooling, prefix } = /** @type {{pooling: "mean"|"cls"|"none", prefix: string}} */ (manifest.params);
      model = { recipe: 'embed-centroid', extractor: pipe, labels, centres, dims, pooling, prefix };
    }
    tally.finishAll();
    const bytes = [engine.library, engine.loader, engine.executor, ...dataFiles]
      .reduce((sum, file) => sum + file.bytes, 0)
      + Object.values(variant.files).reduce((sum, size) => sum + size, 0);
    return {
      variant: name,
      device: variant.device,
      dtype: variant.dtype,
      threads: env.backends.onnx.wasm.numThreads ?? null,
      crossOriginIsolated: self.crossOriginIsolated,
      bytes,
      fallback,
    };
  }
  throw new Failure('load', 'The model did not start: no variant could be tried.');
}

/**
 * Which variants to try, in order, and why the graphics card's one is left out.
 * @param {{cpu?: Variant, gpu?: Variant}} variants
 * @param {"auto"|"wasm"|"webgpu"} device
 * @returns {Promise<{order: {name: "cpu"|"gpu", variant: Variant}[], fallback: Fallback|null}>}
 */
async function plan(variants, device) {
  /** @type {{name: "cpu"|"gpu", variant: Variant}[]} */
  const order = [];
  /** @type {Fallback|null} */
  let fallback = null;
  if (variants.gpu) {
    if (device === 'wasm') {
      fallback = { reason: 'requested', message: 'the page asked for the processor' };
    } else {
      const problem = device === 'webgpu' ? null : await gpuProblem(variants.gpu.dtype);
      if (problem) {
        fallback = problem;
      } else {
        order.push({ name: 'gpu', variant: variants.gpu });
      }
    }
  }
  if (variants.cpu) {
    order.push({ name: 'cpu', variant: variants.cpu });
  }
  return { order, fallback };
}

/**
 * Why the graphics card cannot take weights of this dtype, or null if it can.
 * @param {string} dtype
 * @returns {Promise<Fallback|null>}
 */
async function gpuProblem(dtype) {
  const gpu = /** @type {any} */ (navigator).gpu;
  if (!gpu) {
    return { reason: 'no-webgpu', message: 'this browser has no WebGPU here' };
  }
  let adapter;
  try {
    // The preference the library itself passes to ONNX Runtime.
    adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (error) {
    return { reason: 'no-adapter', message: `requestAdapter() failed: ${describe(error)}` };
  }
  if (!adapter) {
    return { reason: 'no-adapter', message: 'the browser gave no WebGPU adapter' };
  }
  if (adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter) {
    return { reason: 'software-adapter', message: 'the WebGPU adapter is a software one' };
  }
  if (HALF_PRECISION.has(dtype) && !adapter.features.has('shader-f16')) {
    return { reason: 'no-shader-f16', message: `the adapter has no shader-f16, which ${dtype} weights need` };
  }
  return null;
}

/**
 * Reads the centres of the labels; the names come from the manifest.
 * @param {Manifest} manifest
 * @param {(path: string) => string} address
 * @param {Tally} tally
 */
async function readCentres(manifest, address, tally) {
  const labels = manifest.labels.map((label) => label.name);
  const file = manifest.data.centres;
  const bytes = await fetchFile(address(file.path));
  tally.finish(file.path);
  const dims = bytes.byteLength / 4 / labels.length;
  if (!Number.isInteger(dims) || dims < 1) {
    throw new Failure('load', `${file.path} has ${bytes.byteLength} bytes, which is not ${labels.length} rows of float32.`);
  }
  const view = new DataView(bytes);
  const centres = new Float32Array(labels.length * dims);
  for (let index = 0; index < centres.length; index += 1) {
    centres[index] = view.getFloat32(index * 4, true);
  }
  for (let row = 0; row < labels.length; row += 1) {
    let sum = 0;
    for (let column = 0; column < dims; column += 1) {
      sum += centres[row * dims + column] ** 2;
    }
    if (!(Math.abs(Math.sqrt(sum) - 1) <= UNIT_TOLERANCE)) {
      throw new Failure('load', `The centre of ${JSON.stringify(labels[row])} in ${file.path} has length ${Math.sqrt(sum)}, not 1.`);
    }
  }
  return { labels, centres, dims };
}

/**
 * Imports a module of the build. A failed import says only that the module could not be
 * fetched, whatever the cause, so a second request tells a connection that broke (worth another
 * try, "network") from a file that is missing or wrong ("load").
 * @param {string} href
 * @param {string} what
 */
async function importModule(href, what) {
  try {
    return await import(href);
  } catch (error) {
    let again;
    try {
      again = await self.fetch(href, { method: 'HEAD' });
    } catch {
      throw new Failure('network', `Could not download ${what} from ${href}: ${describe(error)}`, error);
    }
    const code = retryable(again.status) ? 'network' : 'load';
    throw new Failure(code, `Could not load ${what} from ${href}: ${describe(error)}`, error);
  }
}

/**
 * The executor of ONNX Runtime from the browser's store, or downloaded with progress and stored.
 * Its address carries the engine's version, so the stored copy never goes stale.
 * @param {string} href
 * @param {string} file
 * @param {Tally} tally
 * @param {string} cacheName The library's store, so one deletion clears everything.
 * @returns {Promise<ArrayBuffer>}
 */
async function readExecutor(href, file, tally, cacheName) {
  let store = null;
  try {
    // Cache API exists only in secure contexts: HTTPS and localhost.
    store = typeof caches === 'undefined' ? null : await caches.open(cacheName);
  } catch {
    store = null;
  }
  const stored = await store?.match(href).catch(() => undefined);
  if (stored) {
    const buffer = await stored.arrayBuffer();
    tally.finish(file);
    return buffer;
  }
  const buffer = await fetchFile(href, (loaded) => tally.update(file, loaded));
  tally.finish(file);
  try {
    await store?.put(href, new Response(buffer, { headers: { 'Content-Type': 'application/wasm' } }));
  } catch (error) {
    console.warn(`cifarium: could not store ${file} in the browser:`, error);
  }
  return buffer;
}

/**
 * The address of every file of this manifest that the library or the runtime may keep in the
 * browser's store: the engine and the model's files of every variant, by the address the store
 * keys them with.
 * @param {Manifest} manifest
 * @param {string} manifestUrl
 * @param {string} modelFolder What goes in front of a model's file (see `load`).
 * @returns {string[]}
 */
function storable(manifest, manifestUrl, modelFolder) {
  const { engine, variants } = manifest;
  const files = [engine.library.path, engine.loader.path, engine.executor.path];
  for (const variant of [variants.cpu, variants.gpu]) {
    if (variant) {
      files.push(...Object.keys(variant.files).map((file) => modelFolder + file));
    }
  }
  return [...new Set(files.map((file) => new URL(file, manifestUrl).href))];
}

/**
 * Clears the files of this folder's earlier builds out of the browser's store. The library keeps
 * every model file it downloads there, and the runtime the executor, each under its address; a
 * rebuild changes the addresses (another revision, another version of the engine, `host` "hub"
 * instead of "bundle"), and without this every rebuild would leave each visitor tens of megabytes
 * more. Which files are this folder's is not guessed from their addresses: the store holds, under
 * the manifest's address with LIST_QUERY, the list of files the last load of this folder could
 * have put there, and what that list has and this manifest does not is deleted. Nothing else in
 * the store is touched: other folders of the site, a page of the site that uses Transformers.js
 * itself. A failure here is written to the console and never fails the load.
 * @param {string} cacheName The library's store (`env.cacheKey`).
 * @param {string} manifestUrl
 * @param {string[]} wanted What `storable` gives for this manifest.
 */
async function forgetEarlierFiles(cacheName, manifestUrl, wanted) {
  if (typeof caches === 'undefined') {
    return; // Cache API exists only in secure contexts: HTTPS and localhost.
  }
  try {
    const store = await caches.open(cacheName);
    const key = new URL(manifestUrl);
    key.search = LIST_QUERY;
    key.hash = '';
    const stored = await store.match(key.href);
    const before = stored ? await stored.json().catch(() => null) : null;
    const keep = new Set(wanted);
    const earlier = Array.isArray(before?.files)
      ? before.files.filter((/** @type {unknown} */ file) => typeof file === 'string' && !keep.has(file))
      : [];
    await Promise.all(earlier.map((/** @type {string} */ file) => store.delete(file)));
    await store.put(key.href, new Response(JSON.stringify({ files: wanted }), {
      headers: { 'Content-Type': 'application/json' },
    }));
  } catch (error) {
    console.warn('cifarium: could not clear the files of earlier builds from the browser:', error);
  }
}

/**
 * Downloads a whole file, reporting the bytes read so far. A request that fails, a transfer
 * that breaks off and an answer of 408, 425, 429 or 5xx are "network"; any other HTTP error is
 * "load": the server says the file is not there or not for us, and asking again changes nothing.
 * @param {string} href
 * @param {(loaded: number) => void} [onBytes]
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchFile(href, onBytes) {
  let response;
  try {
    response = await fetch(href);
  } catch (error) {
    throw new Failure('network', `Could not download ${href}: ${describe(error)}`, error);
  }
  if (!response.ok) {
    throw new Failure(retryable(response.status) ? 'network' : 'load', `Could not download ${href}: HTTP ${response.status}.`);
  }
  try {
    if (!onBytes || !response.body) {
      return await response.arrayBuffer();
    }
    const reader = response.body.getReader();
    /** @type {Uint8Array[]} */
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      chunks.push(value);
      loaded += value.byteLength;
      onBytes(loaded);
    }
    const whole = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      whole.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return whole.buffer;
  } catch (error) {
    throw new Failure('network', `The download of ${href} broke off: ${describe(error)}`, error);
  }
}

/**
 * Scores every label and names the best.
 * @param {string} text
 * @returns {Promise<Result>}
 */
async function run(text) {
  if (!model) {
    throw new Failure('not-loaded', 'The model is not loaded.');
  }
  if (typeof text !== 'string' || text.trim() === '') {
    throw new Failure('input', 'run() needs a text with something in it.');
  }
  const loaded = model;
  /** @type {Score[]} */
  let scores;
  if (loaded.recipe === 'zero-shot') {
    const row = await scoreLabels(loaded.classifier, loaded.sentences, text);
    scores = loaded.labels.map((label, index) => ({ label, score: row[index] }));
  } else {
    const vector = await toVector(loaded.extractor, loaded.prefix + text, loaded.pooling);
    scaleToUnit(vector);
    if (vector.length !== loaded.dims) {
      throw new Failure('run', `The model gave a vector of ${vector.length} numbers, the centres have ${loaded.dims}.`);
    }
    scores = loaded.labels.map((label, row) => {
      let score = 0;
      const start = row * loaded.dims;
      for (let column = 0; column < loaded.dims; column += 1) {
        score += vector[column] * loaded.centres[start + column];
      }
      return { label, score };
    });
  }
  // Sorting is stable: labels with the same score stay in the order of the settings, so the
  // best is the label listed first among equals, as in evaluate.predict and evaluate.zero_shot.
  scores.sort((a, b) => b.score - a.score);
  if (scores.length === 0) {
    throw new Failure('run', 'No label got a score.');
  }
  return { label: scores[0].label, score: scores[0].score, scores };
}

/**
 * The hypotheses of the manifest in the form the library takes: every "$" doubled. The library
 * would read "$&" and the like in a label, so the sentence is checked here the way it will be
 * built, and a sentence that would come out changed is a failed load.
 * @param {string[]} hypotheses
 * @returns {string[]}
 */
function passable(hypotheses) {
  return hypotheses.map((hypothesis, index) => {
    const passed = hypothesis.replace(/\$/g, '$$$$');
    if (WHOLE.replace('{}', passed) !== hypothesis) {
      throw new Failure('load', `The library would check ${JSON.stringify(WHOLE.replace('{}', passed))} instead of hypothesis ${index}.`);
    }
    return passed;
  });
}

/**
 * A model without an "entailment" label cannot check a hypothesis: the library would take logit 2
 * and say so only in the console.
 * @param {any} classifier A "zero-shot-classification" pipeline.
 */
function entailmentOf(classifier) {
  const names = Object.keys(classifier.model.config.label2id ?? {});
  if (!names.some((name) => name.toLowerCase() === 'entailment')) {
    throw new Failure('load', `The model has no "entailment" label (it has ${names.join(', ') || 'none'}): it cannot check hypotheses.`);
  }
}

/**
 * One text, one score for each sentence, in the order of `sentences`. The same call as the bench
 * page makes (web/bench/bench.js, `zeroShot`): one run of the model per sentence and the softmax of
 * the entailment logits over them (`multi_label: false`), so the scores add up to 1. The library
 * sorts the labels by score; this puts them back in order. Equal sentences get equal logits, so one
 * score for a sentence is right for each of them.
 * @param {any} classifier A "zero-shot-classification" pipeline.
 * @param {string[]} sentences As `passable` made them.
 * @param {string} text
 * @returns {Promise<number[]>}
 */
async function scoreLabels(classifier, sentences, text) {
  const output = await classifier(text, sentences, { hypothesis_template: WHOLE, multi_label: false });
  /** @type {Map<string, number>} */
  const scoreOf = new Map(output.labels.map((/** @type {string} */ label, /** @type {number} */ index) => [label, output.scores[index]]));
  const row = sentences.map((sentence) => scoreOf.get(sentence));
  if (!row.every((score) => Number.isFinite(score))) {
    throw new Failure('run', `The model returned scores ${JSON.stringify(row)}.`);
  }
  return /** @type {number[]} */ (row);
}

/**
 * One text to one vector, not yet scaled. The same as the bench page (web/bench/bench.js).
 * @param {any} extractor A "feature-extraction" pipeline.
 * @param {string} text Input with the prefix already in front.
 * @param {"mean"|"cls"|"none"} pooling
 * @returns {Promise<Float32Array>}
 */
async function toVector(extractor, text, pooling) {
  if (pooling !== 'none') {
    const output = await extractor(text, { pooling, normalize: false });
    return Float32Array.from(output.data);
  }
  // The pipeline would return the token vectors (last_hidden_state) here. Models that pool
  // inside the graph give the finished vector as a separate output.
  const inputs = extractor.tokenizer(text, { padding: true, truncation: true });
  const outputs = await extractor.model(inputs);
  const vector = outputs.sentence_embedding;
  if (!vector) {
    throw new Failure('run', `pooling "none" needs a sentence_embedding output; the model has ${Object.keys(outputs).join(', ')}.`);
  }
  if (vector.dims.length !== 2 || vector.dims[0] !== 1) {
    throw new Failure('run', `sentence_embedding has shape [${vector.dims.join(', ')}], expected [1, size].`);
  }
  return Float32Array.from(vector.data);
}

/**
 * Scales a vector to unit length in place.
 * @param {Float32Array} vector
 */
function scaleToUnit(vector) {
  let sum = 0;
  for (const value of vector) {
    sum += value * value;
  }
  const norm = Math.sqrt(sum);
  if (!(norm > 0) || !Number.isFinite(norm)) {
    throw new Failure('run', `The model returned a vector of length ${norm}.`);
  }
  for (let index = 0; index < vector.length; index += 1) {
    vector[index] /= norm;
  }
}

/**
 * Whether a new try may help when the server answered with this status.
 * @param {number} status
 */
function retryable(status) {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

/**
 * The error `fetch` throws when the request or the transfer fails, in the words of Chrome
 * ("Failed to fetch", "network error"), Firefox ("NetworkError when attempting to fetch
 * resource") and Safari ("Load failed").
 * @param {unknown} error
 */
function isTransportError(error) {
  return error instanceof TypeError && /fetch|network|load failed/i.test(error.message);
}

/**
 * An error as plain fields, which cross to the page without loss.
 * @param {unknown} error
 * @returns {{name: string, message: string, stack?: string}|undefined}
 */
function plain(error) {
  if (error === undefined) {
    return undefined;
  }
  if (error instanceof Error) {
    return { name: error.name, message: error.message, stack: error.stack };
  }
  return { name: 'Error', message: String(error) };
}

/** @param {unknown} error */
function describe(error) {
  return error instanceof Error ? error.message : String(error);
}
