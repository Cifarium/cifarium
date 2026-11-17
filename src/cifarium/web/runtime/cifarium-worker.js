/**
 * Background thread of the cifarium runtime. cifarium.js starts it as a module worker and sends
 * numbered requests: one "load", then any number of "run". Each request waits for the one before
 * it, so answers come in the order the page asked (ARCHITECTURE.md, sections 1.5, 1.7 and 8).
 *
 * Replies: {id, type: "progress", progress}, {id, type: "done", value},
 * {id, type: "fail", error: {code, message}}, or, to a load whose graphics card failed,
 * {id, type: "retry", fallback, carry}: the page ends this thread and loads on the processor
 * in a new one.
 */

/** @typedef {import('./cifarium.js').Manifest} Manifest */
/** @typedef {import('./cifarium.js').Variant} Variant */
/** @typedef {import('./cifarium.js').Progress} Progress */

/** Weights that need the adapter's shader-f16. The library checks it only for fp16. */
const HALF_PRECISION = new Set(['fp16', 'q4f16']);

/** How far a centre's length may be from 1; Python writes them in float32. */
const UNIT_TOLERANCE = 1e-3;

/** Short text for the trial run that load() makes on the chosen device. */
const TRIAL_TEXT = 'cifarium';

class Failure extends Error {
  /**
   * @param {string} code
   * @param {string} message
   */
  constructor(code, message) {
    super(message);
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
   * @param {string} fallback Why the card's variant is not in use.
   * @param {[string, {loaded: number, total: number}][]} carry The bytes counted so far.
   */
  constructor(fallback, carry) {
    super(fallback);
    this.fallback = fallback;
    this.carry = carry;
  }
}

/**
 * @typedef {Object} Model What run() needs, set by a successful load.
 * @property {any} extractor A "feature-extraction" pipeline.
 * @property {string[]} labels
 * @property {Float32Array} centres labels × dims, one unit-length row per label.
 * @property {number} dims
 * @property {"mean"|"cls"|"none"} pooling
 * @property {string} prefix
 */

/** @type {Model|null} */
let model = null;
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
    const code = error instanceof Failure ? error.code : request.type === 'load' ? 'load' : 'run';
    self.postMessage({ id: request.id, type: 'fail', error: { code, message: describe(error) } });
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
    /** @type {Progress} */
    const progress = {
      file,
      loaded: entry.loaded,
      total: entry.total,
      allLoaded,
      allTotal,
      percent: allTotal > 0 ? (allLoaded / allTotal) * 100 : 100,
    };
    self.postMessage({ id: this.id, type: 'progress', progress });
  }
}

/**
 * Loads the model on the first variant that works in this thread. When the graphics card's
 * variant fails and the processor's remains, answers with a Retry: the page then starts a new
 * thread with `retry` set, which tries only the processor and goes on with the same count.
 * @param {{id: number, manifestUrl: string, manifest: Manifest, device: "auto"|"wasm"|"webgpu",
 *   retry?: {fallback: string, carry: [string, {loaded: number, total: number}][]}}} request
 */
async function load({ id, manifestUrl, manifest, device, retry }) {
  if (model) {
    throw new Failure('load', 'This background thread has already loaded a model.');
  }
  const address = (/** @type {string} */ path) => new URL(path, manifestUrl).href;
  const { engine, data } = manifest;
  const tally = new Tally(id, retry?.carry);
  const modelFolder = `${manifest.model.path}${manifest.model.repo}/`;
  for (const file of [engine.library, engine.loader, engine.executor, data.labels, data.centres]) {
    tally.expect(file.path, file.bytes);
  }

  let library;
  try {
    library = await import(address(engine.library.path));
  } catch (error) {
    throw new Failure('load', `Could not load Transformers.js from ${address(engine.library.path)}: ${describe(error)}`);
  }
  tally.finish(engine.library.path);
  const { env, pipeline } = library;

  const { labels, centres, dims } = await readData(data, address, tally);

  // Every file comes from this build: nothing from Hugging Face or a CDN. The models' folder
  // must be a path, not a full address: the library checks that a local file exists (it decides
  // so whether to load the tokenizer) only for paths, and skips addresses that start with http.
  const models = new URL(manifest.model.path, manifestUrl);
  if (models.origin !== self.location.origin) {
    throw new Failure('load', `The model lies at ${models.origin}, cifarium-worker.js at ${self.location.origin}; they must be on the same site.`);
  }
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = models.pathname;
  // The runtime downloads the executor itself, to report its progress (the library does not)
  // and to keep it in the same store. With wasmBinary set and no "wasm" in wasmPaths, the
  // library leaves the executor alone and ONNX Runtime takes the buffer.
  env.backends.onnx.wasm.wasmBinary = await readExecutor(address(engine.executor.path), engine.executor.path, tally, env.cacheKey);
  env.backends.onnx.wasm.wasmPaths = { mjs: address(engine.loader.path) };

  const { order, fallback } = retry && manifest.variants.cpu
    ? { order: [{ name: /** @type {"cpu"} */ ('cpu'), variant: manifest.variants.cpu }], fallback: retry.fallback }
    : await plan(manifest.variants, device);
  if (order.length === 0) {
    throw new Failure('load', `The model did not start: no variant could be tried${fallback ? `; ${fallback}` : ''}.`);
  }
  for (const [index, { name, variant }] of order.entries()) {
    for (const [file, bytes] of Object.entries(variant.files)) {
      tally.expect(modelFolder + file, bytes);
    }
    let extractor = null;
    try {
      extractor = await pipeline('feature-extraction', manifest.model.repo, {
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
      const trial = await toVector(extractor, manifest.params.prefix + TRIAL_TEXT, manifest.params.pooling);
      scaleToUnit(trial);
      if (trial.length !== dims) {
        throw new Failure('load', `The model gives vectors of ${trial.length} numbers, the centres have ${dims}.`);
      }
    } catch (error) {
      // Any failure of the card's variant, the trial run included, hands over to the processor.
      const next = order[index + 1]?.variant.files;
      if (!next) {
        const before = fallback && name === 'cpu' ? `; before that, ${fallback}` : '';
        throw new Failure('load', `The model did not start: ${name} (${variant.device}, ${variant.dtype}): ${describe(error)}${before}.`);
      }
      for (const file of Object.keys(variant.files)) {
        if (!(file in next)) {
          tally.drop(modelFolder + file);
        }
      }
      const reason = `the graphics card failed: ${describe(error)}`;
      console.warn(`cifarium: ${reason}; trying the processor in a new thread.`);
      throw new Retry(reason, [...tally.files]);
    }

    model = { extractor, labels, centres, dims, pooling: manifest.params.pooling, prefix: manifest.params.prefix };
    tally.finishAll();
    const bytes = [engine.library, engine.loader, engine.executor, data.labels, data.centres]
      .reduce((sum, file) => sum + file.bytes, 0)
      + Object.values(variant.files).reduce((sum, size) => sum + size, 0);
    return {
      labels,
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
 * @returns {Promise<{order: {name: "cpu"|"gpu", variant: Variant}[], fallback: string|null}>}
 */
async function plan(variants, device) {
  /** @type {{name: "cpu"|"gpu", variant: Variant}[]} */
  const order = [];
  let fallback = null;
  if (variants.gpu) {
    if (device === 'wasm') {
      fallback = 'the page asked for the processor';
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
 * @returns {Promise<string|null>}
 */
async function gpuProblem(dtype) {
  const gpu = /** @type {any} */ (navigator).gpu;
  if (!gpu) {
    return 'this browser has no WebGPU here';
  }
  let adapter;
  try {
    // The preference the library itself passes to ONNX Runtime.
    adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (error) {
    return `requestAdapter() failed: ${describe(error)}`;
  }
  if (!adapter) {
    return 'the browser gave no WebGPU adapter';
  }
  if (adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter) {
    return 'the WebGPU adapter is a software one';
  }
  if (HALF_PRECISION.has(dtype) && !adapter.features.has('shader-f16')) {
    return `the adapter has no shader-f16, which ${dtype} weights need`;
  }
  return null;
}

/**
 * Reads the label names and their centres.
 * @param {Manifest["data"]} data
 * @param {(path: string) => string} address
 * @param {Tally} tally
 */
async function readData(data, address, tally) {
  const labelsText = await fetchFile(address(data.labels.path));
  tally.finish(data.labels.path);
  let labels;
  try {
    labels = JSON.parse(new TextDecoder().decode(labelsText));
  } catch {
    throw new Failure('load', `${data.labels.path} is not JSON.`);
  }
  if (!Array.isArray(labels) || labels.length < 2 || !labels.every((label) => typeof label === 'string' && label)
    || new Set(labels).size !== labels.length) {
    throw new Failure('load', `${data.labels.path} must be a list of at least two different label names.`);
  }

  const bytes = await fetchFile(address(data.centres.path));
  tally.finish(data.centres.path);
  const dims = bytes.byteLength / 4 / labels.length;
  if (!Number.isInteger(dims) || dims < 1) {
    throw new Failure('load', `${data.centres.path} has ${bytes.byteLength} bytes, which is not ${labels.length} rows of float32.`);
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
      throw new Failure('load', `The centre of ${JSON.stringify(labels[row])} in ${data.centres.path} has length ${Math.sqrt(sum)}, not 1.`);
    }
  }
  return { labels, centres, dims };
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
 * Downloads a whole file, reporting the bytes read so far.
 * @param {string} href
 * @param {(loaded: number) => void} [onBytes]
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchFile(href, onBytes) {
  let response;
  try {
    response = await fetch(href);
  } catch (error) {
    throw new Failure('load', `Could not download ${href}: ${describe(error)}`);
  }
  if (!response.ok) {
    throw new Failure('load', `Could not download ${href}: HTTP ${response.status}.`);
  }
  if (!onBytes || !response.body) {
    return response.arrayBuffer();
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
}

/**
 * @param {string} text
 * @returns {Promise<{label: string, score: number}>}
 */
async function run(text) {
  if (!model) {
    throw new Failure('not-loaded', 'The model is not loaded.');
  }
  const vector = await toVector(model.extractor, model.prefix + text, model.pooling);
  scaleToUnit(vector);
  if (vector.length !== model.dims) {
    throw new Failure('run', `The model gave a vector of ${vector.length} numbers, the centres have ${model.dims}.`);
  }
  // Labels in their order; the best changes only on a strictly larger score, so a tie goes
  // to the label listed first, as in evaluate.predict.
  let best = -1;
  let bestScore = -Infinity;
  for (let row = 0; row < model.labels.length; row += 1) {
    let score = 0;
    const start = row * model.dims;
    for (let column = 0; column < model.dims; column += 1) {
      score += vector[column] * model.centres[start + column];
    }
    if (score > bestScore) {
      best = row;
      bestScore = score;
    }
  }
  if (best < 0) {
    throw new Failure('run', 'No label got a score.');
  }
  return { label: model.labels[best], score: bestScore };
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

/** @param {unknown} error */
function describe(error) {
  return error instanceof Error ? error.message : String(error);
}
