/**
 * Bench page. Python opens it in Chrome, calls the functions of
 * `window.cifariumBench` through Playwright and reads back what they return.
 */

/**
 * @typedef {Object} GpuInfo
 * @property {string} vendor
 * @property {string} architecture
 * @property {string} device
 * @property {string} description
 * @property {boolean} isFallbackAdapter True for a software adapter.
 * @property {boolean} shaderF16 Required by fp16 and q4f16 weights.
 */

/**
 * @typedef {Object} DeviceInfo
 * @property {string} userAgent
 * @property {number} hardwareConcurrency
 * @property {number|null} deviceMemory Approximate RAM in GB, as the browser reports it.
 * @property {boolean} crossOriginIsolated
 * @property {GpuInfo|null} webgpu null when the browser gave no adapter.
 * @property {string|null} webgpuError Why there is no adapter, if known.
 */

/**
 * @typedef {Object} EngineFiles
 * @property {string} library URL of the Transformers.js build.
 * @property {string} wasmLoader URL of the ONNX Runtime loader (.mjs).
 * @property {string} wasmBinary URL of the ONNX Runtime executor (.wasm).
 */

/**
 * @typedef {Object} EmbedJob
 * @property {EngineFiles} engine
 * @property {string} modelsPath URL of the folder with models, ends with "/".
 * @property {string} model Model id, for example "Xenova/all-MiniLM-L6-v2".
 * @property {string} dtype Variant of the weights, for example "q8".
 * @property {"wasm"|"webgpu"} device Processor or graphics card.
 * @property {string[]} inputs Texts to turn into vectors.
 * @property {"mean"|"cls"|"none"} pooling How token vectors become one vector. "none": the
 *   model pools by itself and returns the finished vector as its `sentence_embedding` output.
 * @property {string} prefix Text put in front of every input, for example "query: ".
 * @property {string} [modelFileName] Graph name without the folder and without the dtype's
 *   suffix, which the library adds: "model" loads onnx/model_quantized.onnx at q8.
 *   Passed as `model_file_name`; the library's default is "model".
 * @property {number} [externalData] Number of `.onnx_data` files next to the graph. Passed
 *   to the library as `use_external_data_format` when above 0; otherwise the library
 *   reads it from `transformers.js_config` in the model's config.json.
 * @property {number} [threads] Threads of ONNX Runtime on the processor. Without it the
 *   runtime picks them itself: half the cores, at most four, on a crossOriginIsolated page,
 *   and one on a page without the isolation headers. 1 measures such a page here.
 */

/** @type {readonly string[]} */
const POOLINGS = ['mean', 'cls', 'none'];

/**
 * @typedef {Object} EmbedResult
 * @property {number} count Number of vectors.
 * @property {number} dims Length of every vector.
 * @property {string} vectors count × dims float32 values, little-endian, in base64. Every
 *   vector is scaled to unit length, so closeness is a plain dot product.
 * @property {number[]} norms Length of every vector before it was scaled.
 * @property {number} loadMs From creating the pipeline to ready, starting with an empty store.
 * @property {number[]} timesMs Time of every call; the first one includes warm-up.
 * @property {string} device What did the computing: "wasm" or "webgpu".
 * @property {number} threads Threads ONNX Runtime uses on the processor.
 * @property {boolean} crossOriginIsolated
 * @property {string[]} cached Paths of the files the library put into the Cache API.
 * @property {string} library Version of Transformers.js.
 */

/**
 * Describes the machine as this page sees it.
 * @returns {Promise<DeviceInfo>}
 */
async function deviceInfo() {
  /** @type {DeviceInfo} */
  const info = {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemory: navigator.deviceMemory ?? null,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: null,
    webgpuError: null,
  };
  if (!('gpu' in navigator)) {
    info.webgpuError = 'navigator.gpu is missing';
    return info;
  }
  try {
    // The same preference Transformers.js passes to ONNX Runtime, so this is
    // the adapter the engine itself would get.
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      info.webgpuError = 'requestAdapter() returned null';
      return info;
    }
    const details = adapter.info ?? {};
    info.webgpu = {
      vendor: details.vendor ?? '',
      architecture: details.architecture ?? '',
      device: details.device ?? '',
      description: details.description ?? '',
      isFallbackAdapter: Boolean(details.isFallbackAdapter ?? adapter.isFallbackAdapter),
      shaderF16: adapter.features.has('shader-f16'),
    };
  } catch (error) {
    info.webgpuError = String(error);
  }
  return info;
}

/**
 * Loads a model from this server and turns every input into a vector of unit length.
 * @param {EmbedJob} job
 * @returns {Promise<EmbedResult>}
 */
async function embed(job) {
  if (job.inputs.length === 0) {
    throw new Error('The job has no inputs.');
  }
  if (!POOLINGS.includes(job.pooling)) {
    throw new Error(`Unknown pooling ${JSON.stringify(job.pooling)}; expected ${POOLINGS.join(', ')}.`);
  }
  if (job.threads !== undefined && !(Number.isInteger(job.threads) && job.threads >= 1)) {
    throw new Error(`threads must be a whole number from 1, got ${JSON.stringify(job.threads)}.`);
  }
  const prefix = job.prefix ?? '';
  const { env, pipeline } = await import(job.engine.library);

  // Every file comes from this server: nothing from Hugging Face or a CDN.
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = job.modelsPath;
  env.backends.onnx.wasm.wasmPaths = { mjs: job.engine.wasmLoader, wasm: job.engine.wasmBinary };
  if (job.threads !== undefined) {
    // ONNX Runtime reads this once, when it starts on the first model, and picks its own
    // number only if none is set. A new tab per job means every job starts it afresh.
    env.backends.onnx.wasm.numThreads = job.threads;
  }

  // Empty the library's store, so that what follows is a first load.
  await caches.delete(env.cacheKey);

  report(`loading ${job.model} (${job.dtype}, ${job.device}, pooling ${job.pooling})`);
  const loadStart = performance.now();
  /** @type {Record<string, unknown>} */
  const options = { dtype: job.dtype, device: job.device };
  if (job.modelFileName) {
    options.model_file_name = job.modelFileName;
  }
  if ((job.externalData ?? 0) > 0) {
    options.use_external_data_format = job.externalData;
  }
  const extractor = await pipeline('feature-extraction', job.model, options);
  const loadMs = performance.now() - loadStart;

  report(`computing ${job.inputs.length} vectors`);
  /** @type {number[]} */
  const timesMs = [];
  /** @type {Float32Array[]} */
  const rows = [];
  /** @type {number[]} */
  const norms = [];
  for (const input of job.inputs) {
    const start = performance.now();
    const vector = await toVector(extractor, prefix + input, job.pooling);
    const norm = scaleToUnit(vector);
    timesMs.push(performance.now() - start);
    if (rows.length > 0 && vector.length !== rows[0].length) {
      throw new Error(`Vector ${rows.length} has ${vector.length} numbers, the first one ${rows[0].length}.`);
    }
    rows.push(vector);
    norms.push(norm);
  }

  const dims = rows[0].length;
  const vectors = new Float32Array(rows.length * dims);
  rows.forEach((row, index) => vectors.set(row, index * dims));

  const store = await caches.open(env.cacheKey);
  const cached = (await store.keys()).map((request) => new URL(request.url).pathname);

  return {
    count: rows.length,
    dims,
    // Playwright returns only what converts to JSON, hence base64.
    vectors: toBase64(new Uint8Array(vectors.buffer)),
    norms,
    loadMs,
    timesMs,
    device: job.device,
    threads: env.backends.onnx.wasm.numThreads,
    crossOriginIsolated: self.crossOriginIsolated,
    cached,
    library: env.version,
  };
}

/**
 * One text to one vector, not yet scaled.
 * @param {any} extractor A "feature-extraction" pipeline.
 * @param {string} text Input with the prefix already in front.
 * @param {string} pooling
 * @returns {Promise<Float32Array>}
 */
async function toVector(extractor, text, pooling) {
  if (pooling !== 'none') {
    const output = await extractor(text, { pooling, normalize: false });
    return Float32Array.from(output.data);
  }
  // The pipeline would return the token vectors (last_hidden_state) here. Models
  // that pool inside the graph give the finished vector as a separate output, so
  // the tokenizer and the model are called directly, with the pipeline's options.
  const inputs = extractor.tokenizer(text, { padding: true, truncation: true });
  const outputs = await extractor.model(inputs);
  const vector = outputs.sentence_embedding;
  if (!vector) {
    throw new Error(`pooling "none" needs a sentence_embedding output; the model has ${Object.keys(outputs).join(', ')}.`);
  }
  if (vector.dims.length !== 2 || vector.dims[0] !== 1) {
    throw new Error(`sentence_embedding has shape [${vector.dims.join(', ')}], expected [1, size].`);
  }
  return Float32Array.from(vector.data);
}

/**
 * Scales a vector to unit length in place and returns its length before that.
 * @param {Float32Array} vector
 * @returns {number}
 */
function scaleToUnit(vector) {
  let sum = 0;
  for (const value of vector) {
    sum += value * value;
  }
  const norm = Math.sqrt(sum);
  if (!(norm > 0) || !Number.isFinite(norm)) {
    throw new Error(`The model returned a vector of length ${norm}.`);
  }
  for (let index = 0; index < vector.length; index += 1) {
    vector[index] /= norm;
  }
  return norm;
}

/**
 * Sends a line of progress to Python, which prints it in the terminal.
 * @param {string} message
 */
function report(message) {
  window.cifariumProgress?.(message);
}

/**
 * @param {Uint8Array} bytes
 * @returns {string}
 */
function toBase64(bytes) {
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

/**
 * @param {unknown} error
 * @returns {string}
 */
function describe(error) {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

const bench = {
  deviceInfo,

  /**
   * Starts a job and returns at once. Python then waits, with a time limit,
   * until `outcome` appears.
   * @param {EmbedJob} job
   */
  start(job) {
    bench.outcome = undefined;
    embed(job).then(
      (result) => {
        bench.outcome = { result };
      },
      (error) => {
        bench.outcome = { error: describe(error) };
      },
    );
  },

  /** @type {{result: EmbedResult}|{error: string}|undefined} */
  outcome: undefined,
};

// Python waits for this object to appear: it is the page's "ready" signal.
window.cifariumBench = bench;
