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
 */

/**
 * @typedef {Object} EmbedResult
 * @property {number} count Number of vectors.
 * @property {number} dims Length of every vector.
 * @property {string} vectors count × dims float32 values, little-endian, in base64.
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
 * Loads a model from this server and turns every input into a vector.
 * @param {EmbedJob} job
 * @returns {Promise<EmbedResult>}
 */
async function embed(job) {
  if (job.inputs.length === 0) {
    throw new Error('The job has no inputs.');
  }
  const { env, pipeline } = await import(job.engine.library);

  // Every file comes from this server: nothing from Hugging Face or a CDN.
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = job.modelsPath;
  env.backends.onnx.wasm.wasmPaths = { mjs: job.engine.wasmLoader, wasm: job.engine.wasmBinary };

  // Empty the library's store, so that what follows is a first load.
  await caches.delete(env.cacheKey);

  report(`loading ${job.model} (${job.dtype}, ${job.device})`);
  const loadStart = performance.now();
  const extractor = await pipeline('feature-extraction', job.model, {
    dtype: job.dtype,
    device: job.device,
  });
  const loadMs = performance.now() - loadStart;

  report(`computing ${job.inputs.length} vectors`);
  /** @type {number[]} */
  const timesMs = [];
  /** @type {Float32Array[]} */
  const rows = [];
  for (const text of job.inputs) {
    const start = performance.now();
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    timesMs.push(performance.now() - start);
    rows.push(Float32Array.from(output.data));
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
