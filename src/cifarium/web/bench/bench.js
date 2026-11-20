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
 * @property {"embed"} operation Texts to vectors (recipe embed-centroid).
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

/**
 * @typedef {Object} ZeroShotJob
 * @property {"zero-shot"} operation The score of every label for every text (recipe zero-shot).
 * @property {EngineFiles} engine
 * @property {string} modelsPath URL of the folder with models, ends with "/".
 * @property {string} model Model id, for example "Xenova/nli-deberta-v3-xsmall".
 * @property {string} dtype Variant of the weights, for example "q8".
 * @property {"wasm"|"webgpu"} device Processor or graphics card.
 * @property {string[]} inputs Texts to score.
 * @property {string[]} hypotheses One sentence per label, in the order of the labels: what the
 *   model checks against the text ("This example is about a lost card."). Python makes them;
 *   the page passes each one whole, so nothing is put together here.
 * @property {string} [modelFileName] As in EmbedJob.
 * @property {number} [externalData] As in EmbedJob.
 * @property {number} [threads] As in EmbedJob.
 */

/**
 * @typedef {Object} ZeroShotResult
 * @property {number} count Number of texts.
 * @property {number} labels Number of hypotheses.
 * @property {number[][]} scores One row per text, one score per hypothesis in their order: the
 *   softmax of the entailment logits over the labels, so a row sums to 1.
 * @property {number} loadMs As in EmbedResult.
 * @property {number[]} timesMs Time of every text, all of its labels; the first includes warm-up.
 * @property {string} device As in EmbedResult.
 * @property {number} threads As in EmbedResult.
 * @property {boolean} crossOriginIsolated
 * @property {string[]} cached As in EmbedResult.
 * @property {string} library Version of Transformers.js.
 */

/** @type {readonly string[]} */
const POOLINGS = ['mean', 'cls', 'none'];

/** @type {readonly string[]} */
const OPERATIONS = ['embed', 'zero-shot'];

/**
 * The library puts each label into `hypothesis_template` with `String.prototype.replace`, which
 * reads `$&`, `$1` and the like in what it inserts. With the template "{}" and every "$" doubled,
 * what comes out is the hypothesis as Python wrote it.
 */
const WHOLE = '{}';

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
  if (!POOLINGS.includes(job.pooling)) {
    throw new Error(`Unknown pooling ${JSON.stringify(job.pooling)}; expected ${POOLINGS.join(', ')}.`);
  }
  const prefix = job.prefix ?? '';
  const { env, pipe: extractor, loadMs } = await load('feature-extraction', job, `pooling ${job.pooling}`);

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

  return {
    count: rows.length,
    dims,
    // Playwright returns only what converts to JSON, hence base64.
    vectors: toBase64(new Uint8Array(vectors.buffer)),
    norms,
    loadMs,
    timesMs,
    ...(await machine(env, job)),
  };
}

/**
 * Loads an NLI model and scores every hypothesis against every input, the way the library's
 * zero-shot-classification pipeline does it: one run of the model per text and label, then the
 * softmax of the entailment logits over the labels (`multi_label: false`).
 * @param {ZeroShotJob} job
 * @returns {Promise<ZeroShotResult>}
 */
async function zeroShot(job) {
  const hypotheses = job.hypotheses;
  if (!Array.isArray(hypotheses) || hypotheses.length < 2 || !hypotheses.every((h) => typeof h === 'string' && h)) {
    throw new Error('The job needs at least two hypotheses, each a text that is not empty.');
  }
  const passed = hypotheses.map((hypothesis) => hypothesis.replace(/\$/g, '$$$$'));
  passed.forEach((label, index) => {
    // The same call the library makes; if a later version builds the sentence another way,
    // this says so instead of scoring a sentence nobody wrote.
    if (WHOLE.replace('{}', label) !== hypotheses[index]) {
      throw new Error(`The library would check ${JSON.stringify(WHOLE.replace('{}', label))} instead of hypothesis ${index}.`);
    }
  });
  const { env, pipe: classifier, loadMs } = await load('zero-shot-classification', job, `${hypotheses.length} labels`);
  // Without "entailment" among its labels the library takes logit 2 and says so only in the console.
  const names = Object.keys(classifier.model.config.label2id ?? {});
  if (!names.some((name) => name.toLowerCase() === 'entailment')) {
    throw new Error(`The model has no "entailment" label (it has ${names.join(', ') || 'none'}): it cannot check hypotheses.`);
  }

  report(`scoring ${job.inputs.length} texts against ${hypotheses.length} labels`);
  /** @type {number[]} */
  const timesMs = [];
  /** @type {number[][]} */
  const scores = [];
  for (const input of job.inputs) {
    const start = performance.now();
    const output = await classifier(input, passed, { hypothesis_template: WHOLE, multi_label: false });
    timesMs.push(performance.now() - start);
    // The library sorts the labels by score; this puts them back in the order of the job. Equal
    // hypotheses get equal logits, so one score per sentence is right for each of them.
    /** @type {Map<string, number>} */
    const scoreOf = new Map(output.labels.map((label, index) => [label, output.scores[index]]));
    const row = passed.map((label) => scoreOf.get(label));
    if (!row.every((score) => Number.isFinite(score))) {
      throw new Error(`The model returned scores ${JSON.stringify(row)} for text ${scores.length}.`);
    }
    scores.push(/** @type {number[]} */ (row));
  }

  return {
    count: scores.length,
    labels: hypotheses.length,
    scores,
    loadMs,
    timesMs,
    ...(await machine(env, job)),
  };
}

/**
 * Checks what every job has, points the library at this server, empties its store and loads
 * the model into a pipeline of `task`.
 * @param {string} task Name of the library's pipeline.
 * @param {EmbedJob|ZeroShotJob} job
 * @param {string} what Words for the line of progress.
 * @returns {Promise<{env: any, pipe: any, loadMs: number}>}
 */
async function load(task, job, what) {
  if (!Array.isArray(job.inputs) || job.inputs.length === 0) {
    throw new Error('The job has no inputs.');
  }
  if (job.threads !== undefined && !(Number.isInteger(job.threads) && job.threads >= 1)) {
    throw new Error(`threads must be a whole number from 1, got ${JSON.stringify(job.threads)}.`);
  }
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

  report(`loading ${job.model} (${job.dtype}, ${job.device}, ${what})`);
  const loadStart = performance.now();
  /** @type {Record<string, unknown>} */
  const options = { dtype: job.dtype, device: job.device };
  if (job.modelFileName) {
    options.model_file_name = job.modelFileName;
  }
  if ((job.externalData ?? 0) > 0) {
    options.use_external_data_format = job.externalData;
  }
  const pipe = await pipeline(task, job.model, options);
  return { env, pipe, loadMs: performance.now() - loadStart };
}

/**
 * What every result says about where it was computed.
 * @param {any} env The library's settings.
 * @param {EmbedJob|ZeroShotJob} job
 */
async function machine(env, job) {
  const store = await caches.open(env.cacheKey);
  const cached = (await store.keys()).map((request) => new URL(request.url).pathname);
  return {
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
   * @param {EmbedJob|ZeroShotJob} job
   */
  start(job) {
    bench.outcome = undefined;
    run(job).then(
      (result) => {
        bench.outcome = { result };
      },
      (error) => {
        bench.outcome = { error: describe(error) };
      },
    );
  },

  /** @type {{result: EmbedResult|ZeroShotResult}|{error: string}|undefined} */
  outcome: undefined,
};

/**
 * The operation the job names.
 * @param {EmbedJob|ZeroShotJob} job
 * @returns {Promise<EmbedResult|ZeroShotResult>}
 */
async function run(job) {
  if (job.operation === 'embed') {
    return embed(/** @type {EmbedJob} */ (job));
  }
  if (job.operation === 'zero-shot') {
    return zeroShot(/** @type {ZeroShotJob} */ (job));
  }
  throw new Error(`Unknown operation ${JSON.stringify(job.operation)}; expected ${OPERATIONS.join(', ')}.`);
}

// Python waits for this object to appear: it is the page's "ready" signal.
window.cifariumBench = bench;
