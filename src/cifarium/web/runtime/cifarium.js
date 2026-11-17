/**
 * cifarium runtime: runs the model of a cifarium build on a web page.
 *
 *     import { createCifarium } from './cifarium.js';
 *     const model = await createCifarium('./cifarium.json');
 *     await model.load({ onProgress: (p) => bar.value = p.percent });
 *     const { label, score, scores } = await model.run('Where is my parcel?');
 *
 * The model computes in a background thread (cifarium-worker.js, next to this file), on the
 * graphics card when the build has a variant for it and the browser offers an adapter, and on
 * the processor otherwise or when the card fails. Nothing is sent anywhere: the runtime only
 * reads the files of its own folder. ARCHITECTURE.md, section 8, describes every call, and
 * section 14 says which parts of it are a promise to the developer.
 */

/** Version of this runtime; the same as the cifarium package it comes with. */
export const VERSION = '0.0.1';

/**
 * The newest manifest format this runtime reads (`format` in cifarium.json). It also reads every
 * older one; the first format is 1, so today that is the one.
 */
export const MANIFEST_FORMAT = 1;

/**
 * Every `code` a CifariumError has today. A later version may add codes: treat one you do not
 * know as a plain failure. A code is never renamed or given another meaning.
 */
export const ERROR_CODES = Object.freeze([
  'manifest', 'unsupported', 'input', 'not-loaded', 'network', 'load', 'run', 'disposed',
]);

/**
 * Every `reason` of `info().fallback` today. New reasons may be added; show an unknown one with
 * its `message`.
 */
export const FALLBACK_REASONS = Object.freeze([
  'no-webgpu', 'no-adapter', 'software-adapter', 'no-shader-f16', 'requested', 'failed',
]);

/** Every `phase` of a progress event today. New phases may be added. */
export const PROGRESS_PHASES = Object.freeze(['download', 'prepare']);

const TASKS = { classify: ['embed-centroid'] };
const POOLINGS = ['mean', 'cls', 'none'];
const DEVICES = ['auto', 'wasm', 'webgpu'];
/** Which `device` each variant of the manifest must have. */
const VARIANT_DEVICES = { cpu: 'wasm', gpu: 'webgpu' };

/**
 * @typedef {Object} FileRef A file of the build.
 * @property {string} path Relative to the manifest's address.
 * @property {number} bytes Size on disk, before any compression by the hosting.
 */

/**
 * @typedef {Object} Variant One way to run the model.
 * @property {"wasm"|"webgpu"} device Processor or graphics card.
 * @property {string} dtype Weights of the graph, for example "q8".
 * @property {string} graph Graph name without "onnx/" and without the dtype's suffix, which the
 *   library adds: "model" at q8 is onnx/model_quantized.onnx.
 * @property {number} externalData Number of `.onnx_data` files next to the graph.
 * @property {Record<string, number>} files Every file of the model this variant reads, relative
 *   to the model's folder, with its size in bytes.
 */

/**
 * @typedef {Object} Label One class of a classification.
 * @property {string} name
 * @property {string|null} description What the settings say about it; null when nothing.
 */

/**
 * @typedef {Object} Manifest cifarium.json, format 1.
 * @property {number} format
 * @property {string} cifarium Version of cifarium that made the build.
 * @property {string} task
 * @property {string} recipe
 * @property {Label[]} labels Only for the task "classify", in the order of the settings.
 * @property {{repo: string, revision: string, license: string, path: string}} model `path` is
 *   the folder the library reads models from; the files lie in `<path><repo>/`.
 * @property {{cpu?: Variant, gpu?: Variant}} variants
 * @property {{pooling: "mean"|"cls"|"none", prefix: string}} params
 * @property {{transformers: string, onnxruntimeWeb: string, library: FileRef, loader: FileRef,
 *   executor: FileRef}} engine
 * @property {{centres: FileRef}} data The files of the recipe; "embed-centroid" has the centres.
 */

/**
 * @typedef {Object} Progress What `onProgress` receives, once per step of any file.
 * @property {"download"|"prepare"} phase "download" while bytes are still coming. "prepare" once
 *   every byte of this load is in and the session and the trial run are left: one event with
 *   `percent` 100 opens it, and it is the last of a load. If the graphics card fails and the
 *   processor's files are added, "download" may come again once.
 * @property {string} file Path of the file relative to the manifest. In "prepare" it is the
 *   last file that arrived.
 * @property {number} loaded Bytes of this file read so far.
 * @property {number} total Bytes of this file.
 * @property {number} allLoaded Bytes of every file of this load read so far.
 * @property {number} allTotal Bytes of every file of this load. It grows if the graphics
 *   card fails and the processor's files are added, so `percent` can step back once.
 * @property {number} percent allLoaded / allTotal, from 0 to 100. The last call has 100.
 */

/**
 * @typedef {Object} LoadOptions
 * @property {(progress: Progress) => void} [onProgress]
 * @property {"auto"|"wasm"|"webgpu"} [device] "auto" (the default): the graphics card if the
 *   build has a variant for it and the browser offers a real adapter, else the processor.
 *   "wasm": the processor only. "webgpu": the card's variant without checking the adapter;
 *   if it fails, the processor still takes over.
 *
 * The name `signal` (an AbortSignal) is reserved for load() and run(); today it is ignored.
 */

/**
 * @typedef {Object} Fallback Why the graphics card's variant of the build is not in use.
 * @property {"no-webgpu"|"no-adapter"|"software-adapter"|"no-shader-f16"|"requested"|"failed"} reason
 *   For a program. "no-webgpu": the browser has no WebGPU here (no `navigator.gpu`).
 *   "no-adapter": it gave no adapter. "software-adapter": the adapter is a software one, slower
 *   than the processor. "no-shader-f16": the adapter lacks what half-precision weights need.
 *   "requested": the page asked for the processor. "failed": the card failed to start the
 *   model or on the trial run, and the processor took over.
 * @property {string} message For a person: the same in a sentence, with the text of the error.
 */

/**
 * @typedef {Object} Info
 * @property {"created"|"loading"|"ready"|"failed"|"disposed"} state
 * @property {string} runtime Version of this runtime.
 * @property {string} cifarium Version of cifarium that made the build.
 * @property {number} format Manifest format.
 * @property {string} task
 * @property {string} recipe
 * @property {string} model Model on Hugging Face, for example "Xenova/all-MiniLM-L6-v2".
 * @property {string} revision Commit of the model.
 * @property {string} license
 * @property {string} transformers Version of Transformers.js.
 * @property {string} onnxruntimeWeb Version of onnxruntime-web.
 * @property {Label[]} labels In the order of the settings; there from createCifarium on.
 * @property {"cpu"|"gpu"|null} variant The variant of the manifest in use; null until loaded.
 * @property {"wasm"|"webgpu"|null} device
 * @property {string|null} dtype
 * @property {number|null} threads Threads of ONNX Runtime on the processor.
 * @property {boolean|null} crossOriginIsolated Of the background thread: without it, one thread.
 * @property {number|null} bytes What a first visit downloads for the variant in use: engine,
 *   model and data, before compression.
 * @property {Fallback|null} fallback Why the graphics card's variant is not in use, when the
 *   manifest has one; null when it is in use or there is none.
 */

/**
 * @typedef {Object} Score
 * @property {string} label
 * @property {number} score
 */

/**
 * @typedef {Object} Result What run() answers for a classification.
 * @property {string} label The best label: the same as `scores[0].label`.
 * @property {number} score Its score. A larger score is closer; the scale depends on `recipe`
 *   (for "embed-centroid" it is the dot product of unit vectors, from -1 to 1), so compare
 *   scores of one build with each other and never with a number from another build.
 * @property {Score[]} scores Every label, the best first. Equal scores keep the order of the
 *   settings.
 */

/**
 * The only kind of error the runtime throws; `code` is one of ERROR_CODES:
 * "manifest" (cifarium.json is not JSON, lacks fields, or the server says it is not there),
 * "unsupported" (a manifest of a newer format, an unknown task or recipe), "input" (a wrong
 * argument, an empty text), "not-loaded" (run() before load()), "network" (a file did not
 * download in a way a new try may fix: no connection, a dropped transfer, HTTP 408, 425, 429 or
 * 5xx), "load" (the model did not start on any device, a file is missing or wrong, the thread
 * failed), "run" (the model failed on a text), "disposed" (the instance was disposed).
 * `cause` is the error the runtime caught, when there was one; it is absent for an error the
 * runtime raised itself.
 */
export class CifariumError extends Error {
  /**
   * @param {string} code
   * @param {string} message
   * @param {unknown} [cause]
   */
  constructor(code, message, cause) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'CifariumError';
    this.code = code;
  }
}

/**
 * Reads the manifest and checks that this runtime can run it. Downloads nothing else.
 * @param {string | URL} manifestUrl Address of cifarium.json, relative to the page.
 * @returns {Promise<Cifarium>}
 */
export async function createCifarium(manifestUrl) {
  let url;
  try {
    url = new URL(manifestUrl, globalThis.location?.href).href;
  } catch (error) {
    throw new CifariumError('input', `${JSON.stringify(String(manifestUrl))} is not an address.`, error);
  }
  let text;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new CifariumError(retryable(response.status) ? 'network' : 'manifest',
        `Could not fetch ${url}: HTTP ${response.status}.`);
    }
    text = await response.text();
  } catch (error) {
    if (error instanceof CifariumError) {
      throw error;
    }
    throw new CifariumError('network', `Could not fetch ${url}: ${describe(error)}`, error);
  }
  let manifest;
  try {
    manifest = JSON.parse(text);
  } catch (error) {
    throw new CifariumError('manifest', `${url} is not JSON.`, error);
  }
  checkManifest(manifest, url);
  return new Cifarium(url, manifest);
}

class Cifarium {
  /** @type {string} */ #url;
  /** @type {Manifest} */ #manifest;
  /** @type {Info["state"]} */ #state = 'created';
  /** @type {Worker|null} */ #worker = null;
  /** @type {Promise<Info>|null} */ #loading = null;
  /** @type {any} What the background thread reported after loading. */ #loaded = null;
  /** @type {Map<number, {resolve: Function, reject: Function, onProgress?: Function}>} */
  #pending = new Map();
  #nextId = 1;

  /**
   * @param {string} url
   * @param {Manifest} manifest
   */
  constructor(url, manifest) {
    this.#url = url;
    this.#manifest = manifest;
  }

  /**
   * Starts the background thread, picks the device and downloads the engine, the model and
   * the data. A second call while loading or after success returns the same promise; after a
   * failure, a "network" one included, it starts again.
   * @param {LoadOptions} [options]
   * @returns {Promise<Info>}
   */
  load(options = {}) {
    if (this.#state === 'disposed') {
      return Promise.reject(disposed());
    }
    if (this.#loading && this.#state !== 'failed') {
      return this.#loading;
    }
    const { onProgress, device = 'auto' } = options;
    if (!DEVICES.includes(device)) {
      return Promise.reject(new CifariumError('input',
        `device must be one of ${DEVICES.join(', ')}, got ${JSON.stringify(device)}.`));
    }
    if (onProgress !== undefined && typeof onProgress !== 'function') {
      return Promise.reject(new CifariumError('input', 'onProgress must be a function.'));
    }
    this.#loading = this.#start(device, onProgress);
    return this.#loading;
  }

  /**
   * Names the label of a text and scores every label. Calls may overlap: they run one after
   * another, in order. A call made while load() is still running waits for it. A text that is
   * empty or only spaces is an "input" error. A long text is not refused: the model's tokenizer
   * cuts it at its own limit (512 tokens for all-MiniLM-L6-v2) and the rest counts for nothing.
   * @param {string} text
   * @returns {Promise<Result>}
   */
  async run(text) {
    if (this.#state === 'disposed') {
      throw disposed();
    }
    if (typeof text !== 'string') {
      throw new CifariumError('input', `run() takes a string, got ${text === null ? 'null' : typeof text}.`);
    }
    if (text.trim() === '') {
      throw new CifariumError('input', 'run() needs a text with something in it, not an empty one or only spaces.');
    }
    if (!this.#loading) {
      throw new CifariumError('not-loaded', 'Call load() before run().');
    }
    await this.#loading;
    if (this.#state === 'disposed') {
      throw disposed();
    }
    if (!this.#worker) {
      throw new CifariumError('load', 'The background thread stopped. Call load() again.');
    }
    return this.#request(this.#worker, { type: 'run', text });
  }

  /**
   * What the build is and how it runs here. Fields about the device stay null until loaded;
   * the labels are there from the start.
   * @returns {Info}
   */
  info() {
    const manifest = this.#manifest;
    const loaded = this.#loaded;
    return {
      state: this.#state,
      runtime: VERSION,
      cifarium: manifest.cifarium,
      format: manifest.format,
      task: manifest.task,
      recipe: manifest.recipe,
      model: manifest.model.repo,
      revision: manifest.model.revision,
      license: manifest.model.license,
      transformers: manifest.engine.transformers,
      onnxruntimeWeb: manifest.engine.onnxruntimeWeb,
      labels: manifest.labels.map(({ name, description }) => ({ name, description })),
      variant: loaded?.variant ?? null,
      device: loaded?.device ?? null,
      dtype: loaded?.dtype ?? null,
      threads: loaded?.threads ?? null,
      crossOriginIsolated: loaded?.crossOriginIsolated ?? null,
      bytes: loaded?.bytes ?? null,
      fallback: loaded?.fallback ? { ...loaded.fallback } : null,
    };
  }

  /**
   * Stops the background thread, which frees the model's memory. Calls still waiting fail
   * with "disposed", and so does every later call except info().
   * @returns {Promise<void>}
   */
  async dispose() {
    if (this.#state === 'disposed') {
      return;
    }
    this.#state = 'disposed';
    this.#stop(disposed());
  }

  /**
   * @param {LoadOptions["device"]} device
   * @param {LoadOptions["onProgress"]} onProgress
   * @returns {Promise<Info>}
   */
  async #start(device, onProgress) {
    this.#state = 'loading';
    this.#loaded = null;
    /** @type {{fallback: Fallback, carry: any[]}|undefined} */
    let retry;
    for (;;) {
      const worker = this.#spawn();
      try {
        this.#loaded = await this.#request(
          worker,
          { type: 'load', manifestUrl: this.#url, manifest: this.#manifest, device, retry },
          onProgress,
        );
        break;
      } catch (error) {
        if (error instanceof RetryOnProcessor && !retry && this.#state === 'loading') {
          // After a failed session the engine cannot be trusted in that thread.
          worker.terminate();
          retry = { fallback: error.fallback, carry: error.carry };
          continue;
        }
        if (this.#state !== 'disposed') {
          this.#state = 'failed';
          this.#stop(error);
        }
        throw error instanceof RetryOnProcessor
          ? new CifariumError('load', `The model did not start: ${error.fallback.message}.`)
          : error;
      }
    }
    this.#state = 'ready';
    return this.info();
  }

  /** Starts a background thread and makes it the current one. */
  #spawn() {
    const worker = new Worker(new URL('./cifarium-worker.js', import.meta.url), {
      type: 'module',
      name: 'cifarium',
    });
    this.#worker = worker;
    worker.onmessage = (event) => this.#receive(event.data);
    worker.onerror = (event) => {
      event.preventDefault();
      // A thread whose script did not load gives no message, and the browser does not say
      // whether the network or the file is to blame: that is "load".
      const reason = event.message || 'its script did not load';
      this.#stop(new CifariumError('load', `The background thread failed: ${reason}`));
    };
    return worker;
  }

  /**
   * Ends the background thread and fails every call that waits for it.
   * @param {unknown} error
   */
  #stop(error) {
    this.#worker?.terminate();
    this.#worker = null;
    const waiting = [...this.#pending.values()];
    this.#pending.clear();
    for (const call of waiting) {
      call.reject(error);
    }
    if (this.#state === 'loading' || this.#state === 'ready') {
      this.#state = 'failed';
    }
  }

  /**
   * @param {Worker} worker
   * @param {object} message
   * @param {Function} [onProgress]
   * @returns {Promise<any>}
   */
  #request(worker, message, onProgress) {
    return new Promise((resolve, reject) => {
      const id = this.#nextId++;
      this.#pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ id, ...message });
    });
  }

  /** @param {{id: number, type: string, value?: any, progress?: Progress, fallback?: Fallback, carry?: any[], error?: {code: string, message: string, cause?: {name: string, message: string, stack?: string}}}} message */
  #receive(message) {
    const call = this.#pending.get(message.id);
    if (!call) {
      return;
    }
    if (message.type === 'progress') {
      try {
        call.onProgress?.(message.progress);
      } catch (error) {
        console.error('cifarium: onProgress threw', error);
      }
      return;
    }
    this.#pending.delete(message.id);
    if (message.type === 'done') {
      call.resolve(message.value);
    } else if (message.type === 'retry') {
      call.reject(new RetryOnProcessor(/** @type {Fallback} */ (message.fallback), message.carry ?? []));
    } else {
      call.reject(new CifariumError(
        message.error?.code ?? 'run',
        message.error?.message ?? 'Unknown error.',
        revive(message.error?.cause),
      ));
    }
  }
}

/** Internal: the graphics card failed in a thread, the processor is to be tried in a new one. */
class RetryOnProcessor extends Error {
  /**
   * @param {Fallback} fallback
   * @param {any[]} carry
   */
  constructor(fallback, carry) {
    super(fallback.message);
    this.fallback = fallback;
    this.carry = carry;
  }
}

function disposed() {
  return new CifariumError('disposed', 'This cifarium instance was disposed.');
}

/**
 * An error that crossed from the background thread as plain fields, made an Error again.
 * @param {{name: string, message: string, stack?: string}|undefined} plain
 * @returns {Error|undefined}
 */
function revive(plain) {
  if (!plain) {
    return undefined;
  }
  const error = new Error(plain.message);
  error.name = plain.name || 'Error';
  if (plain.stack) {
    error.stack = plain.stack;
  }
  return error;
}

/**
 * Whether a new try may help when the server answered with this status.
 * @param {number} status
 */
function retryable(status) {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

/**
 * Throws a CifariumError unless this runtime can run the manifest.
 * @param {any} manifest
 * @param {string} url
 */
function checkManifest(manifest, url) {
  const where = `The manifest ${url}`;
  if (!isObject(manifest)) {
    throw new CifariumError('manifest', `${where} is not a JSON object.`);
  }
  if (!Number.isInteger(manifest.format) || manifest.format < 1) {
    throw new CifariumError('manifest', `${where} has no format number (found ${JSON.stringify(manifest.format)}).`);
  }
  if (manifest.format > MANIFEST_FORMAT) {
    // A runtime reads its own format and every older one. When format 2 exists, the reader of
    // format 1 stays here next to it.
    throw new CifariumError('unsupported',
      `${where} has format ${manifest.format}; this runtime (${VERSION}) reads formats up to ` +
      `${MANIFEST_FORMAT}. Use the cifarium.js and cifarium-worker.js from the same build.`);
  }
  const recipes = Object.hasOwn(TASKS, manifest.task) ? TASKS[manifest.task] : null;
  if (!recipes) {
    throw new CifariumError('unsupported',
      `${where} is for task ${JSON.stringify(manifest.task)}; this runtime (${VERSION}) runs ` +
      `${Object.keys(TASKS).join(', ')}.`);
  }
  if (!recipes.includes(manifest.recipe)) {
    throw new CifariumError('unsupported',
      `${where} uses recipe ${JSON.stringify(manifest.recipe)} for ${manifest.task}; this ` +
      `runtime (${VERSION}) knows ${recipes.join(', ')}.`);
  }

  /** @type {string[]} */
  const problems = [];
  const need = (ok, what) => {
    if (!ok) problems.push(what);
  };
  const isFile = (value) => isObject(value) && isText(value.path) && isCount(value.bytes);
  need(isText(manifest.cifarium), 'cifarium: a version');
  const labels = manifest.labels;
  need(Array.isArray(labels) && labels.length >= 2
    && labels.every((label) => isObject(label) && isText(label.name)
      && (label.description === null || typeof label.description === 'string'))
    && new Set(labels.map((label) => label?.name)).size === labels.length,
  'labels: at least two different names, each with a description or null');
  const model = manifest.model;
  need(isObject(model) && isText(model.repo) && isText(model.revision) && isText(model.path)
    && typeof model.license === 'string', 'model: repo, revision, license and path');
  const params = manifest.params;
  need(isObject(params) && POOLINGS.includes(params.pooling) && typeof params.prefix === 'string',
    `params: pooling (${POOLINGS.join(', ')}) and prefix`);
  const engine = manifest.engine;
  need(isObject(engine) && isText(engine.transformers) && isText(engine.onnxruntimeWeb)
    && isFile(engine.library) && isFile(engine.loader) && isFile(engine.executor),
    'engine: transformers, onnxruntimeWeb, library, loader and executor');
  // The files of `data` depend on the recipe; this one needs the centres.
  const data = manifest.data;
  need(isObject(data) && isFile(data.centres), 'data: centres');
  const variants = manifest.variants;
  const named = isObject(variants) ? Object.keys(VARIANT_DEVICES).filter((name) => name in variants) : [];
  need(named.length > 0, 'variants: cpu or gpu');
  for (const name of named) {
    const variant = variants[name];
    need(isObject(variant) && variant.device === VARIANT_DEVICES[name] && isText(variant.dtype)
      && isText(variant.graph) && isCount(variant.externalData) && isObject(variant.files)
      && Object.values(variant.files).every(isCount),
      `variants.${name}: device "${VARIANT_DEVICES[name]}", dtype, graph, externalData and files`);
  }
  if (problems.length > 0) {
    throw new CifariumError('manifest', `${where} is missing or has wrong ${problems.join('; ')}.`);
  }
}

/** @param {unknown} value */
function isObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** @param {unknown} value */
function isText(value) {
  return typeof value === 'string' && value.length > 0;
}

/** @param {unknown} value */
function isCount(value) {
  return Number.isInteger(value) && value >= 0;
}

/** @param {unknown} error */
function describe(error) {
  return error instanceof Error ? error.message : String(error);
}
