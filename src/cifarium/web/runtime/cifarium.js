/**
 * cifarium runtime: runs the model of a cifarium build on a web page.
 *
 *     import { createCifarium } from './cifarium.js';
 *     const model = await createCifarium('./cifarium.json');
 *     await model.load({ onProgress: (p) => bar.value = p.percent });
 *     const { label, score } = await model.run('Where is my parcel?');
 *
 * The model computes in a background thread (cifarium-worker.js, next to this file), on the
 * graphics card when the build has a variant for it and the browser offers an adapter, and on
 * the processor otherwise or when the card fails. Nothing is sent anywhere: the runtime only
 * reads the files of its own folder. ARCHITECTURE.md, section 8, describes every call.
 */

/** Version of this runtime; the same as the cifarium package it comes with. */
export const VERSION = '0.0.1';

/** The only manifest format this runtime reads (`format` in cifarium.json). */
export const MANIFEST_FORMAT = 1;

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
 * @typedef {Object} Manifest cifarium.json, format 1.
 * @property {number} format
 * @property {string} cifarium Version of cifarium that made the build.
 * @property {string} task
 * @property {string} recipe
 * @property {{repo: string, revision: string, license: string, path: string}} model `path` is
 *   the folder the library reads models from; the files lie in `<path><repo>/`.
 * @property {{cpu?: Variant, gpu?: Variant}} variants
 * @property {{pooling: "mean"|"cls"|"none", prefix: string}} params
 * @property {{transformers: string, onnxruntime_web: string, library: FileRef, loader: FileRef,
 *   executor: FileRef}} engine
 * @property {{labels: FileRef, centres: FileRef}} data
 */

/**
 * @typedef {Object} Progress What `onProgress` receives, once per step of any file.
 * @property {string} file Path of the file relative to the manifest.
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
 * @property {string[]|null} labels In the order of the settings; null until loaded.
 * @property {"cpu"|"gpu"|null} variant The variant of the manifest in use; null until loaded.
 * @property {"wasm"|"webgpu"|null} device
 * @property {string|null} dtype
 * @property {number|null} threads Threads of ONNX Runtime on the processor.
 * @property {boolean|null} crossOriginIsolated Of the background thread: without it, one thread.
 * @property {number|null} bytes What a first visit downloads for the variant in use: engine,
 *   model and data, before compression.
 * @property {string|null} fallback Why the graphics card's variant is not in use, when the
 *   manifest has one; null when it is in use or there is none.
 */

/**
 * @typedef {Object} Result
 * @property {string} label The label whose centre is nearest.
 * @property {number} score Dot product of the text's vector with that centre, from -1 to 1.
 */

/**
 * The only kind of error the runtime throws. `code` says what went wrong:
 * "manifest" (cannot fetch or read cifarium.json), "unsupported" (another manifest format, task
 * or recipe), "input" (a wrong argument), "not-loaded" (run() before load()), "load" (the model
 * did not start on any device), "run" (the model failed on a text), "disposed" (the instance
 * was disposed).
 */
export class CifariumError extends Error {
  /**
   * @param {string} code
   * @param {string} message
   */
  constructor(code, message) {
    super(message);
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
  } catch {
    throw new CifariumError('input', `${JSON.stringify(String(manifestUrl))} is not an address.`);
  }
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new CifariumError('manifest', `Could not fetch ${url}: ${describe(error)}`);
  }
  if (!response.ok) {
    throw new CifariumError('manifest', `Could not fetch ${url}: HTTP ${response.status}.`);
  }
  let manifest;
  try {
    manifest = await response.json();
  } catch {
    throw new CifariumError('manifest', `${url} is not JSON.`);
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
   * failure it starts again.
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
   * Names the label of a text. Calls may overlap: they run one after another, in order.
   * A call made while load() is still running waits for it.
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
   * What the build is and how it runs here. Fields about the device stay null until loaded.
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
      onnxruntimeWeb: manifest.engine.onnxruntime_web,
      labels: loaded ? [...loaded.labels] : null,
      variant: loaded?.variant ?? null,
      device: loaded?.device ?? null,
      dtype: loaded?.dtype ?? null,
      threads: loaded?.threads ?? null,
      crossOriginIsolated: loaded?.crossOriginIsolated ?? null,
      bytes: loaded?.bytes ?? null,
      fallback: loaded?.fallback ?? null,
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
    /** @type {{fallback: string, carry: any[]}|undefined} */
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
          ? new CifariumError('load', `The model did not start: ${error.fallback}.`)
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

  /** @param {{id: number, type: string, value?: any, progress?: Progress, error?: {code: string, message: string}}} message */
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
      call.reject(new RetryOnProcessor(message.fallback, message.carry));
    } else {
      call.reject(new CifariumError(message.error?.code ?? 'run', message.error?.message ?? 'Unknown error.'));
    }
  }
}

/** Internal: the graphics card failed in a thread, the processor is to be tried in a new one. */
class RetryOnProcessor extends Error {
  /**
   * @param {string} fallback
   * @param {any[]} carry
   */
  constructor(fallback, carry) {
    super(fallback);
    this.fallback = fallback;
    this.carry = carry;
  }
}

function disposed() {
  return new CifariumError('disposed', 'This cifarium instance was disposed.');
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
  if (manifest.format !== MANIFEST_FORMAT) {
    throw new CifariumError('unsupported',
      `${where} has format ${JSON.stringify(manifest.format)}; this runtime (${VERSION}) reads ` +
      `format ${MANIFEST_FORMAT}. Use the cifarium.js and cifarium-worker.js from the same build.`);
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
  const model = manifest.model;
  need(isObject(model) && isText(model.repo) && isText(model.revision) && isText(model.path)
    && typeof model.license === 'string', 'model: repo, revision, license and path');
  const params = manifest.params;
  need(isObject(params) && POOLINGS.includes(params.pooling) && typeof params.prefix === 'string',
    `params: pooling (${POOLINGS.join(', ')}) and prefix`);
  const engine = manifest.engine;
  need(isObject(engine) && isText(engine.transformers) && isText(engine.onnxruntime_web)
    && isFile(engine.library) && isFile(engine.loader) && isFile(engine.executor),
    'engine: transformers, onnxruntime_web, library, loader and executor');
  const data = manifest.data;
  need(isObject(data) && isFile(data.labels) && isFile(data.centres), 'data: labels and centres');
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
