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

// Python waits for this object to appear: it is the page's "ready" signal.
window.cifariumBench = { deviceInfo };
