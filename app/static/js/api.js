(function () {
  "use strict";

  const API_ROOT = "/api";

  class ApiError extends Error {
    constructor(message, options) {
      super(message);
      this.name = "ApiError";
      this.status = options.status || 0;
      this.code = options.code || "REQUEST_FAILED";
      this.requestId = options.requestId || "";
      this.retryable = Boolean(options.retryable);
    }
  }

  async function request(path, options) {
    const config = Object.assign({ headers: {} }, options || {});
    if (config.body && !(config.body instanceof FormData)) {
      config.headers["Content-Type"] = "application/json";
      if (typeof config.body !== "string") config.body = JSON.stringify(config.body);
    }
    config.headers.Accept = config.headers.Accept || "application/json";

    let response;
    try {
      response = await fetch(`${API_ROOT}${path}`, config);
    } catch (error) {
      throw new ApiError("无法连接到服务，请确认 FastAPI 已启动。", {
        code: "NETWORK_ERROR",
        retryable: true
      });
    }

    if (!response.ok) {
      let payload = {};
      try { payload = await response.json(); } catch (_) { payload = {}; }
      const detail = payload.error || payload.detail || payload;
      throw new ApiError(detail.message || `请求失败 (${response.status})`, {
        status: response.status,
        code: detail.code || `HTTP_${response.status}`,
        requestId: detail.request_id || "",
        retryable: detail.retryable !== false && response.status >= 500
      });
    }

    if (response.status === 204) return null;
    const contentType = response.headers.get("content-type") || "";
    return contentType.includes("application/json") ? response.json() : response.blob();
  }

  function query(params) {
    const values = new URLSearchParams();
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") values.set(key, String(value));
    });
    const encoded = values.toString();
    return encoded ? `?${encoded}` : "";
  }

  function segment(value) { return encodeURIComponent(String(value)); }

  async function analyzeStream(payload, onStage) {
    let response;
    try {
      response = await fetch(`${API_ROOT}/analyze/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(payload)
      });
    } catch (_) {
      throw new ApiError("无法连接到服务，请确认 FastAPI 已启动。", { code: "NETWORK_ERROR", retryable: true });
    }
    if (!response.ok) {
      let errorPayload = {};
      try { errorPayload = await response.json(); } catch (_) { errorPayload = {}; }
      const detail = errorPayload.error || errorPayload.detail || errorPayload;
      throw new ApiError(detail.message || `请求失败 (${response.status})`, {
        status: response.status,
        code: detail.code || `HTTP_${response.status}`,
        requestId: detail.request_id || "",
        retryable: detail.retryable !== false && response.status >= 500
      });
    }
    if (!response.body || !response.body.getReader) return request("/analyze", { method: "POST", body: payload });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let finalResult = null;

    function consume(block) {
      if (!block.trim()) return;
      let eventName = "message";
      const dataLines = [];
      block.split(/\r?\n/).forEach(line => {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      });
      if (!dataLines.length) return;
      let data;
      try { data = JSON.parse(dataLines.join("\n")); } catch (_) { return; }
      if (eventName === "stage") onStage?.(data);
      if (eventName === "result") finalResult = data;
      if (eventName === "error") {
        throw new ApiError(data.message || "分析未完成", {
          code: data.code || "ANALYSIS_FAILED",
          requestId: data.request_id || "",
          retryable: Boolean(data.retryable)
        });
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      blocks.forEach(consume);
      if (done) break;
    }
    consume(buffer);
    if (!finalResult) throw new ApiError("分析流已结束，但没有收到最终结果。", { code: "STREAM_RESULT_MISSING", retryable: true });
    return finalResult;
  }

  const api = {
    health: () => request("/health"),
    datasets: () => request("/datasets"),
    questions: (params) => request(`/questions${query(params)}`),
    question: (stage, split, id) => request(`/questions/${segment(stage)}/${segment(split)}/${segment(id)}`),
    documentUrl: (stage, split, filename, download) => `${API_ROOT}/documents/${segment(stage)}/${segment(split)}/${segment(filename)}${download ? "?download=true" : ""}`,
    documentMeta: (stage, split, filename) => request(`/documents/${segment(stage)}/${segment(split)}/${segment(filename)}/meta`),
    documentOcr: (stage, split, filename) => request(`/documents/${segment(stage)}/${segment(split)}/${segment(filename)}/ocr`),
    pageImageUrl: (stage, split, filename, page) => `${API_ROOT}/documents/${segment(stage)}/${segment(split)}/${segment(filename)}/pages/${segment(page)}`,
    analyze: (payload) => request("/analyze", { method: "POST", body: payload }),
    analyzeStream,
    jobs: () => request("/jobs"),
    createJob: (payload) => request("/jobs", { method: "POST", body: payload }),
    job: (id) => request(`/jobs/${segment(id)}`),
    cancelJob: (id) => request(`/jobs/${segment(id)}/cancel`, { method: "POST" }),
    retryJob: (id) => request(`/jobs/${segment(id)}/retry`, { method: "POST" }),
    exportUrl: (id) => `${API_ROOT}/jobs/${segment(id)}/export`,
    jobSummaryUrl: (id) => `${API_ROOT}/jobs/${segment(id)}/summary`,
    runs: () => request("/runs"),
    settings: () => request("/settings/models"),
    updateSettings: (payload) => request("/settings/models", { method: "PUT", body: payload }),
    testModel: (kind) => request("/settings/models/test", { method: "POST", body: { target: kind, kind } })
  };

  window.InduDocAPI = Object.freeze(api);
  window.ApiError = ApiError;
})();
