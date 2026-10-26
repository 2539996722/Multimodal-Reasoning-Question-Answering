(function () {
  "use strict";

  const savedStage = sessionStorage.getItem("indudoc-stage");
  const savedAnalysisMode = sessionStorage.getItem("indudoc-analysis-mode");
  const savedOcrContext = sessionStorage.getItem("indudoc-ocr-context");
  const state = {
    view: location.hash.replace("#", "") || "overview",
    stage: savedStage === "semi_final" ? "semi_final" : "preliminary",
    split: "train",
    health: null,
    datasets: null,
    settings: null,
    questions: [],
    questionTotal: 0,
    questionPage: 1,
    pageSize: 30,
    selectedQuestion: null,
    documentMeta: null,
    pdfPage: 1,
    zoom: 100,
    fitWidth: true,
    analysis: null,
    analysisStatus: "idle",
    analysisMode: savedAnalysisMode === "real" ? "real" : "demo",
    evaluateSemantic: false,
    ocrEnabled: savedOcrContext !== "false",
    analysisEvents: [],
    comparisonResults: { withOcr: null, withoutOcr: null },
    jobs: [],
    runs: [],
    filters: { keyword: "", document: "", id: "" },
    jobPoller: null,
    mobilePane: "questions"
  };

  function setStage(stage) {
    state.stage = stage === "semi_final" ? "semi_final" : "preliminary";
    sessionStorage.setItem("indudoc-stage", state.stage);
  }

  function unwrap(payload) {
    if (payload == null) return payload;
    return payload.data !== undefined ? payload.data : payload.result !== undefined ? payload.result : payload;
  }

  function listFrom(payload, keys) {
    const value = unwrap(payload);
    if (Array.isArray(value)) return value;
    for (const key of keys || ["items"]) if (Array.isArray(value && value[key])) return value[key];
    return [];
  }

  function field(object, names, fallback) {
    for (const name of names) {
      if (object && object[name] !== undefined && object[name] !== null) return object[name];
    }
    return fallback;
  }

  window.InduDocState = state;
  window.StateTools = Object.freeze({ setStage, unwrap, listFrom, field });
})();
