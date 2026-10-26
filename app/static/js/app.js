(function () {
  "use strict";

  const API = window.InduDocAPI;
  const S = window.InduDocState;
  const T = window.StateTools;
  const V = window.InduDocViews;
  const main = document.getElementById("main-content");
  const validViews = ["overview", "workspace", "batch", "runs", "settings"];
  let filterTimer = null;
  let ocrReturnContext = null;
  let ocrViewerState = null;
  let documentOcr = null;
  let batchAuditJobId = null;
  let batchAuditIndex = 0;

  function icons() {
    if (window.lucide) window.lucide.createIcons({ attrs: { "aria-hidden": "true" } });
  }

  function toast(message, type) {
    const item = document.createElement("div");
    item.className = "toast";
    const iconName = type === "error" ? "circle-alert" : type === "success" ? "circle-check" : "info";
    item.innerHTML = `${V.icon(iconName)}<span>${V.escapeHtml(message)}</span><button type="button" aria-label="关闭通知">${V.icon("x")}</button>`;
    item.querySelector("button").addEventListener("click", () => item.remove());
    document.getElementById("toast-region").appendChild(item);
    icons();
    setTimeout(() => item.remove(), 4500);
  }

  function setOffline(isOffline) {
    document.getElementById("offline-banner").hidden = !isOffline;
    document.getElementById("api-dot").className = `status-dot ${isOffline ? "error" : ""}`;
    document.getElementById("api-label").textContent = isOffline ? "服务不可达" : "服务已连接";
  }

  function updateGlobal() {
    document.querySelectorAll("[data-stage]").forEach(button => button.classList.toggle("active", button.dataset.stage === S.stage));
    const settings = T.unwrap(S.settings) || {};
    const health = T.unwrap(S.health) || {};
    const modelHealth = health.models || {};
    const badge = document.getElementById("mode-badge");
    badge.textContent = S.analysisMode === "real" ? "真实评测" : "演示回放";
    badge.className = S.analysisMode === "real" ? "tag" : "mode-badge";
    ["ocr", "vlm", "retrieval", "reranker"].forEach(kind => {
      const model = settings[kind] || modelHealth[kind] || {};
      const configured = V.field(model, ["configured"], false);
      const element = document.getElementById(`${kind}-status`);
      if (!element) return;
      const label = kind === "ocr" ? "OCR" : kind === "vlm" ? "VLM" : kind === "retrieval" ? "Retrieval" : "Reranker";
      element.innerHTML = `<span class="status-dot ${configured ? "" : "warning"}"></span>${label} ${configured ? "已配置" : "待配置"}`;
    });
    document.getElementById("api-version").textContent = V.field(health, ["version"], "FastAPI");
  }

  function closeOverlays() {
    if (!document.getElementById("ocr-viewer").hidden) closeOcrViewer(false);
    if (!document.getElementById("analysis-viewer").hidden) closeAnalysisViewer(false);
    if (!document.getElementById("batch-audit-viewer").hidden) closeBatchAudit();
  }

  function navigate(view) {
    const target = validViews.includes(view) ? view : "overview";
    closeOverlays();
    S.view = target;
    if (location.hash !== `#${target}`) history.pushState(null, "", `#${target}`);
    document.querySelectorAll("[data-view]").forEach(button => {
      const active = button.dataset.view === target;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    document.getElementById("sidebar").classList.remove("open");
    render();
  }

  async function render() {
    clearJobPoller();
    if (S.view === "overview") {
      main.innerHTML = V.overview(); bindOverview(); icons(); await loadOverview(); return;
    }
    if (S.view === "workspace") {
      main.innerHTML = V.workspace(); bindWorkspace(); icons(); await loadQuestions(); return;
    }
    if (S.view === "batch") {
      main.innerHTML = V.batch(); bindBatch(); icons(); await loadJobs(); return;
    }
    if (S.view === "runs") {
      main.innerHTML = V.runs(); bindRuns(); icons(); await loadRuns(); return;
    }
    if (S.view === "settings") {
      if (!S.settings) await loadSettings(false);
      main.innerHTML = V.settings(); bindSettings(); icons();
    }
  }

  async function loadCore() {
    try {
      const [healthResult, settingsResult] = await Promise.allSettled([API.health(), API.settings()]);
      if (healthResult.status === "fulfilled") S.health = healthResult.value;
      if (settingsResult.status === "fulfilled") {
        S.settings = settingsResult.value;
        if (!sessionStorage.getItem("indudoc-analysis-mode")) {
          S.analysisMode = V.field(T.unwrap(S.settings) || {}, ["demo_mode"], true) ? "demo" : "real";
        }
      }
      setOffline(healthResult.status === "rejected");
      updateGlobal();
    } catch (_) { setOffline(true); }
  }

  async function loadOverview() {
    const body = document.getElementById("overview-body");
    if (!body) return;
    body.innerHTML = V.loadingState("正在扫描数据资产"); icons();
    try {
      const [datasets, runs] = await Promise.all([API.datasets(), API.runs().catch(() => [])]);
      S.datasets = datasets;
      S.runs = T.listFrom(runs, ["runs", "items"]);
      body.innerHTML = V.overviewContent();
      setOffline(false); bindGoButtons(); icons();
    } catch (error) {
      body.innerHTML = V.errorState(error, "overview-retry");
      document.getElementById("overview-retry")?.addEventListener("click", loadOverview);
      setOffline(error.code === "NETWORK_ERROR"); icons();
    }
  }

  function bindOverview() {
    document.getElementById("overview-refresh").addEventListener("click", loadOverview);
    bindGoButtons();
  }

  function bindGoButtons() {
    main.querySelectorAll("[data-go]").forEach(button => {
      if (button.dataset.goBound) return;
      button.dataset.goBound = "1";
      button.addEventListener("click", () => navigate(button.dataset.go));
    });
    main.querySelectorAll("[data-stage-go]").forEach(button => {
      if (button.dataset.stageGoBound) return;
      button.dataset.stageGoBound = "1";
      button.addEventListener("click", () => {
      T.setStage(button.dataset.stageGo);
      S.analysis = null;
      S.analysisEvents = [];
      S.analysisStatus = "idle";
      S.comparisonResults = { withOcr: null, withoutOcr: null };
      updateGlobal();
      navigate("workspace");
      });
    });
  }

  function questionParams() {
    return { stage: S.stage, split: S.split, page: S.questionPage, page_size: S.pageSize, keyword: S.filters.keyword, document: S.filters.document, document_name: S.filters.document, id: S.filters.id, question_id: S.filters.id };
  }

  async function loadQuestions() {
    const list = document.getElementById("question-list");
    if (!list) return;
    list.innerHTML = V.loadingState("正在加载题目", true); icons();
    try {
      const payload = await API.questions(questionParams());
      const data = T.unwrap(payload) || {};
      S.questions = T.listFrom(data, ["items", "questions", "results"]);
      S.questionTotal = Number(V.field(data, ["total", "count", "total_count"], S.questions.length));
      document.getElementById("question-total").textContent = `${new Intl.NumberFormat("zh-CN").format(S.questionTotal)} 条`;
      document.getElementById("question-page").textContent = `第 ${S.questionPage} 页`;
      document.getElementById("question-prev").disabled = S.questionPage <= 1;
      document.getElementById("question-next").disabled = S.questionPage * S.pageSize >= S.questionTotal;
      list.innerHTML = V.questionList(); bindQuestionItems(); setOffline(false); icons();
    } catch (error) {
      list.innerHTML = V.questionList(error);
      document.getElementById("questions-retry")?.addEventListener("click", loadQuestions);
      setOffline(error.code === "NETWORK_ERROR"); icons();
    }
  }

  function bindQuestionItems() {
    document.querySelectorAll("[data-question-id]").forEach(button => button.addEventListener("click", () => selectQuestion(button.dataset.questionId)));
    document.getElementById("clear-filters")?.addEventListener("click", () => {
      S.filters = { keyword: "", document: "", id: "" }; S.questionPage = 1;
      ["filter-keyword", "filter-document", "filter-id"].forEach(id => { const input = document.getElementById(id); if (input) input.value = ""; }); loadQuestions();
    });
  }

  async function selectQuestion(id) {
    const source = S.questions.find(item => String(V.field(item, ["id", "question_id"], "")) === String(id));
    S.selectedQuestion = source || null;
    documentOcr = null;
    S.analysis = null; S.analysisEvents = []; S.analysisStatus = "idle";
    S.comparisonResults = { withOcr: null, withoutOcr: null };
    refreshWorkspaceParts();
    try {
      const detail = await API.question(S.stage, S.split, id);
      S.selectedQuestion = T.unwrap(detail) || source;
      S.pdfPage = derivePage(S.selectedQuestion);
      refreshWorkspaceParts();
      await loadDocument();
      if (matchMedia("(max-width: 760px)").matches) setMobilePane("document");
    } catch (error) {
      toast(error.message, "error");
      document.getElementById("analysis-panel").innerHTML = `<div class="analysis-content">${V.errorState(error)}</div>`;
      icons();
    }
  }

  function derivePage(question) {
    const explicit = Number(V.field(question, ["page", "page_number", "target_page", "evidence_page", "page_hint"], 0));
    if (explicit > 0) return explicit;
    const text = V.field(question, ["question", "query", "text"], "");
    const match = String(text).match(/第\s*(\d+)\s*页/);
    return match ? Math.max(1, Number(match[1])) : 1;
  }

  function documentName() {
    return V.field(S.selectedQuestion, ["document", "document_name", "filename", "pdf", "file"], "");
  }

  async function loadDocument() {
    const filename = documentName();
    if (!filename) {
      document.getElementById("pdf-canvas").innerHTML = V.emptyState("题目未关联 PDF", "该条记录没有可访问的文档名称。" ); icons(); return;
    }
    document.getElementById("pdf-title").textContent = filename;
    const download = document.getElementById("pdf-download");
    download.href = API.documentUrl(S.stage, S.split, filename, true);
    download.removeAttribute("aria-disabled"); download.setAttribute("download", filename);
    document.getElementById("pdf-canvas").innerHTML = V.loadingState(`正在加载第 ${S.pdfPage} 页`, true); icons();
    try {
      S.documentMeta = T.unwrap(await API.documentMeta(S.stage, S.split, filename)) || {};
      try { documentOcr = T.unwrap(await API.documentOcr(S.stage, S.split, filename)) || null; }
      catch (_) { documentOcr = null; }
      const ocrButton = document.getElementById("open-document-ocr");
      if (ocrButton) { ocrButton.disabled = !documentOcr?.text; ocrButton.title = documentOcr?.text ? `查看 ${documentOcr.page_count || 0} 页 OCR 缓存` : "当前 PDF 没有 OCR TXT 缓存"; }
      const pages = Number(V.field(S.documentMeta, ["page_count", "pages", "total_pages"], 0));
      if (pages && S.pdfPage > pages) S.pdfPage = pages;
      updatePdfControls(); renderPdfPage();
    } catch (error) {
      S.documentMeta = null; updatePdfControls();
      renderPdfFrame(error);
    }
  }

  function updatePdfControls() {
    const total = Number(V.field(S.documentMeta, ["page_count", "pages", "total_pages"], 0));
    const input = document.getElementById("pdf-page"); if (!input) return;
    input.value = S.pdfPage; input.max = total || 9999;
    document.getElementById("pdf-pages").textContent = `/ ${total || "--"}`;
    document.getElementById("pdf-prev").disabled = S.pdfPage <= 1;
    document.getElementById("pdf-next").disabled = total ? S.pdfPage >= total : false;
    document.getElementById("zoom-label").textContent = `${S.zoom}%`;
  }

  function renderPdfPage() {
    const canvas = document.getElementById("pdf-canvas");
    const filename = documentName(); if (!canvas || !filename) return;
    const image = document.createElement("img");
    image.className = "pdf-page-image"; image.alt = `${filename} 第 ${S.pdfPage} 页`;
    image.src = API.pageImageUrl(S.stage, S.split, filename, S.pdfPage);
    image.style.width = S.fitWidth ? "min(100%, 860px)" : `${S.zoom}%`;
    image.addEventListener("load", () => setOffline(false));
    image.addEventListener("error", () => renderPdfFrame());
    canvas.replaceChildren(image);
  }

  function renderPdfFrame(error) {
    const canvas = document.getElementById("pdf-canvas"); const filename = documentName(); if (!canvas || !filename) return;
    const frame = document.createElement("iframe"); frame.className = "pdf-frame"; frame.title = `${filename} PDF 原文件预览`;
    frame.src = `${API.documentUrl(S.stage, S.split, filename)}#page=${S.pdfPage}&zoom=${S.fitWidth ? "page-width" : S.zoom}`;
    canvas.replaceChildren(frame);
    if (error) toast("页图渲染不可用，已切换到原 PDF 预览。", "info");
  }

  function setPdfPage(page) {
    const total = Number(V.field(S.documentMeta, ["page_count", "pages", "total_pages"], 0));
    S.pdfPage = Math.max(1, total ? Math.min(total, page) : page); updatePdfControls(); renderPdfPage();
  }

  function changeZoom(delta) {
    S.fitWidth = false; S.zoom = Math.max(50, Math.min(200, S.zoom + delta)); updatePdfControls(); renderPdfPage();
  }

  function refreshWorkspaceParts() {
    const list = document.getElementById("question-list"); if (list) { list.innerHTML = V.questionList(); bindQuestionItems(); }
    const panel = document.getElementById("analysis-panel"); if (panel) { panel.innerHTML = V.analysisPanel(); bindAnalysis(); }
    icons();
  }

  function bindWorkspace() {
    document.querySelectorAll("[data-work-stage]").forEach(button => button.addEventListener("click", () => {
      T.setStage(button.dataset.workStage); S.questionPage = 1; S.selectedQuestion = null; S.analysis = null; S.analysisEvents = []; S.comparisonResults = { withOcr: null, withoutOcr: null };
      updateGlobal(); render();
    }));
    document.getElementById("work-split").addEventListener("change", event => { S.split = event.target.value; S.questionPage = 1; S.selectedQuestion = null; S.analysis = null; S.analysisEvents = []; S.analysisStatus = "idle"; S.comparisonResults = { withOcr: null, withoutOcr: null }; render(); });
    [["filter-keyword", "keyword"], ["filter-document", "document"], ["filter-id", "id"]].forEach(([id, key]) => document.getElementById(id).addEventListener("input", event => {
      S.filters[key] = event.target.value.trim(); clearTimeout(filterTimer); filterTimer = setTimeout(() => { S.questionPage = 1; loadQuestions(); }, 320);
    }));
    document.getElementById("question-prev").addEventListener("click", () => { if (S.questionPage > 1) { S.questionPage -= 1; loadQuestions(); } });
    document.getElementById("question-next").addEventListener("click", () => { S.questionPage += 1; loadQuestions(); });
    document.getElementById("pdf-prev").addEventListener("click", () => setPdfPage(S.pdfPage - 1));
    document.getElementById("pdf-next").addEventListener("click", () => setPdfPage(S.pdfPage + 1));
    document.getElementById("pdf-page").addEventListener("change", event => setPdfPage(Number(event.target.value) || 1));
    document.getElementById("zoom-out").addEventListener("click", () => changeZoom(-10));
    document.getElementById("zoom-in").addEventListener("click", () => changeZoom(10));
    document.getElementById("fit-width").addEventListener("click", () => { S.fitWidth = true; S.zoom = 100; updatePdfControls(); renderPdfPage(); });
    document.getElementById("open-document-ocr").addEventListener("click", openDocumentOcrViewer);
    document.getElementById("analysis-toggle").addEventListener("click", () => document.querySelector(".analysis-column").classList.add("drawer-open"));
    document.getElementById("analysis-close").addEventListener("click", () => document.querySelector(".analysis-column").classList.remove("drawer-open"));
    document.querySelectorAll("[data-pane]").forEach(button => button.addEventListener("click", () => setMobilePane(button.dataset.pane)));
    bindAnalysis();
  }

  function setMobilePane(pane) {
    S.mobilePane = pane;
    document.querySelectorAll("[data-pane]").forEach(button => button.classList.toggle("active", button.dataset.pane === pane));
    document.querySelectorAll("[data-pane-column]").forEach(column => column.classList.toggle("mobile-active", column.dataset.paneColumn === pane));
  }

  function bindAnalysis() {
    document.getElementById("run-analysis")?.addEventListener("click", runAnalysis);
    document.getElementById("reset-analysis")?.addEventListener("click", () => { S.analysis = null; S.analysisEvents = []; S.analysisStatus = "idle"; S.comparisonResults = { withOcr: null, withoutOcr: null }; refreshWorkspaceParts(); });
    document.querySelectorAll("[data-analysis-mode]").forEach(button => button.addEventListener("click", () => {
      S.analysisMode = button.dataset.analysisMode === "real" ? "real" : "demo";
      sessionStorage.setItem("indudoc-analysis-mode", S.analysisMode);
      S.analysis = null; S.analysisEvents = []; S.analysisStatus = "idle";
      updateGlobal();
      refreshWorkspaceParts();
    }));
    document.getElementById("semantic-toggle")?.addEventListener("change", event => {
      S.evaluateSemantic = event.target.checked;
      refreshWorkspaceParts();
    });
    document.getElementById("ocr-context-toggle")?.addEventListener("change", event => {
      S.ocrEnabled = event.target.checked;
      sessionStorage.setItem("indudoc-ocr-context", String(S.ocrEnabled));
      const saved = S.comparisonResults[S.ocrEnabled ? "withOcr" : "withoutOcr"];
      S.analysis = saved;
      S.analysisEvents = saved?.technical_trace || [];
      S.analysisStatus = saved ? "completed" : "idle";
      refreshWorkspaceParts();
    });
    document.querySelectorAll("[data-comparison-variant]").forEach(button => button.addEventListener("click", () => {
      const withOcr = button.dataset.comparisonVariant === "withOcr";
      const saved = S.comparisonResults[withOcr ? "withOcr" : "withoutOcr"];
      if (!saved) return;
      S.ocrEnabled = withOcr;
      sessionStorage.setItem("indudoc-ocr-context", String(withOcr));
      S.analysis = saved;
      S.analysisEvents = saved.technical_trace || [];
      S.analysisStatus = "completed";
      refreshWorkspaceParts();
    }));
    document.querySelectorAll("[data-evidence-page]").forEach(button => button.addEventListener("click", () => {
      const page = Number(button.dataset.evidencePage);
      if (!Number.isFinite(page) || page < 1) return;
      setPdfPage(page);
      if (matchMedia("(max-width: 760px)").matches) setMobilePane("document");
    }));
    document.getElementById("open-ocr-viewer")?.addEventListener("click", openOcrViewer);
    document.getElementById("open-analysis-viewer")?.addEventListener("click", openAnalysisViewer);
  }

  function bindAuditViewerActions() {
    document.querySelectorAll("#analysis-viewer [data-evidence-page]").forEach(button => button.addEventListener("click", () => {
      const page = Number(button.dataset.evidencePage);
      if (!Number.isFinite(page) || page < 1) return;
      setPdfPage(page);
      closeAnalysisViewer();
      if (matchMedia("(max-width: 760px)").matches) setMobilePane("document");
    }));
    document.getElementById("open-ocr-viewer-modal")?.addEventListener("click", () => {
      const analysisViewer = document.getElementById("analysis-viewer");
      const analysisContent = document.getElementById("analysis-viewer-content");
      ocrReturnContext = { source: "analysis-viewer", scrollTop: analysisContent.scrollTop };
      analysisViewer.hidden = true;
      openOcrViewer();
    });
  }

  function openAnalysisViewer() {
    const viewer = document.getElementById("analysis-viewer");
    const content = document.getElementById("analysis-viewer-content");
    const result = T.unwrap(S.analysis) || {};
    content.innerHTML = V.analysisAuditContent();
    content.scrollTop = 0;
    document.getElementById("analysis-viewer-meta").textContent = `${documentName() || "工业文档"} · ${V.field(result.models || {}, ["vlm"], "未标记模型")} · ${V.field(result, ["mode"], S.analysisMode) === "real" ? "真实评测" : "演示回放"}`;
    viewer.hidden = false;
    document.body.classList.add("modal-open");
    bindAuditViewerActions();
    icons();
    document.getElementById("analysis-viewer-close").focus();
  }

  function closeAnalysisViewer(restoreFocus = true) {
    const viewer = document.getElementById("analysis-viewer");
    viewer.hidden = true;
    document.getElementById("analysis-viewer-content").replaceChildren();
    if (document.getElementById("ocr-viewer").hidden) document.body.classList.remove("modal-open");
    if (restoreFocus) document.getElementById("open-analysis-viewer")?.focus();
  }

  function updateAnalysisStage(stage) {
    const index = S.analysisEvents.findIndex(item => item.name === stage.name);
    if (index >= 0) S.analysisEvents[index] = stage;
    else S.analysisEvents.push(stage);
    if (stage.status === "running") S.analysisStatus = stage.name;
    if (stage.status === "failed") S.analysisStatus = "failed";
    refreshWorkspaceParts();
  }

  async function runAnalysis() {
    if (!S.selectedQuestion || !["idle", "completed", "failed"].includes(S.analysisStatus)) return;
    S.analysis = null; S.analysisEvents = []; S.analysisStatus = "locating"; refreshWorkspaceParts();
    try {
      const id = V.field(S.selectedQuestion, ["id", "question_id"], "");
      const result = await API.analyzeStream({ stage: S.stage, split: S.split, id, question_id: id, page: S.pdfPage, page_number: S.pdfPage, force_demo: S.analysisMode === "demo", use_ocr: S.stage === "preliminary" ? S.ocrEnabled : true, evaluate_semantic: S.evaluateSemantic }, updateAnalysisStage);
      S.analysis = result; S.analysisStatus = "completed";
      const resultData = T.unwrap(result) || {};
      if (S.analysisMode === "real" && S.stage === "preliminary") {
        S.comparisonResults[S.ocrEnabled ? "withOcr" : "withoutOcr"] = resultData;
      }
      const evidencePages = V.field(resultData, ["evidence_pages"], []);
      const evidence = Number(V.field(resultData, ["evidence_page", "page", "page_number"], evidencePages[0] || 0)); if (evidence > 0 && evidence !== S.pdfPage) setPdfPage(evidence);
      refreshWorkspaceParts(); toast("分析完成，结果已写入运行记录。", "success");
    } catch (error) {
      const settings = T.unwrap(S.settings) || {};
      S.analysis = {
        failed: true,
        message: error.message,
        error_code: error.code,
        retryable: error.retryable,
        request_id: error.requestId,
        mode: S.analysisMode,
        ocr_enabled: S.ocrEnabled,
        models: {
          ocr: V.field(settings.ocr || {}, ["model"], null),
          vlm: V.field(settings.vlm || {}, ["model"], null)
        },
        technical_trace: S.analysisEvents
      };
      S.analysisStatus = "failed"; refreshWorkspaceParts(); toast(error.message, "error");
    }
  }

  function fallbackOcrPages(text, pages, fallbackPage) {
    const source = String(text || "");
    const markers = [...source.matchAll(/^===== PDF P\.(\d+) =====\s*$/gm)];
    if (markers.length) {
      return markers.map((marker, index) => {
        const start = marker.index + marker[0].length;
        const end = index + 1 < markers.length ? markers[index + 1].index : source.length;
        const value = source.slice(start, end).trim();
        return { page: Number(marker[1]), cached: Boolean(value), characters: value.length, text: value };
      });
    }
    const page = Number(Array.isArray(pages) && pages.length === 1 ? pages[0] : fallbackPage) || 1;
    return [{ page, cached: Boolean(source.trim()), characters: source.length, text: source }];
  }

  function prepareOcrViewer(payload, { initialPage, model }) {
    const rawItems = Array.isArray(payload?.page_items) && payload.page_items.length
      ? payload.page_items
      : fallbackOcrPages(payload?.text, payload?.pages, initialPage);
    const items = rawItems.map((item, index) => ({
      page: Number(item.page) || index + 1,
      cached: item.cached !== false && Boolean(String(item.text || "").trim()),
      characters: Number(item.characters) || String(item.text || "").length,
      text: String(item.text || "")
    })).sort((left, right) => left.page - right.page);
    const requestedPage = Number(initialPage);
    let index = items.findIndex(item => item.page === requestedPage);
    if (index < 0) index = Math.max(0, items.findIndex(item => item.cached));
    ocrViewerState = { items, index, model: model || "TXT 缓存" };
    renderOcrViewerPage();
  }

  function renderOcrCorpusNote(corpus) {
    const region = document.getElementById("ocr-corpus-note");
    if (!region) return;
    region.innerHTML = V.ocrCorpusNote(corpus);
    icons();
  }

  function renderOcrViewerPage() {
    if (!ocrViewerState?.items?.length) return;
    const { items, index, model } = ocrViewerState;
    const item = items[index];
    const text = item.cached
      ? item.text.trim()
      : `PDF P.${item.page}\n\n当前页面没有 OCR TXT 缓存。`;
    const content = document.getElementById("ocr-viewer-content");
    content.textContent = text;
    content.scrollTop = 0;
    document.getElementById("ocr-viewer-count").textContent = item.cached
      ? `${new Intl.NumberFormat("zh-CN").format(item.characters || text.length)} 字符`
      : "未缓存";
    document.getElementById("ocr-page-position").textContent = `P.${item.page} / ${items.length}`;
    document.getElementById("ocr-viewer-meta").textContent = `${documentName() || "工业文档"} · PDF P.${item.page} · ${item.cached ? "TXT 缓存" : "未缓存"} · ${model}`;
    document.getElementById("ocr-page-prev").disabled = index <= 0;
    document.getElementById("ocr-page-next").disabled = index >= items.length - 1;
    document.getElementById("ocr-viewer-copy").disabled = !item.cached;
    document.getElementById("ocr-viewer-download").disabled = !item.cached;
  }

  function changeOcrPage(delta) {
    if (!ocrViewerState?.items?.length) return;
    const next = Math.max(0, Math.min(ocrViewerState.items.length - 1, ocrViewerState.index + delta));
    if (next === ocrViewerState.index) return;
    ocrViewerState.index = next;
    renderOcrViewerPage();
  }

  function openOcrViewer() {
    const result = T.unwrap(S.analysis) || {};
    const text = V.field(result, ["ocr_text", "text", "recognized_text"], "");
    const fullDocument = documentOcr?.page_items?.length ? documentOcr : null;
    if (!text && !fullDocument) return;
    const viewer = document.getElementById("ocr-viewer");
    const model = V.field(result.models || {}, ["ocr"], S.analysisMode === "demo" ? "演示占位" : "--");
    const pages = V.field(result, ["ocr_pages", "context_pages"], []);
    const page = V.field(result, ["evidence_page"], S.pdfPage);
    prepareOcrViewer(fullDocument || { text, pages }, { initialPage: page, model });
    renderOcrCorpusNote(result.ocr_corpus || documentOcr?.corpus || null);
    const returnButton = document.getElementById("ocr-viewer-return");
    returnButton.querySelector("span").textContent = ocrReturnContext?.source === "analysis-viewer" ? "返回完整分析" : "返回分析区";
    viewer.hidden = false;
    document.body.classList.add("modal-open");
    document.getElementById("ocr-viewer-close").focus();
  }

  function openDocumentOcrViewer() {
    if (!documentOcr?.page_items?.some(item => item.cached) && !documentOcr?.text) { toast("当前 PDF 没有可读取的 OCR TXT 缓存。", "info"); return; }
    const viewer = document.getElementById("ocr-viewer");
    prepareOcrViewer(documentOcr, { initialPage: S.pdfPage, model: "TXT 缓存" });
    renderOcrCorpusNote(documentOcr?.corpus || null);
    ocrReturnContext = { source: "document" };
    document.getElementById("ocr-viewer-return").querySelector("span").textContent = "返回 PDF 预览";
    viewer.hidden = false;
    document.body.classList.add("modal-open");
    document.getElementById("ocr-viewer-close").focus();
  }

  function closeOcrViewer(returnToSource = true) {
    document.getElementById("ocr-viewer").hidden = true;
    ocrViewerState = null;
    if (returnToSource && ocrReturnContext?.source === "analysis-viewer") {
      const analysisViewer = document.getElementById("analysis-viewer");
      const analysisContent = document.getElementById("analysis-viewer-content");
      analysisViewer.hidden = false;
      analysisContent.scrollTop = ocrReturnContext.scrollTop || 0;
      ocrReturnContext = null;
      document.body.classList.add("modal-open");
      document.getElementById("open-ocr-viewer-modal")?.focus();
      return;
    }
    const returnSource = ocrReturnContext?.source;
    ocrReturnContext = null;
    if (document.getElementById("analysis-viewer").hidden) document.body.classList.remove("modal-open");
    (returnSource === "document" ? document.getElementById("open-document-ocr") : document.getElementById("open-ocr-viewer") || document.getElementById("analysis-panel"))?.focus();
  }

  async function copyOcrText() {
    const text = document.getElementById("ocr-viewer-content").textContent;
    try {
      await navigator.clipboard.writeText(text);
      toast("OCR 文本已复制。", "success");
    } catch (_) {
      toast("浏览器未允许复制，请在文档区域手动选择文本。", "error");
    }
  }

  function downloadOcrText() {
    const text = document.getElementById("ocr-viewer-content").textContent;
    const basename = (documentName() || "ocr-document").replace(/\.pdf$/i, "").replace(/[^\w\u4e00-\u9fa5-]+/g, "-");
    const page = ocrViewerState?.items?.[ocrViewerState.index]?.page || S.pdfPage;
    const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url; link.download = `${basename}-P${page}-ocr.txt`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function loadJobs() {
    const list = document.getElementById("jobs-list"); if (!list) return;
    if (!S.jobs.length) { list.innerHTML = V.loadingState("正在加载任务", true); icons(); }
    try {
      const payload = await API.jobs(); S.jobs = T.listFrom(payload, ["jobs", "items"]);
      list.innerHTML = V.jobsList();
      const dashboard = document.getElementById("batch-dashboard"); if (dashboard) dashboard.innerHTML = V.batchDashboard();
      const consoleNode = document.getElementById("batch-live-console"); if (consoleNode) consoleNode.innerHTML = V.batchLiveConsole();
      document.getElementById("job-count").textContent = `${S.jobs.length} 个任务`; bindJobActions(); bindBatchAudit(); if (!document.getElementById("batch-audit-viewer").hidden) renderBatchAuditCard(); icons(); startJobPoller();
    } catch (error) { list.innerHTML = V.jobsList(error); document.getElementById("jobs-retry")?.addEventListener("click", loadJobs); icons(); }
  }

  function bindBatch() {
    document.getElementById("jobs-refresh").addEventListener("click", loadJobs);
    bindBatchAudit();
    document.getElementById("batch-stage").addEventListener("change", event => {
      const preliminary = event.target.value === "preliminary";
      document.getElementById("metric-hint").textContent = preliminary ? "Accuracy" : "Exact Match / ROUGE-1 F1";
      const toggle = document.getElementById("batch-use-ocr");
      if (toggle) { toggle.disabled = !preliminary; if (!preliminary) toggle.checked = true; }
      document.getElementById("batch-ocr-label").textContent = preliminary ? "其他页面 OCR" : "缓存优先 OCR";
      document.getElementById("batch-ocr-help").textContent = preliminary ? "关闭后运行纯视觉对照批次" : "复赛优先读取分页 TXT 缓存";
    });
    document.getElementById("batch-form").addEventListener("submit", createJob);
    bindJobActions(); startJobPoller();
  }

  function batchAuditJob() {
    return S.jobs.find(job => String(V.field(job, ["job_id", "id"], "")) === String(batchAuditJobId)) || S.jobs.find(job => ["running", "queued", "pending"].includes(V.field(job, ["status", "state"], ""))) || S.jobs[0] || null;
  }

  function renderBatchAuditCard() {
    const job = batchAuditJob();
    const modal = document.getElementById("batch-audit-viewer");
    if (!modal || !job) return;
    batchAuditJobId = V.field(job, ["job_id", "id"], batchAuditJobId);
    const items = Array.isArray(V.field(job, ["items", "item_states"], [])) ? V.field(job, ["items", "item_states"], []) : [];
    if (!items.length) return;
    batchAuditIndex = Math.max(0, Math.min(items.length - 1, batchAuditIndex));
    const item = items[batchAuditIndex] || {};
    const status = V.field(item, ["status"], "queued");
    const summary = Array.isArray(V.field(item, ["reasoning_summary"], [])) ? V.field(item, ["reasoning_summary"], []) : [];
    const evidence = Array.isArray(V.field(item, ["evidence"], [])) ? V.field(item, ["evidence"], []) : [];
    const options = Array.isArray(V.field(item, ["options"], [])) ? V.field(item, ["options"], []) : [];
    const error = V.field(V.field(item, ["error"], {}), ["message"], "");
    const html = `<div class="batch-audit-card-hero"><div><span class="eyebrow">QUESTION #${V.escapeHtml(V.field(item, ["id"], "--"))}</span><h3>${V.escapeHtml(status === "running" ? V.field(item, ["title"], "正在处理") : status === "completed" ? `答案 ${V.field(item, ["answer"], "--")}` : V.field(item, ["title"], "等待执行"))}</h3><p>${V.escapeHtml(V.field(item, ["question"], "题目文本尚未返回"))}</p></div><span class="status-text ${status === "failed" ? "error" : status === "completed" ? "success" : "info"}"><span class="status-dot"></span>${V.escapeHtml(V.statusName(status))}</span></div><div class="batch-audit-meta-grid"><div><span>当前阶段</span><strong>${V.escapeHtml(V.field(item, ["title"], V.field(item, ["current_stage"], "--")))}</strong></div><div><span>技术实现</span><strong>${V.escapeHtml(V.field(item, ["technology"], "OpenAI-compatible Pipeline"))}</strong></div><div><span>耗时</span><strong>${V.field(item, ["duration_ms"], null) == null ? "--" : `${V.field(item, ["duration_ms"], "--")} ms`}</strong></div><div><span>OCR 页面</span><strong>${Array.isArray(V.field(item, ["ocr_pages"], [])) ? V.field(item, ["ocr_pages"], []).map(page => `P.${page}`).join(", ") || "--" : "--"}</strong></div></div>${options.length ? `<div class="batch-audit-options"><span class="eyebrow">OPTIONS</span>${options.map(option => `<span>${V.escapeHtml(option)}</span>`).join("")}</div>` : ""}<section class="batch-audit-report-block"><span class="eyebrow">MODEL CONCLUSION</span><h4>答案与解释</h4><p>${V.escapeHtml(V.field(item, ["answer_explanation"], error || (status === "running" ? "模型正在处理，完成后显示答案解释。" : "暂无答案解释。")))}</p></section>${summary.length ? `<section class="batch-audit-report-block"><span class="eyebrow">AUDITABLE REASONING</span><h4>可审计推理摘要</h4><ol>${summary.map(step => `<li>${V.escapeHtml(step)}</li>`).join("")}</ol></section>` : ""}${evidence.length ? `<section class="batch-audit-report-block"><span class="eyebrow">EVIDENCE</span><h4>证据</h4>${evidence.map(value => `<blockquote>${value.page ? `P.${V.escapeHtml(value.page)} · ` : ""}${V.escapeHtml(value.quote || value.relevance || "已返回证据")}</blockquote>`).join("")}</section>` : ""}`;
    document.getElementById("batch-audit-content").innerHTML = html;
    document.getElementById("batch-audit-meta").textContent = `${V.field(job, ["job_id", "id"], "--")} · ${V.field(job, ["completed"], 0)} / ${V.field(job, ["total"], items.length)} 题完成`;
    document.getElementById("batch-audit-position").textContent = `${batchAuditIndex + 1} / ${items.length}`;
    document.getElementById("batch-audit-prev").disabled = batchAuditIndex <= 0;
    document.getElementById("batch-audit-next").disabled = batchAuditIndex >= items.length - 1;
    icons();
  }

  function openBatchAudit() {
    const job = batchAuditJob();
    if (!job) return toast("当前没有可查看的批量任务。", "info");
    batchAuditJobId = V.field(job, ["job_id", "id"], null);
    batchAuditIndex = 0;
    const modal = document.getElementById("batch-audit-viewer");
    modal.hidden = false; document.body.classList.add("modal-open"); renderBatchAuditCard(); document.getElementById("batch-audit-close").focus();
  }

  function closeBatchAudit() { document.getElementById("batch-audit-viewer").hidden = true; if (document.getElementById("ocr-viewer").hidden && document.getElementById("analysis-viewer").hidden) document.body.classList.remove("modal-open"); }

  function bindBatchAudit() {
    const open = document.getElementById("batch-audit-open");
    if (open && !open.dataset.bound) { open.addEventListener("click", openBatchAudit); open.dataset.bound = "1"; }
    const close = document.getElementById("batch-audit-close");
    if (close && !close.dataset.bound) { close.addEventListener("click", closeBatchAudit); close.dataset.bound = "1"; }
    const prev = document.getElementById("batch-audit-prev");
    if (prev && !prev.dataset.bound) { prev.addEventListener("click", () => { batchAuditIndex -= 1; renderBatchAuditCard(); }); prev.dataset.bound = "1"; }
    const next = document.getElementById("batch-audit-next");
    if (next && !next.dataset.bound) { next.addEventListener("click", () => { batchAuditIndex += 1; renderBatchAuditCard(); }); next.dataset.bound = "1"; }
    const viewer = document.getElementById("batch-audit-viewer");
    if (viewer && !viewer.dataset.bound) { viewer.addEventListener("click", event => { if (event.target.id === "batch-audit-viewer") closeBatchAudit(); }); viewer.dataset.bound = "1"; }
  }

  async function createJob(event) {
    event.preventDefault(); const form = event.currentTarget; const button = document.getElementById("batch-submit");
    const values = Object.fromEntries(new FormData(form));
    const payload = { stage: values.stage, split: values.split, offset: Number(values.offset), limit: Number(values.limit), concurrency: Number(values.concurrency), force_demo: false, use_ocr: document.getElementById("batch-use-ocr").checked, use_rag: values.stage === "semi_final", evaluate_semantic: document.getElementById("batch-evaluate-semantic").checked };
    button.disabled = true; button.classList.add("loading-button"); button.innerHTML = `${V.icon("loader-circle")}正在创建`; icons();
    try { await API.createJob(payload); toast("批量任务已创建。", "success"); await loadJobs(); }
    catch (error) { toast(error.message, "error"); }
    finally { button.disabled = false; button.classList.remove("loading-button"); button.innerHTML = `${V.icon("play")}启动真实评测`; icons(); }
  }

  function bindJobActions() {
    document.querySelectorAll("[data-cancel-job]").forEach(button => button.addEventListener("click", async () => {
      button.disabled = true; try { await API.cancelJob(button.dataset.cancelJob); toast("已提交取消请求。", "success"); await loadJobs(); } catch (error) { toast(error.message, "error"); button.disabled = false; }
    }));
    document.querySelectorAll("[data-retry-job]").forEach(button => button.addEventListener("click", async () => {
      button.disabled = true;
      const job = S.jobs.find(item => String(V.field(item, ["job_id", "id"], "")) === String(button.dataset.retryJob));
      const demoBatch = V.field(job || {}, ["mode"], "real") === "demo";
      try { await API.retryJob(button.dataset.retryJob); toast(demoBatch ? "失败题目已创建新的演示重试批次。" : "失败题目已创建新的真实重试批次。", "success"); await loadJobs(); } catch (error) { toast(error.message, "error"); button.disabled = false; }
    }));
  }

  function startJobPoller() {
    clearJobPoller(); if (S.view !== "batch" || !S.jobs.some(job => ["running", "pending", "queued"].includes(V.field(job, ["status", "state"], "")))) return;
    S.jobPoller = setInterval(loadJobs, 2000);
  }
  function clearJobPoller() { if (S.jobPoller) { clearInterval(S.jobPoller); S.jobPoller = null; } }

  async function loadRuns() {
    const list = document.getElementById("runs-list"); if (!list) return;
    list.innerHTML = V.loadingState("正在加载运行记录", true); icons();
    try { const payload = await API.runs(); S.runs = T.listFrom(payload, ["runs", "items"]); list.innerHTML = V.runsList(); icons(); }
    catch (error) { list.innerHTML = V.runsList(error); document.getElementById("runs-retry")?.addEventListener("click", loadRuns); icons(); }
  }
  function bindRuns() { document.getElementById("runs-refresh").addEventListener("click", loadRuns); }

  async function loadSettings(renderAfter) {
    try { S.settings = await API.settings(); updateGlobal(); if (renderAfter && S.view === "settings") render(); }
    catch (error) { toast(error.message, "error"); }
  }

  function bindSettings() {
    document.getElementById("settings-reset").addEventListener("click", () => loadSettings(true));
    document.getElementById("settings-form").addEventListener("submit", saveSettings);
    document.querySelectorAll("[data-toggle-key]").forEach(button => button.addEventListener("click", () => {
      const input = document.getElementById(`${button.dataset.toggleKey}-key`); input.type = input.type === "password" ? "text" : "password";
      button.innerHTML = V.icon(input.type === "password" ? "eye" : "eye-off"); icons();
    }));
    document.querySelectorAll("[data-test-model]").forEach(button => button.addEventListener("click", () => testModel(button.dataset.testModel, button)));
  }

  function settingsPayload(form) {
    const values = Object.fromEntries(new FormData(form));
    const payload = {
      demo_mode: document.getElementById("demo-mode").checked,
      ocr: { base_url: values.ocr_base_url, model: values.ocr_model, timeout: Number(values.ocr_timeout) },
      vlm: { base_url: values.vlm_base_url, model: values.vlm_model, timeout: Number(values.vlm_timeout), temperature: Number(values.vlm_temperature) },
      retrieval: { base_url: values.retrieval_base_url, model: values.retrieval_model, timeout: Number(values.retrieval_timeout) },
      reranker: { base_url: values.reranker_base_url, model: values.reranker_model, timeout: Number(values.reranker_timeout) }
    };
    if (values.ocr_api_key) payload.ocr.api_key = values.ocr_api_key;
    if (values.vlm_api_key) payload.vlm.api_key = values.vlm_api_key;
    if (values.retrieval_api_key) payload.retrieval.api_key = values.retrieval_api_key;
    if (values.reranker_api_key) payload.reranker.api_key = values.reranker_api_key;
    return payload;
  }

  async function saveSettings(event) {
    event.preventDefault(); const button = document.getElementById("settings-save");
    button.disabled = true; button.classList.add("loading-button"); button.innerHTML = `${V.icon("loader-circle")}正在保存`; icons();
    try { S.settings = await API.updateSettings(settingsPayload(event.currentTarget)); updateGlobal(); toast("模型配置已保存，服务重启后会自动加载。", "success"); render(); }
    catch (error) { toast(error.message, "error"); }
    finally { if (document.getElementById("settings-save")) { button.disabled = false; button.classList.remove("loading-button"); button.innerHTML = `${V.icon("save")}保存进程配置`; icons(); } }
  }

  async function testModel(kind, button) {
    const state = document.getElementById(`${kind}-test-state`); button.disabled = true; button.classList.add("loading-button"); state.className = "status-text info"; state.textContent = "连接中";
    try {
      const result = T.unwrap(await API.testModel(kind)) || {};
      const item = Array.isArray(result.items) ? result.items[0] || {} : result;
      const protocol = V.field(item, ["protocol"], "");
      const dimensions = V.field(item, ["dimensions"], null);
      const labels = { openai_embeddings: "Embeddings", compatible_rerank: "Rerank API", chat_completions_fallback: "Chat 回退" };
      state.className = "status-text success";
      state.textContent = `连接成功${protocol ? ` · ${labels[protocol] || protocol}` : ""}${dimensions ? ` · ${dimensions} 维` : ""}`;
    }
    catch (error) { state.className = "status-text error"; state.textContent = error.message; }
    finally { button.disabled = false; button.classList.remove("loading-button"); }
  }

  function bindShell() {
    document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => navigate(button.dataset.view)));
    document.querySelectorAll("[data-stage]").forEach(button => button.addEventListener("click", () => { T.setStage(button.dataset.stage); S.analysis = null; S.analysisEvents = []; S.analysisStatus = "idle"; S.comparisonResults = { withOcr: null, withoutOcr: null }; updateGlobal(); if (S.view === "workspace" || S.view === "overview") render(); }));
    document.getElementById("settings-shortcut").addEventListener("click", () => navigate("settings"));
    document.getElementById("mobile-menu").addEventListener("click", () => document.getElementById("sidebar").classList.toggle("open"));
    document.getElementById("refresh-global").addEventListener("click", async () => { await loadCore(); toast("全局状态已刷新。", "success"); if (S.view === "overview") loadOverview(); });
    document.getElementById("offline-retry").addEventListener("click", async () => { await loadCore(); render(); });
    window.addEventListener("hashchange", () => { const view = location.hash.replace("#", ""); if (validViews.includes(view) && view !== S.view) { S.view = view; render(); } });
    window.addEventListener("online", () => loadCore()); window.addEventListener("offline", () => setOffline(true));
    document.getElementById("ocr-viewer-close").addEventListener("click", () => closeOcrViewer(false));
    document.getElementById("ocr-viewer-done").addEventListener("click", () => closeOcrViewer(false));
    document.getElementById("ocr-viewer-return").addEventListener("click", () => closeOcrViewer(true));
    document.getElementById("ocr-viewer-copy").addEventListener("click", copyOcrText);
    document.getElementById("ocr-viewer-download").addEventListener("click", downloadOcrText);
    document.getElementById("ocr-page-prev").addEventListener("click", () => changeOcrPage(-1));
    document.getElementById("ocr-page-next").addEventListener("click", () => changeOcrPage(1));
    document.getElementById("ocr-viewer").addEventListener("click", event => { if (event.target.id === "ocr-viewer") closeOcrViewer(false); });
    document.getElementById("analysis-viewer-close").addEventListener("click", () => closeAnalysisViewer());
    document.getElementById("analysis-viewer-done").addEventListener("click", () => closeAnalysisViewer());
    document.getElementById("analysis-viewer").addEventListener("click", event => { if (event.target.id === "analysis-viewer") closeAnalysisViewer(); });
    document.addEventListener("keydown", event => {
      const ocrVisible = !document.getElementById("ocr-viewer").hidden;
      if (ocrVisible && !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey && ["ArrowLeft", "ArrowRight"].includes(event.key)) {
        event.preventDefault();
        changeOcrPage(event.key === "ArrowLeft" ? -1 : 1);
        return;
      }
      if (event.key === "Escape") { document.getElementById("sidebar").classList.remove("open"); document.querySelector(".analysis-column")?.classList.remove("drawer-open"); if (ocrVisible) closeOcrViewer(Boolean(ocrReturnContext?.source === "analysis-viewer")); else if (!document.getElementById("analysis-viewer").hidden) closeAnalysisViewer(); else if (!document.getElementById("batch-audit-viewer").hidden) closeBatchAudit(); }
    });
  }

  async function init() {
    if (!validViews.includes(S.view)) S.view = "overview";
    bindShell(); icons(); updateGlobal(); await loadCore(); navigate(S.view);
  }

  init();
})();
