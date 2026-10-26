(function () {
  "use strict";

  const S = window.InduDocState;
  const T = window.StateTools;

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  }

  function icon(name) { return `<i data-lucide="${name}" aria-hidden="true"></i>`; }
  function formatNumber(value) { if (value === null || value === undefined || value === "") return "--"; return Number.isFinite(Number(value)) ? new Intl.NumberFormat("zh-CN").format(Number(value)) : "--"; }
  function formatTime(value) {
    if (!value) return "--";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? escapeHtml(value) : new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
  }
  function stageName(stage) { return stage === "semi_final" || stage === "semifinal" ? "复赛" : "初赛"; }
  function splitName(split) { return split === "test" ? "测试集" : "训练集"; }
  function stateClass(status) {
    if (["completed", "success", "healthy", "connected", "ok"].includes(status)) return "success";
    if (["failed", "error", "unavailable"].includes(status)) return "error";
    if (["cancelled", "partial", "warning"].includes(status)) return "warning";
    return "info";
  }
  function statusName(status) {
    return ({ pending: "等待中", queued: "排队中", running: "运行中", completed: "已完成", success: "成功", failed: "失败", error: "错误", cancelled: "已取消", partial: "部分失败", locating: "定位页面", ocr: "OCR 识别", reasoning: "视觉推理" })[status] || status || "未知";
  }
  function field(object, names, fallback) { return T.field(object, names, fallback); }

  function pageHeader(title, description, actions) {
    return `<header class="page-header"><div class="page-title"><h1>${title}</h1><p>${description}</p></div><div class="page-actions">${actions || ""}</div></header>`;
  }

  function loadingState(label, compact) {
    return `<div class="state-block ${compact ? "compact" : ""}">${icon("loader-circle")}<strong>${escapeHtml(label || "正在加载")}</strong><div style="width:min(260px,80%)"><div class="skeleton skeleton-line"></div><div class="skeleton skeleton-line"></div><div class="skeleton skeleton-line"></div></div></div>`;
  }

  function emptyState(title, detail, action) {
    return `<div class="state-block">${icon("inbox")}<strong>${escapeHtml(title)}</strong><p>${escapeHtml(detail)}</p>${action || ""}</div>`;
  }

  function errorState(error, retryId) {
    const message = error && error.message ? error.message : "请求未完成，请稍后重试。";
    const request = error && error.requestId ? `<small class="mono">Request ID: ${escapeHtml(error.requestId)}</small>` : "";
    return `<div class="state-block">${icon("circle-alert")}<strong>内容加载失败</strong><p>${escapeHtml(message)}</p>${request}${retryId ? `<button class="secondary-button" id="${retryId}" type="button">${icon("refresh-cw")}重试</button>` : ""}</div>`;
  }

  function datasetRows() {
    const data = T.unwrap(S.datasets) || {};
    let items = T.listFrom(data, ["datasets", "items", "stages"]);
    if (!items.length) {
      items = ["preliminary", "semi_final"].map(key => {
        const entry = data[key] || data[key === "preliminary" ? "preliminary_dataset" : "semi_final_dataset"];
        return entry ? Object.assign({ stage: key }, entry) : null;
      }).filter(Boolean);
    }
    return items;
  }

  function datasetStats(item) {
    const splitList = Array.isArray(item.splits) ? item.splits : [];
    const splits = Array.isArray(item.splits) ? Object.fromEntries(item.splits.map(value => [value.id, value])) : (item.splits || {});
    const train = splits.train || item.train || {};
    const test = splits.test || item.test || {};
    const anomalies = splitList.flatMap(value => value.anomalies || []);
    return {
      stage: field(item, ["stage", "name", "id"], "preliminary"),
      train: field(train, ["question_count", "questions", "count"], field(item, ["train_questions", "train_count"], null)),
      test: field(test, ["question_count", "questions", "count"], field(item, ["test_questions", "test_count"], null)),
      pdf: field(item, ["pdf_count", "documents", "document_count"], field(train, ["pdf_count", "documents"], 0) + field(test, ["pdf_count", "documents"], 0)),
      warnings: field(item, ["warnings", "anomalies", "errors"], anomalies),
      available: field(item, ["available", "exists"], true)
    };
  }

  function overview() {
    const bandHealth = T.unwrap(S.health) || {};
    const bandModels = bandHealth.models || {};
    const bandMode = field(bandModels, ["effective_mode"], "") === "real" ? "真实模型" : "演示模式";
    const bandService = ["ok", "healthy"].includes(field(bandHealth, ["status"], "")) ? "服务正常" : "状态待确认";
    const bandRag = bandHealth.rag || {};
    const bandFramework = bandRag.framework || "LangChain";
    const bandIndex = bandRag.vector_index || "FAISS";
    return `<section class="page enterprise-overview" data-page="overview">
      <header class="overview-command-bar">
        <div class="overview-command-copy">
          <span class="eyebrow">INDUSTRIAL DOCUMENT INTELLIGENCE</span>
          <h1>项目总览 <em>InduDoc AI</em></h1>
          <p>把不可编辑的工业专利文档变成可核验的多模态答案：页面定位 → OCR → LangChain 分块 → FAISS 召回 → Reranker 重排 → VLM 短答案，全程留痕可审计。</p>
        </div>
        <div class="overview-command-frame">
          <div class="overview-command-facts">
            <div><span>任务形态</span><strong>初赛四选一 · 复赛开放问答</strong></div>
            <div><span>检索引擎</span><strong>${escapeHtml(bandFramework)} + ${escapeHtml(bandIndex)}</strong></div>
            <div><span>运行模式</span><strong>${escapeHtml(bandMode)}</strong></div>
            <div><span>服务状态</span><strong>${escapeHtml(bandService)}</strong></div>
          </div>
        </div>
        <div class="page-actions">
          <button class="secondary-button" id="overview-refresh" type="button" title="刷新数据" aria-label="刷新数据">${icon("refresh-cw")}<span>刷新数据</span></button>
          <button class="secondary-button" type="button" title="模型设置" aria-label="模型设置" data-go="settings">${icon("sliders-horizontal")}<span>模型设置</span></button>
          <button class="secondary-button" type="button" title="运行记录" aria-label="运行记录" data-go="runs">${icon("history")}<span>运行记录</span></button>
          <button class="primary-button" type="button" title="开始分析" aria-label="开始分析" data-go="workspace">${icon("play")}<span>开始分析</span></button>
        </div>
      </header>
      <div class="overview-canvas" id="overview-body">${loadingState("正在扫描数据资产")}</div>
    </section>`;
  }

  function overviewContent() {
    const rows = datasetRows().map(datasetStats);
    const totals = rows.reduce((sum, item) => ({
      train: sum.train + Number(item.train || 0),
      test: sum.test + Number(item.test || 0),
      pdf: sum.pdf + Number(item.pdf || 0),
      warnings: sum.warnings + (Array.isArray(item.warnings) ? item.warnings.length : Number(item.warnings || 0))
    }), { train: 0, test: 0, pdf: 0, warnings: 0 });
    const byStage = Object.fromEntries(rows.map(row => [row.stage, row]));
    const health = T.unwrap(S.health) || {};
    const models = health.models || S.settings || {};
    const rag = health.rag || {};
    const appHealthy = ["ok", "healthy"].includes(field(health, ["status"], ""));
    const recent = S.runs.slice(0, 8);
    const succeeded = S.runs.filter(run => field(run, ["status"], "") === "completed").length;
    const successRate = S.runs.length ? succeeded / S.runs.length : null;
    const stageDetail = key => {
      const preliminary = byStage.preliminary || {};
      const semiFinal = byStage.semi_final || {};
      return `初赛 ${formatNumber(preliminary[key])} · 复赛 ${formatNumber(semiFinal[key])}`;
    };
    const kpis = [
      ["训练题", formatNumber(totals.train), stageDetail("train")],
      ["测试题", formatNumber(totals.test), stageDetail("test")],
      ["PDF", formatNumber(totals.pdf), stageDetail("pdf")],
      ["数据异常", formatNumber(totals.warnings), "来自数据集目录扫描"]
    ];
    const chain = [
      ["file-scan", "PyMuPDF/OCR", "页面渲染与文本识别"],
      ["workflow", "LangChain 分块", "递归字符切分与文档编排"],
      ["boxes", "FAISS 召回", "向量索引 Top-K 检索"],
      ["list-filter", "Reranker", "候选片段相关性重排"],
      ["brain-circuit", "VLM", "图文联合推理生成短答案"],
      ["chart-no-axes-column", "Metrics", "Accuracy / EM / ROUGE-1"]
    ];
    const assetRows = rows.map(row => `<tr>
      <td><strong>${escapeHtml(stageName(row.stage))}</strong></td>
      <td>${row.stage === "semi_final" ? "开放问答" : "四选一"}</td>
      <td class="mono">${formatNumber(row.train)}</td>
      <td class="mono">${formatNumber(row.test)}</td>
      <td class="mono">${formatNumber(row.pdf)}</td>
      <td class="mono">${formatNumber(Array.isArray(row.warnings) ? row.warnings.length : Number(row.warnings || 0))}</td>
      <td><span class="status-text ${row.available ? "success" : "error"}"><span class="status-dot ${row.available ? "" : "error"}"></span>${row.available ? "可用" : "不可用"}</span></td>
    </tr>`).join("");
    const runRows = recent.map(run => `<tr>
      <td class="mono">${escapeHtml(formatTime(field(run, ["created_at", "timestamp", "time"], "")))}</td>
      <td>${field(run, ["type"], "") === "batch" ? "批量" : "单题"}</td>
      <td class="mono">${escapeHtml(field(run, ["question_id", "job_id", "id"], "--"))}</td>
      <td>${escapeHtml(stageName(field(run, ["stage"], "preliminary")))}</td>
      <td><span class="status-text ${stateClass(field(run, ["status"], ""))}">${statusName(field(run, ["status"], ""))}</span></td>
      <td class="mono">${field(run, ["duration_ms"], null) == null ? "--" : `${formatNumber(field(run, ["duration_ms"], null))} ms`}</td>
    </tr>`).join("");
    return `
      <div class="overview-kpi-grid">${kpis.map(([label, value, detail]) => `<article class="overview-kpi"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></article>`).join("")}</div>
      <div class="overview-layout">
        <section class="panel overview-assets"><div class="panel-head"><div><h2>数据资产</h2><p>初赛与复赛数据目录扫描结果</p></div><div class="page-actions"><span class="tag">只读</span><button class="secondary-button" type="button" title="进入初赛工作区" aria-label="进入初赛工作区" data-stage-go="preliminary">${icon("file-search-2")}<span>初赛工作区</span></button><button class="secondary-button" type="button" title="进入复赛工作区" aria-label="进入复赛工作区" data-stage-go="semi_final">${icon("file-search-2")}<span>复赛工作区</span></button></div></div>${rows.length ? `<div class="table-scroll"><table class="dataset-table"><thead><tr><th>阶段</th><th>任务类型</th><th>训练题</th><th>测试题</th><th>PDF</th><th>数据异常</th><th>状态</th></tr></thead><tbody>${assetRows}</tbody></table></div>` : emptyState("未发现数据集", "请确认初赛与复赛数据目录位于项目根目录。")}</section>
        <section class="panel overview-models"><div class="panel-head"><div><h2>模型配置状态</h2><p>只显示配置状态；连通结果需在模型设置中手动测试</p></div><button class="secondary-button" type="button" title="模型设置" aria-label="模型设置" data-go="settings">模型设置</button></div><div class="panel-body health-list">${modelHealthRow("OCR", models.ocr || {}, "file-scan")}${modelHealthRow("VLM", models.vlm || {}, "brain-circuit")}${modelHealthRow("Retrieval", models.retrieval || {}, "search-code")}${modelHealthRow("Reranker", models.reranker || {}, "list-filter")}<div class="health-row"><span class="status-dot ${appHealthy ? "" : "warning"}"></span><div><strong>应用服务</strong><small>${escapeHtml(field(health, ["version"], "FastAPI"))}</small></div><span class="status-text ${appHealthy ? "success" : "warning"}">${appHealthy ? "正常" : "待确认"}</span></div></div></section>
      </div>
      <section class="panel overview-pipeline"><div class="panel-head"><div><h2>处理链路</h2><p>PyMuPDF/OCR → LangChain 分块 → FAISS 召回 → Reranker → VLM → Metrics</p></div><span class="tag">真实实现</span></div><div class="pipeline overview-chain">${chain.map(([iconName, title, detail]) => `<div class="pipeline-step"><span class="pipeline-icon">${icon(iconName)}</span><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small></div>`).join("")}</div><div class="overview-chain-note"><span>${icon("braces")}框架 ${escapeHtml(field(rag, ["framework"], "LangChain"))}</span><span>${icon("database")}向量索引 ${escapeHtml(field(rag, ["vector_index"], "FAISS"))}</span><span>${icon("hard-drive")}持久化 ${escapeHtml(field(rag, ["persistence"], "运行时缓存"))}</span><span>${icon("shield-check")}数据库 ${field(rag, ["database_required"], false) ? "需要" : "不需要"}</span></div></section>
      <section class="panel overview-runs"><div class="panel-head"><div><h2>最近运行</h2><p>当前服务进程内记录</p></div><div class="page-actions">${S.runs.length ? `<span class="status-text success"><span class="status-dot"></span>成功率 ${(successRate * 100).toFixed(1)}%（${succeeded} / ${S.runs.length}）</span>` : `<span class="status-text info"><span class="status-dot muted"></span>暂无运行记录</span>`}<button class="secondary-button" type="button" title="运行记录" aria-label="运行记录" data-go="runs">运行记录</button><button class="secondary-button" type="button" title="批量评测" aria-label="批量评测" data-go="batch">批量评测</button></div></div>${recent.length ? `<div class="table-scroll"><table class="runs-table overview-runs-table"><thead><tr><th>时间</th><th>类型</th><th>对象</th><th>阶段</th><th>状态</th><th>耗时</th></tr></thead><tbody>${runRows}</tbody></table></div>` : `<div class="overview-empty-runs"><strong>暂无运行记录</strong><span>完成一次单题分析或批量评测后，这里会显示当前服务进程内的真实记录与成功率。</span></div>`}</section>
    `;
  }
  function modelHealthRow(label, data, iconName) {
    const configured = field(data, ["configured"], false);
    return `<div class="health-row"><span class="pipeline-icon">${icon(iconName)}</span><div><strong>${label}</strong><small>${escapeHtml(field(data, ["model", "model_name"], "未配置模型"))}</small></div><span class="status-text ${configured ? "success" : "warning"}">${configured ? "已配置" : "待配置"}</span></div>`;
  }
  function pipelineStep(iconName, title, detail) { return `<div class="pipeline-step"><span class="pipeline-icon">${icon(iconName)}</span><strong>${title}</strong><small>${detail}</small></div>`; }

  function workspace() {
    return `<section class="workspace-page" data-page="workspace">
      <div class="workspace-toolbar"><div class="toolbar-group"><div class="stage-switch" role="group" aria-label="工作区数据集"><button type="button" data-work-stage="preliminary" class="${S.stage === "preliminary" ? "active" : ""}">初赛</button><button type="button" data-work-stage="semi_final" class="${S.stage === "semi_final" ? "active" : ""}">复赛</button></div><select id="work-split" aria-label="数据分区"><option value="train" ${S.split === "train" ? "selected" : ""}>训练集</option><option value="test" ${S.split === "test" ? "selected" : ""}>测试集</option></select></div><div class="toolbar-group"><span class="tag">${S.stage === "preliminary" ? "四选一问答" : "开放式问答"}</span><button class="secondary-button" id="analysis-toggle" type="button">${icon("panel-right-open")}分析面板</button></div></div>
      <div class="mobile-work-tabs" role="tablist"><button type="button" class="${S.mobilePane === "questions" ? "active" : ""}" data-pane="questions">题目</button><button type="button" class="${S.mobilePane === "document" ? "active" : ""}" data-pane="document">PDF</button><button type="button" class="${S.mobilePane === "analysis" ? "active" : ""}" data-pane="analysis">分析</button></div>
      <div class="workspace-grid">
        <section class="work-column questions-column ${S.mobilePane === "questions" ? "mobile-active" : ""}" data-pane-column="questions"><div class="column-head"><h2>题目队列</h2><span class="mono" id="question-total">-- 条</span></div><div class="filters"><div class="search-wrap">${icon("search")}<input class="field-input" id="filter-keyword" value="${escapeHtml(S.filters.keyword)}" placeholder="搜索问题、文档或 ID" aria-label="搜索题目"></div><input class="field-input" id="filter-document" value="${escapeHtml(S.filters.document)}" placeholder="文档名" aria-label="按文档名筛选"><input class="field-input" id="filter-id" value="${escapeHtml(S.filters.id)}" placeholder="题目 ID" aria-label="按题目 ID 筛选"></div><div class="column-scroll" id="question-list">${loadingState("正在加载题目", true)}</div><div class="pagination"><button class="icon-button" id="question-prev" type="button" title="上一页" aria-label="上一页">${icon("chevron-left")}</button><span id="question-page">第 ${S.questionPage} 页</span><button class="icon-button" id="question-next" type="button" title="下一页" aria-label="下一页">${icon("chevron-right")}</button></div></section>
        <section class="work-column document-column ${S.mobilePane === "document" ? "mobile-active" : ""}" data-pane-column="document"><div class="column-head"><h2 id="pdf-title">PDF 预览</h2><div class="pdf-toolbar"><button class="icon-button" id="pdf-prev" type="button" title="上一页" aria-label="上一页">${icon("chevron-left")}</button><input class="field-input page-field" id="pdf-page" type="number" min="1" value="${S.pdfPage}" aria-label="页码"><span class="mono" id="pdf-pages">/ --</span><button class="icon-button" id="pdf-next" type="button" title="下一页" aria-label="下一页">${icon("chevron-right")}</button><button class="icon-button" id="zoom-out" type="button" title="缩小" aria-label="缩小">${icon("zoom-out")}</button><span class="zoom-label" id="zoom-label">${S.zoom}%</span><button class="icon-button" id="zoom-in" type="button" title="放大" aria-label="放大">${icon("zoom-in")}</button><button class="icon-button" id="fit-width" type="button" title="适合宽度" aria-label="适合宽度">${icon("move-horizontal")}</button><button class="secondary-button pdf-ocr-button" id="open-document-ocr" type="button" disabled>${icon("file-scan")}OCR 文档</button><a class="icon-button" id="pdf-download" title="下载原文件" aria-label="下载原文件" aria-disabled="true">${icon("download")}</a></div></div><div class="pdf-canvas" id="pdf-canvas"><div class="pdf-empty">${icon("file-text")}<strong>选择题目以预览 PDF</strong><span>系统会自动定位问题中提到的页码</span></div></div></section>
        <section class="work-column analysis-column ${S.mobilePane === "analysis" ? "mobile-active" : ""}" data-pane-column="analysis"><div class="column-head"><h2>智能分析</h2><button class="icon-button" id="analysis-close" type="button" title="关闭分析面板" aria-label="关闭分析面板">${icon("x")}</button></div><div class="column-scroll" id="analysis-panel">${emptyState("尚未选择题目", "从左侧题目队列选择一项后，可运行 OCR 与 VLM 分析。")}</div></section>
      </div>
    </section>`;
  }

  function questionList(error) {
    if (error) return errorState(error, "questions-retry");
    if (!S.questions.length) return emptyState("没有匹配题目", "请调整阶段、分区或搜索条件。", `<button class="secondary-button" id="clear-filters" type="button">清除筛选</button>`);
    return `<div class="question-list">${S.questions.map((question, index) => {
      const id = field(question, ["id", "question_id"], `item-${index}`);
      const selectedId = field(S.selectedQuestion, ["id", "question_id"], null);
      const text = field(question, ["question", "query", "text"], "未提供问题文本");
      const document = field(question, ["document", "document_name", "filename", "pdf"], "未关联文档");
      const page = field(question, ["page", "page_number", "target_page", "page_hint"], null);
      return `<button class="question-item ${String(selectedId) === String(id) ? "active" : ""}" type="button" data-question-id="${escapeHtml(id)}"><div class="question-item-head"><strong class="mono">#${escapeHtml(id)}</strong>${page ? `<span class="tag">P.${escapeHtml(page)}</span>` : ""}</div><p>${escapeHtml(text)}</p><div class="question-meta"><span>${icon("file-text")}${escapeHtml(document)}</span></div></button>`;
    }).join("")}</div>`;
  }

  function traceBlueprint() {
    const definitions = [
      ["locating", "文档定位与页面渲染", "DatasetRepository + PyMuPDF"],
      ["ocr", "OCR 文档识别", "OpenAI-compatible image_url"],
      ["reasoning", "VLM 图文联合推理", "Multimodal Chat Completions"],
      ["validation", "答案规范化与评测", "Normalizer + jieba Metrics"]
    ];
    const active = ({ locating: 0, ocr: 1, reasoning: 2, validation: 3 })[S.analysisStatus];
    return definitions.map(([name, title, technology], index) => {
      const ocrDisabled = name === "ocr" && S.stage === "preliminary" && S.analysisMode === "real" && !S.ocrEnabled;
      return {
        name,
        title: ocrDisabled ? "其他页面 OCR 上下文" : title,
        technology: ocrDisabled ? "OCR Feature Flag" : technology,
        status: ocrDisabled ? "skipped" : active == null ? "pending" : index < active ? "completed" : index === active ? "running" : "pending",
        detail: ocrDisabled ? "纯视觉基线已关闭 OCR" : "等待执行"
      };
    });
  }

  function traceTimeline(items) {
    const icons = { locating: "file-search", ocr: "file-scan", reasoning: "brain-circuit", validation: "badge-check" };
    return `<ol class="trace-list">${items.map(item => {
      const status = field(item, ["status"], "pending");
      const duration = field(item, ["duration_ms"], null);
      const technology = field(item, ["technology"], "--");
      const model = field(item, ["model"], "");
      return `<li class="trace-item ${escapeHtml(status)}"><span class="trace-marker">${icon(icons[item.name] || "circle")}</span><div class="trace-body"><div class="trace-head"><strong>${escapeHtml(field(item, ["title"], item.name))}</strong><span class="status-text ${stateClass(status)}">${status === "skipped" ? "已跳过" : statusName(status)}</span></div><p>${escapeHtml(field(item, ["detail"], "等待执行"))}</p><div class="trace-tech"><span>${escapeHtml(technology)}</span>${model ? `<span>${escapeHtml(model)}</span>` : ""}${duration != null ? `<span>${formatNumber(duration)} ms</span>` : ""}</div>${field(item, ["input_summary"], "") ? `<small>${escapeHtml(field(item, ["input_summary"], ""))}</small>` : ""}${field(item, ["output_summary"], "") ? `<small>${escapeHtml(field(item, ["output_summary"], ""))}</small>` : ""}</div></li>`;
    }).join("")}</ol>`;
  }

  function comparisonCard(key, title, description, result) {
    if (!result) {
      return `<article class="comparison-card empty"><div class="comparison-card-head"><div><span class="eyebrow">${key === "withOcr" ? "OCR AUGMENTED" : "VISION BASELINE"}</span><h5>${escapeHtml(title)}</h5></div><span class="tag">待运行</span></div><p>${escapeHtml(description)}</p><div class="comparison-empty">运行该配置后显示答案、Accuracy、耗时和推理摘要</div></article>`;
    }
    const metrics = result.metrics || {};
    const summary = field(result, ["reasoning_summary", "analysis_summary"], []);
    const pages = field(result, ["ocr_pages", "context_pages"], []);
    const accuracy = metrics.applicable && metrics.accuracy != null ? formatMetric(metrics.accuracy) : "不适用";
    const duration = field(result, ["duration_ms", "elapsed_ms"], null);
    return `<article class="comparison-card ${key === "withOcr" ? "enhanced" : "baseline"}"><div class="comparison-card-head"><div><span class="eyebrow">${key === "withOcr" ? "OCR AUGMENTED" : "VISION BASELINE"}</span><h5>${escapeHtml(title)}</h5></div><span class="tag">已完成</span></div><div class="comparison-stats"><div><span>答案</span><strong>${escapeHtml(field(result, ["answer"], "--"))}</strong></div><div><span>Accuracy</span><strong>${accuracy}</strong></div><div><span>耗时</span><strong>${duration == null ? "--" : `${formatNumber(duration)} ms`}</strong></div><div><span>OCR 页</span><strong>${Array.isArray(pages) && pages.length ? pages.length : 0}</strong></div></div>${Array.isArray(summary) && summary.length ? `<ol class="comparison-reasoning">${summary.slice(0, 3).map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ol>` : ""}<button class="quiet-button" type="button" data-comparison-variant="${key}">${icon("panel-top-open")}查看完整结果</button></article>`;
  }

  function ocrComparisonPanel() {
    const withOcr = S.comparisonResults?.withOcr;
    const withoutOcr = S.comparisonResults?.withoutOcr;
    if (!withOcr && !withoutOcr) return "";
    let verdict = "再运行另一种配置后，系统会计算答案和 Accuracy 差异。";
    if (withOcr && withoutOcr) {
      const withAccuracy = withOcr.metrics?.accuracy;
      const withoutAccuracy = withoutOcr.metrics?.accuracy;
      const answerChanged = String(withOcr.answer) !== String(withoutOcr.answer);
      const durationDelta = Number(withOcr.duration_ms || 0) - Number(withoutOcr.duration_ms || 0);
      const accuracyText = withAccuracy != null && withoutAccuracy != null
        ? `Accuracy ${withAccuracy > withoutAccuracy ? "提升" : withAccuracy < withoutAccuracy ? "下降" : "持平"} ${Math.abs((withAccuracy - withoutAccuracy) * 100).toFixed(0)} 个百分点`
        : "Accuracy 无参考答案";
      verdict = `${accuracyText}；答案${answerChanged ? `由 ${withoutOcr.answer} 变为 ${withOcr.answer}` : "未变化"}；OCR 增强耗时${durationDelta >= 0 ? "增加" : "减少"} ${formatNumber(Math.abs(durationDelta))} ms。`;
    }
    return `<section class="result-section comparison-section"><div class="result-title"><div><span class="eyebrow">CONTROLLED COMPARISON</span><h4>OCR 增强对照</h4></div><span class="tag">同题 · 同目标页</span></div><div class="comparison-verdict">${icon("git-compare-arrows")}<span>${escapeHtml(verdict)}</span></div><div class="comparison-grid">${comparisonCard("withoutOcr", "纯视觉基线", "VLM 仅接收目标设计图页和题目。", withoutOcr)}${comparisonCard("withOcr", "其他页 OCR 增强", "目标设计图页交给 VLM，其他页面 OCR 后作为额外上下文。", withOcr)}</div></section>`;
  }

  function evidenceSourceLabel(source) {
    return ({ ocr: "OCR 原文", target_image: "目标图像", dataset_reference: "数据集参考" })[source] || "文档线索";
  }

  function answerInsightPanel(answer, explanation, mode, result) {
    if (!answer) return "";
    const badge = mode === "demo" || mode === "replay" ? "演示回放" : field(result, ["ocr_enabled"], false) ? "OCR 增强" : "纯视觉";
    return `<section class="answer-insight"><div class="answer-seal"><span>答案</span><strong>${escapeHtml(answer)}</strong></div><div class="answer-insight-copy"><div class="answer-insight-head"><div><span class="eyebrow">MODEL CONCLUSION</span><h4>为什么选择这个选项</h4></div><span class="mode-badge">${badge}</span></div><p>${escapeHtml(explanation || "模型未提供答案解释。")}</p></div></section>`;
  }

  function ragPageSummary(pages) {
    const values = Array.isArray(pages) ? pages.map(Number).filter(page => Number.isFinite(page) && page > 0) : [];
    if (!values.length) return "0 页";
    const visible = values.slice(0, 6).map(page => `P.${page}`).join("、");
    return `${values.length} 页 · ${visible}${values.length > 6 ? ` 等 ${values.length} 页` : ""}`;
  }

  function ragProtocolLabel(protocol) {
    return ({
      openai_embeddings: "OpenAI Embeddings",
      langchain_faiss_openai_embeddings: "LangChain + FAISS 向量召回",
      hybrid_keyword_vector_npy: "关键词 + NPY 向量混合召回",
      keyword_only_degraded: "关键词召回（向量降级）",
      compatible_rerank: "OpenAI-compatible Rerank",
      retrieval_score_fallback: "保留混合召回排序",
      chat_completions_fallback: "Chat Completions 回退评分",
      openai_chat_completions: "OpenAI Chat Completions",
      empty_index: "空索引",
      skipped: "未执行"
    })[protocol] || protocol || "未标记协议";
  }

  function ragReportPanel(rag) {
    if (!rag || typeof rag !== "object") return "";
    const items = Array.isArray(rag.items) ? rag.items : [];
    const pdfPages = Array.isArray(rag.pdf_text_pages) ? rag.pdf_text_pages : [];
    const ocrPages = Array.isArray(rag.ocr_pages) ? rag.ocr_pages : [];
    const declaredSources = Array.isArray(rag.corpus_sources) ? rag.corpus_sources : [];
    const hasPdfText = declaredSources.includes("pdf_text") || pdfPages.length > 0;
    const hasOcr = declaredSources.includes("ocr") || ocrPages.length > 0;
    const degraded = rag.vector_index_status === "degraded" || rag.rerank_protocol === "retrieval_score_fallback";
    const retrievalDegradation = field(rag, ["degradation_reason"], null);
    const rerankDegradation = field(rag, ["rerank_degradation_reason"], null);
    const degradationNote = retrievalDegradation || rerankDegradation ? `<div class="rag-degraded-note"><span class="status-dot warning"></span><div><strong>降级原因</strong><p>${escapeHtml([
      retrievalDegradation ? `检索：${field(retrievalDegradation, ["message"], field(retrievalDegradation, ["code"], "未提供原因"))}` : "",
      rerankDegradation ? `重排：${field(rerankDegradation, ["message"], field(rerankDegradation, ["code"], "未提供原因"))}` : ""
    ].filter(Boolean).join("；"))}</p></div></div>` : "";
    const status = rag.enabled ? (items.length ? (degraded ? "降级完成" : "混合检索完成") : "索引无命中") : rag.requested ? "配置未就绪" : "未启用";
    const sourceTags = [
      hasPdfText ? `<span class="rag-source pdf-source">${icon("file-text")}<span><strong>PDF 原生文本</strong><small>${escapeHtml(ragPageSummary(pdfPages))}</small></span></span>` : "",
      hasOcr ? `<span class="rag-source ocr-source">${icon("file-scan")}<span><strong>OCR 识别文本</strong><small>${escapeHtml(ragPageSummary(ocrPages))}</small></span></span>` : ""
    ].filter(Boolean).join("");
    const hits = items.length ? items.map((item, index) => {
      const page = Number(field(item, ["page", "page_number"], 0));
      const source = field(item, ["source"], "pdf_text");
      const retrievalScore = Number(item.retrieval_score);
      const rerankScore = Number(item.rerank_score);
      const text = String(field(item, ["text", "quote", "content"], "未返回片段文本。"));
      const excerpt = text.length > 420 ? `${text.slice(0, 420).trim()}…` : text;
      return `<article class="rag-hit"><div class="rag-hit-rank">${String(index + 1).padStart(2, "0")}</div><div class="rag-hit-copy"><div class="rag-hit-meta"><span class="rag-hit-source">${icon(source === "ocr" ? "file-scan" : "file-text")}${source === "ocr" ? "OCR 语料" : "PDF 原生文本"}</span><span class="rag-scores">${Number.isFinite(retrievalScore) ? `<small>召回 ${retrievalScore.toFixed(3)}</small>` : ""}${Number.isFinite(rerankScore) ? `<small>重排 ${rerankScore.toFixed(3)}</small>` : ""}</span>${page > 0 ? `<button type="button" data-evidence-page="${page}" title="在 PDF 中查看第 ${page} 页">${icon("file-search")}P.${page}</button>` : ""}</div><blockquote>${escapeHtml(excerpt)}</blockquote></div></article>`;
    }).join("") : `<div class="rag-empty">${icon("search-x")}<span><strong>当前没有可展示的命中片段</strong><small>${rag.enabled ? "索引为空或检索未召回相关内容。" : rag.requested ? "请完善 Retrieval 与 Reranker 模型配置后重新运行。" : "本次分析未启用 RAG。"}</small></span></div>`;
    return `<section class="rag-report"><div class="result-title"><div><span class="eyebrow">${icon("database")}RAG RETRIEVAL REPORT</span><h4>文档检索与重排序</h4></div><span class="tag rag-status">${escapeHtml(status)}</span></div><div class="rag-source-strip"><span class="rag-source-label">语料来源</span><div>${sourceTags || `<span class="rag-source-empty">未建立可检索语料</span>`}</div></div><div class="rag-stats"><div><span>文档分块</span><strong>${formatNumber(field(rag, ["chunk_count"], 0))}</strong></div><div><span>Retrieval 召回</span><strong>${formatNumber(field(rag, ["recalled_count"], 0))}</strong></div><div><span>Reranker 重排</span><strong>${formatNumber(field(rag, ["reranked_count"], 0))}</strong></div></div><div class="rag-flow"><div class="rag-stage retrieval"><span class="rag-stage-icon">${icon("search-code")}</span><div><small>RETRIEVAL PROTOCOL</small><strong>${escapeHtml(ragProtocolLabel(rag.retrieval_protocol))}</strong><span>${escapeHtml(field(rag, ["retrieval_model"], "未标记模型"))}</span></div></div><span class="rag-flow-arrow">${icon("arrow-right")}</span><div class="rag-stage reranker"><span class="rag-stage-icon">${icon("list-filter")}</span><div><small>RERANK PROTOCOL</small><strong>${escapeHtml(ragProtocolLabel(rag.rerank_protocol))}</strong><span>${escapeHtml(field(rag, ["reranker_model"], "未标记模型"))}</span></div></div></div>${degradationNote}<div class="rag-hit-head"><span>TOP MATCHES</span><strong>${items.length} 个命中片段</strong></div><div class="rag-hit-list">${hits}</div></section>`;
  }

  function evidencePanel(items, fallbackPage) {
    if (!Array.isArray(items) || !items.length) {
      return `<section class="evidence-section"><div class="result-title"><div><span class="eyebrow">EVIDENCE LEDGER</span><h4>证据是什么</h4></div><span class="tag warning-tag">证据不足</span></div><div class="evidence-empty">${icon("search-x")}<span>模型没有返回可定位的证据。请结合目标 PDF 页人工核验答案。</span></div></section>`;
    }
    return `<section class="evidence-section"><div class="result-title"><div><span class="eyebrow">EVIDENCE LEDGER</span><h4>证据是什么</h4></div><span class="tag sage-tag">${items.length} 条可核验线索</span></div><div class="evidence-list">${items.map((item, index) => {
      const page = Number(field(item, ["page", "page_number"], 0));
      const source = field(item, ["source"], "target_image");
      const sourceIcon = source === "ocr" ? "file-scan" : source === "dataset_reference" ? "database" : "image";
      return `<article class="evidence-card source-${escapeHtml(source)}"><div class="evidence-index">${String(index + 1).padStart(2, "0")}</div><div class="evidence-copy"><div class="evidence-meta"><span>${icon(sourceIcon)}${escapeHtml(evidenceSourceLabel(source))}</span>${page > 0 ? `<button type="button" data-evidence-page="${page}" title="在 PDF 中查看第 ${page} 页">${icon("file-search")}P.${page}</button>` : `<span class="reference-label">参考标注</span>`}</div><blockquote>${escapeHtml(field(item, ["quote", "observation"], "模型未提供证据原文。"))}</blockquote><p>${escapeHtml(field(item, ["relevance", "reason"], "模型未说明该证据与答案的关系。"))}</p></div></article>`;
    }).join("")}</div>${fallbackPage ? `<button class="secondary-button evidence-fallback" type="button" data-evidence-page="${escapeHtml(fallbackPage)}">${icon("panel-top-open")}回到目标图页 P.${escapeHtml(fallbackPage)}</button>` : ""}</section>`;
  }

  function optionJudgementPanel(items, options, answer) {
    if (!Array.isArray(options) || !options.length) return "";
    const byOption = Object.fromEntries((Array.isArray(items) ? items : []).map(item => [String(field(item, ["option", "choice"], "")).toUpperCase(), item]));
    const labels = { selected: "选择", rejected: "排除", uncertain: "待核验" };
    const icons = { selected: "badge-check", rejected: "circle-x", uncertain: "circle-help" };
    return `<section class="option-judgement"><div class="result-title"><div><span class="eyebrow">OPTION REVIEW</span><h4>逐项判断</h4></div><span class="tag">为何选 / 为何不选</span></div><div class="option-judgement-list">${options.map(option => {
      const item = byOption[option.key] || {};
      const verdict = field(item, ["verdict"], option.key === answer ? "selected" : "uncertain");
      const reason = field(item, ["reason", "explanation"], option.key === answer ? "最终答案选择了该项。" : "模型未提供该项的排除理由。");
      return `<article class="option-judgement-row ${escapeHtml(verdict)}"><span class="option-letter">${escapeHtml(option.key)}</span><div><div class="option-verdict">${icon(icons[verdict] || "circle-help")}<strong>${labels[verdict] || "待核验"}</strong><span>${escapeHtml(option.value)}</span></div><p>${escapeHtml(reason)}</p></div></article>`;
    }).join("")}</div></section>`;
  }

  function semiFinalReportIntro(question, answer, explanation, reference) {
    return `<section class="semi-report-intro"><div class="result-title"><div><span class="eyebrow">ANSWER COMPARISON</span><h4>复赛开放问答 · 答案对照</h4></div><span class="tag">正文报告</span></div><div class="semi-report-question"><span>问题</span><p>${escapeHtml(question || "未提供问题文本")}</p></div><div class="semi-report-answer"><span>模型预测答案</span><p>${escapeHtml(answer || "答案已生成")}</p></div><div class="semi-report-reference"><span>数据集参考答案</span><p>${escapeHtml(reference || "当前分区没有参考答案")}</p></div>${explanation ? `<div class="semi-report-rationale"><span>回答说明</span><p>${escapeHtml(explanation)}</p></div>` : ""}</section>`;
  }

  function normalizedQuestionOptions(questionRecord) {
    const options = field(questionRecord, ["options", "choices"], []);
    if (Array.isArray(options)) {
      return options.map((value, index) => ({
        key: String.fromCharCode(65 + index),
        value: typeof value === "object" ? field(value, ["text", "value", "content"], "") : value
      }));
    }
    return Object.entries(options || {}).map(([key, value]) => ({ key, value }));
  }

  function resultLaunchCard(result, answer, explanation, evidenceItems, optionAnalysis, elapsed, mode) {
    if (S.stage === "semi_final") {
      return `<section class="result-launch-card semi-final-result"><div class="result-launch-heading"><span class="eyebrow">MODEL ANSWER · 复赛开放问答</span><span class="tag">${answer ? "已生成" : "等待生成"}</span></div><p class="semi-final-answer">${escapeHtml(answer || "答案已生成")}</p><button class="primary-button result-launch-button" id="open-analysis-viewer" type="button">${icon("panel-top-open")}查看完整分析</button></section>`;
    }
    const evidenceCount = Array.isArray(evidenceItems) ? evidenceItems.length : 0;
    const optionCount = Array.isArray(optionAnalysis)
      ? optionAnalysis.filter(item => !String(field(item, ["reason"], "")).includes("未返回")).length
      : 0;
    const auditComplete = field(result, ["audit_complete"], evidenceCount > 0 && optionCount >= 4);
    const statusIcon = auditComplete ? "badge-check" : "triangle-alert";
    const statusLabel = auditComplete ? "分析完成" : "完成 · 待人工复核";
    const ocrLabel = mode === "demo" || mode === "replay" ? "演示回放 · OCR 未调用" : field(result, ["ocr_enabled"], false) ? "OCR 上下文已开启" : "OCR 上下文已关闭";
    return `<section class="result-launch-card ${auditComplete ? "complete" : "incomplete"}"><div class="result-launch-top"><div class="answer-orb"><span>答案</span><strong>${escapeHtml(answer)}</strong></div><div class="result-launch-copy"><span class="eyebrow">MODEL RESULT</span><h4>分析结果已生成</h4><div class="result-summary"><span class="result-quality">${icon(statusIcon)}${statusLabel}</span><span>${icon("file-scan")}${escapeHtml(ocrLabel)}</span></div></div></div><button class="primary-button result-launch-button" id="open-analysis-viewer" type="button">${icon("panel-top-open")}查看完整分析</button></section>`;
  }

  function failureLaunchCard(result, trace) {
    const failedStage = (Array.isArray(trace) ? trace : []).find(item => field(item, ["status"], "") === "failed");
    const title = field(failedStage || {}, ["title"], "分析未完成");
    return `<section class="failure-launch-card"><div class="failure-launch-head"><span class="failure-icon">${icon("triangle-alert")}</span><div><span class="eyebrow">RUN INTERRUPTED</span><h4>${escapeHtml(title)}</h4><span class="failure-status">${icon("circle-x")}执行失败 · 未产生答案</span></div></div><button class="secondary-button result-launch-button" id="open-analysis-viewer" type="button">${icon("panel-top-open")}查看失败详情</button></section>`;
  }

  function analysisAuditContent() {
    const q = S.selectedQuestion || {};
    const result = T.unwrap(S.analysis) || {};
    const answer = field(result, ["answer", "final_answer", "prediction"], "");
    if (!answer && field(result, ["failed"], false)) {
      const trace = field(result, ["technical_trace"], S.analysisEvents.length ? S.analysisEvents : traceBlueprint());
      const message = field(result, ["message", "error"], "模型服务返回错误，请检查配置后重试。");
      const errorCode = field(result, ["error_code", "code"], "ANALYSIS_FAILED");
      const model = field(result.models || {}, ["vlm"], "--");
      return `<div class="audit-report-grid failure-report"><div class="audit-report-main"><section class="failure-report-hero"><span class="failure-icon large">${icon("triangle-alert")}</span><div><span class="eyebrow">ANALYSIS FAILED</span><h3>本次分析没有产生最终答案</h3><p>${escapeHtml(message)}</p></div></section><section class="failure-guidance"><div class="result-title"><div><span class="eyebrow">RECOVERY GUIDE</span><h4>系统会如何处理</h4></div><span class="tag">可恢复错误</span></div><div class="guidance-list"><div>${icon("refresh-cw")}<span><strong>自动结构化重试</strong><small>空内容或无效答案会使用更高输出预算自动重试一次。</small></span></div><div>${icon("braces")}<span><strong>JSON 输出约束</strong><small>要求模型只输出最终 JSON，不输出 think、Markdown 或模板。</small></span></div><div>${icon("scan-eye")}<span><strong>模型能力核验</strong><small>若仍失败，请确认当前 VLM 支持 image_url 多模态输入。</small></span></div></div></section><section class="failure-detail-grid"><div><span>错误代码</span><strong>${escapeHtml(errorCode)}</strong></div><div><span>是否可重试</span><strong>${field(result, ["retryable"], false) ? "是" : "否"}</strong></div><div><span>请求 ID</span><strong>${escapeHtml(field(result, ["request_id"], "--"))}</strong></div><div><span>推理模型</span><strong>${escapeHtml(model)}</strong></div></section></div><aside class="audit-report-side"><section class="trace-panel"><div class="result-title"><div><span class="eyebrow">EXECUTION TRACE</span><h4>失败发生在哪里</h4></div><span class="tag warning-tag">执行中断</span></div>${traceTimeline(trace)}</section></aside></div>`;
    }
    if (!answer) return emptyState("暂无完整分析", `完成一次${S.stage === "semi_final" ? "复赛开放问答" : "分析"}后可在这里查看答案、证据和执行链路。`);
    const questionText = field(q, ["question", "query", "text"], "未提供问题文本");
    const normalized = normalizedQuestionOptions(q);
    const rationale = field(result, ["rationale", "reason", "explanation"], "");
    const explanation = field(result, ["answer_explanation", "explanation"], rationale);
    const evidenceItems = field(result, ["evidence", "evidence_items"], []);
    const optionAnalysis = field(result, ["option_analysis", "choice_analysis"], []);
    const reasoningSummary = field(result, ["reasoning_summary", "analysis_summary"], []);
    const evidencePages = field(result, ["evidence_pages"], []);
    const evidencePage = field(result, ["evidence_page", "page", "page_number"], evidencePages[0] || S.pdfPage);
    const evidenceFallbackPage = S.stage === "preliminary" ? evidencePage : null;
    const trace = field(result, ["technical_trace"], S.analysisEvents.length ? S.analysisEvents : traceBlueprint());
    const elapsed = field(result, ["elapsed_ms", "latency_ms", "duration_ms"], null);
    const model = field(result, ["model", "vlm_model", "model_name"], field(result.models || {}, ["vlm"], "--"));
    const mode = field(result, ["mode"], S.analysisMode);
    const execution = result.execution || {};
    const metrics = result.metrics || {};
    const ocrText = field(result, ["ocr_text", "text", "recognized_text"], "");
    const corpusNote = ocrCorpusNote(result.ocr_corpus);
    const ocrPages = field(result, ["ocr_pages", "context_pages"], []);
    const warnings = field(result, ["output_warnings"], []);
    const auditComplete = field(result, ["audit_complete"], Array.isArray(evidenceItems) && evidenceItems.length > 0);
    const ragPanel = S.stage === "semi_final" && result.rag ? ragReportPanel(result.rag) : "";
    const referenceAnswer = field(q, ["answer", "reference_answer", "gold_answer"], "");
    const reportIntro = S.stage === "semi_final" ? semiFinalReportIntro(questionText, answer, explanation, referenceAnswer) : answerInsightPanel(answer, explanation, mode, result);
    const reportWarning = !auditComplete ? `<div class="audit-warning-callout">${icon("triangle-alert")}<div><strong>模型输出不完整，答案需人工复核</strong><p>${escapeHtml(Array.isArray(warnings) && warnings.length ? warnings.join("；") : "模型未按约定返回解释、证据或执行摘要。")}</p></div></div>` : "";
    const reportEvidence = evidencePanel(evidenceItems, evidenceFallbackPage);
    const reportOptions = S.stage === "preliminary" ? optionJudgementPanel(optionAnalysis, normalized, answer) : "";
    const ocrPageLabel = Array.isArray(ocrPages) && ocrPages.length ? ocrPages.map(item => `P.${escapeHtml(item)}`).join(", ") : S.stage === "preliminary" ? `P.${escapeHtml(evidencePage)}` : "文档 OCR 上下文";
    const reportMeta = S.stage === "semi_final"
      ? `${evaluationMetricCells(metrics)}<div class="meta-cell"><span>ROUGE-1 F1</span><strong>${metrics.rouge1_f1 == null ? "不适用" : formatMetric(metrics.rouge1_f1)}</strong></div><div class="meta-cell"><span>Exact Match</span><strong>${metrics.applicable ? formatMetric(metrics.exact_match) : "不适用"}</strong></div><div class="meta-cell"><span>耗时</span><strong>${elapsed == null ? "--" : `${formatNumber(elapsed)} ms`}</strong></div><div class="meta-cell"><span>推理模型</span><strong>${escapeHtml(model)}</strong></div><div class="meta-cell"><span>输出格式</span><strong>${escapeHtml(field(result, ["output_format"], "兼容格式"))}</strong></div><div class="meta-cell"><span>执行引擎</span><strong>${escapeHtml(field(execution, ["engine"], "--"))}</strong></div><div class="meta-cell"><span>协议</span><strong>${escapeHtml(field(execution, ["protocol"], "--"))}</strong></div>`
      : `<div class="meta-cell"><span>目标图页</span><strong>P.${escapeHtml(evidencePage)}</strong></div>${evaluationMetricCells(metrics)}<div class="meta-cell"><span>耗时</span><strong>${elapsed == null ? "--" : `${formatNumber(elapsed)} ms`}</strong></div><div class="meta-cell"><span>推理模型</span><strong>${escapeHtml(model)}</strong></div><div class="meta-cell"><span>输出格式</span><strong>${escapeHtml(field(result, ["output_format"], "兼容格式"))}</strong></div><div class="meta-cell"><span>参考答案</span><strong>${escapeHtml(field(q, ["answer", "reference_answer", "gold_answer"], "不适用"))}</strong></div><div class="meta-cell"><span>Accuracy</span><strong>${metrics.applicable ? formatMetric(metrics.accuracy) : "不适用"}</strong></div><div class="meta-cell"><span>执行引擎</span><strong>${escapeHtml(field(execution, ["engine"], "--"))}</strong></div><div class="meta-cell"><span>协议</span><strong>${escapeHtml(field(execution, ["protocol"], "--"))}</strong></div>`;
    return `<div class="audit-report-grid ${S.stage === "semi_final" ? "semi-final-audit" : ""}"><div class="audit-report-main">${reportIntro}${reportWarning}${ragPanel}${reportEvidence}${reportOptions}</div><aside class="audit-report-side"><section class="trace-panel"><div class="result-title"><div><span class="eyebrow">EXECUTION TRACE</span><h4>中间发生了什么</h4></div><span class="tag">执行完成</span></div>${traceTimeline(trace)}</section>${corpusNote}${ocrText ? `<div class="result-section"><div class="artifact-row"><span class="artifact-icon">${icon("file-scan")}</span><div><strong>OCR 识别文档</strong><span>${formatNumber(String(ocrText).length)} 字符 · ${ocrPageLabel}</span></div><button class="secondary-button" id="open-ocr-viewer-modal" type="button">${icon("file-scan")}查看 OCR 文档</button></div></div>` : ""}${Array.isArray(reasoningSummary) && reasoningSummary.length ? `<div class="result-section"><div class="result-title"><div><span class="eyebrow">AUDITABLE SUMMARY</span><h4>可审计分析摘要</h4></div><span class="tag">执行摘要</span></div><ol class="reasoning-list">${reasoningSummary.map((item, index) => `<li><span>${String(index + 1).padStart(2, "0")}</span><p>${escapeHtml(item)}</p></li>`).join("")}</ol></div>` : ""}<div class="meta-grid result-meta-grid">${reportMeta}</div></aside></div>`;
  }

  function analysisPanel() {
    const q = S.selectedQuestion;
    if (!q) return emptyState("尚未选择题目", "从题目队列选择一项后，可选择演示回放或真实评测。");
    const question = field(q, ["question", "query", "text"], "未提供问题文本");
    const normalized = normalizedQuestionOptions(q);

    const settings = T.unwrap(S.settings) || {};
    const ocrSettings = settings.ocr || {};
    const vlmSettings = settings.vlm || {};
    const vlmConfigured = Boolean(vlmSettings.configured);
    const ocrRequired = S.stage !== "preliminary" || S.ocrEnabled;
    const realConfigured = Boolean(vlmConfigured && (!ocrRequired || ocrSettings.configured));
    const busy = !["idle", "completed", "failed"].includes(S.analysisStatus);
    const canRun = !busy && (S.analysisMode === "demo" || realConfigured);
    const result = T.unwrap(S.analysis) || {};
    const answer = field(result, ["answer", "final_answer", "prediction"], "");
    const rationale = field(result, ["rationale", "reason", "explanation", "evidence"], "");
    const answerExplanation = field(result, ["answer_explanation", "explanation"], rationale);
    const evidenceItems = field(result, ["evidence", "evidence_items"], []);
    const optionAnalysis = field(result, ["option_analysis", "choice_analysis"], []);
    const reasoningSummary = field(result, ["reasoning_summary", "analysis_summary"], []);
    const ocrText = field(result, ["ocr_text", "text", "recognized_text"], "");
    const corpusNote = ocrCorpusNote(result.ocr_corpus);
    const ocrPages = field(result, ["ocr_pages", "context_pages"], []);
    const evidencePages = field(result, ["evidence_pages"], []);
    const evidencePage = field(result, ["evidence_page", "page", "page_number"], evidencePages[0] || S.pdfPage);
    const elapsed = field(result, ["elapsed_ms", "latency_ms", "duration_ms"], null);
    const model = field(result, ["model", "vlm_model", "model_name"], field(result.models || {}, ["vlm"], "--"));
    const mode = field(result, ["mode"], S.analysisMode);
    const metrics = result.metrics || {};
    const execution = result.execution || {};
    const trace = field(result, ["technical_trace"], S.analysisEvents.length ? S.analysisEvents : traceBlueprint());
    const ocrModel = field(ocrSettings, ["model"], "OCR 未配置");
    const vlmModel = field(vlmSettings, ["model"], "VLM 未配置");
    const modeCopy = S.analysisMode === "demo"
      ? { title: "参考答案回放引擎", detail: "读取数据集参考答案并执行格式校验；不会调用 OCR 或 VLM，也不会伪装成 AI 结果。", tech: "Reference Replay Engine" }
      : S.stage === "preliminary" && !S.ocrEnabled
      ? { title: "VLM 纯视觉基线", detail: `PyMuPDF 将目标页渲染为图片，只交给 ${vlmModel} 推理；不会调用 OCR。`, tech: "Vision-only Chat Completions" }
      : S.stage === "preliminary"
      ? { title: "其他页 OCR + 目标图 VLM", detail: `目标页作为图片交给 ${vlmModel}；最多 8 个其他页面经 ${ocrModel} 识别后作为带页码上下文。`, tech: "Context OCR + Multimodal VLM" }
      : { title: "OCR + VLM 真实评测", detail: `PyMuPDF 渲染页面，经 ${ocrModel} 识别后，将图像与文本交给 ${vlmModel} 联合分析。`, tech: "OpenAI-compatible Chat Completions" };
    const configurationMessage = !vlmConfigured ? "请先完整配置 VLM" : ocrRequired && !ocrSettings.configured ? "开启 OCR 需要先配置 OCR 模型" : "";

    return `<div class="analysis-content">
      <div class="question-detail"><div class="eyebrow">${icon("circle-help")}题目 #${escapeHtml(field(q, ["id", "question_id"], "--"))}</div><h3>${escapeHtml(question)}</h3>${normalized.length ? `<div class="options">${normalized.map(item => `<div class="option ${answer === item.key ? "selected" : ""}"><b>${escapeHtml(item.key)}</b><span>${escapeHtml(item.value)}</span>${answer === item.key ? icon("check") : ""}</div>`).join("")}</div>` : ""}</div>
      <section class="mode-console" aria-label="评测模式"><div class="mode-switch" role="radiogroup" aria-label="分析运行模式"><button type="button" data-analysis-mode="demo" class="${S.analysisMode === "demo" ? "active" : ""}" ${busy ? "disabled" : ""}>${icon("clapperboard")}演示回放</button><button type="button" data-analysis-mode="real" class="${S.analysisMode === "real" ? "active" : ""}" ${busy || !vlmConfigured ? "disabled" : ""}>${icon("sparkles")}真实评测</button></div><div class="mode-guide"><div><span class="mode-thread"></span><strong>${escapeHtml(modeCopy.title)}</strong></div><p>${escapeHtml(modeCopy.detail)}</p><span class="tech-label">${escapeHtml(modeCopy.tech)}</span>${S.analysisMode === "real" && !realConfigured ? `<small class="status-text error">${escapeHtml(configurationMessage)}</small>` : ""}</div>${S.analysisMode === "real" && S.stage === "preliminary" ? `<label class="ocr-context-control"><span><strong>其他页面 OCR 上下文</strong><small>关闭后运行纯视觉基线；切换不会清除另一组结果</small></span><span class="switch"><input id="ocr-context-toggle" type="checkbox" ${S.ocrEnabled ? "checked" : ""} ${busy ? "disabled" : ""}><span class="switch-track"></span></span></label>` : ""}${S.stage === "semi_final" ? `<label class="ocr-context-control"><span><strong>语义评审（额外模型调用）</strong><small>仅复赛开放问答生效；开启后额外调用一次 LLM 评审，产出语义 F1 与代理分，默认关闭以节省模型费用。该分数不是官方 BERTScore。</small></span><span class="switch"><input id="semantic-toggle" type="checkbox" ${S.evaluateSemantic ? "checked" : ""} ${busy ? "disabled" : ""}><span class="switch-track"></span></span></label>` : ""}</section>
      <div class="analysis-action"><button class="primary-button ${busy ? "loading-button" : ""}" id="run-analysis" type="button" ${canRun ? "" : "disabled"}>${busy ? icon("loader-circle") + "正在执行真实阶段" : icon(S.analysisMode === "real" ? "sparkles" : "play") + (S.analysisStatus === "failed" ? "重新运行" : S.analysisMode === "real" && S.stage === "preliminary" ? S.ocrEnabled ? "运行 OCR 增强" : "运行纯视觉基线" : S.analysisMode === "real" ? "运行真实评测" : "运行演示回放")}</button><button class="icon-button" id="reset-analysis" type="button" title="清除结果" aria-label="清除结果" ${S.analysis || S.analysisEvents.length || S.comparisonResults?.withOcr || S.comparisonResults?.withoutOcr ? "" : "disabled"}>${icon("rotate-ccw")}</button></div>
      ${S.analysisMode === "real" && S.stage === "preliminary" ? ocrComparisonPanel() : ""}
      ${S.analysisStatus === "failed" ? failureLaunchCard(result, trace) : answer ? resultLaunchCard(result, answer, answerExplanation, evidenceItems, optionAnalysis, elapsed, mode) : `<section class="trace-panel"><div class="result-title"><div><span class="eyebrow">EXECUTION TRACE</span><h4>中间发生了什么</h4></div><span class="tag">${busy ? "实时更新" : "等待运行"}</span></div>${traceTimeline(trace)}</section>`}
    </div>`;
  }

  function batch() {
    return `<section class="page batch-page" data-page="batch">${pageHeader("批量评测", "调用真实 OCR、RAG 与 VLM，逐题追踪执行阶段并汇总竞赛指标", `<span class="mode-badge real-badge">${icon("sparkles")}真实模型</span><button class="secondary-button" id="batch-audit-open" type="button" ${activeBatchJob() ? "" : "disabled"}>${icon("gallery-horizontal-end")}逐题审计</button><button class="secondary-button" id="jobs-refresh" type="button">${icon("refresh-cw")}<span>刷新任务</span></button>`)}<div class="content-pad batch-canvas">
      <section id="batch-dashboard">${batchDashboard()}</section>
      <div class="batch-command-grid"><div class="panel batch-create-panel"><div class="panel-head"><div><h2>新建真实评测</h2><p>任务创建后直接调用当前服务端模型配置</p></div><span class="tag">REAL RUN</span></div><form class="panel-body" id="batch-form"><div class="batch-real-contract">${icon("shield-check")}<span><strong>真实模型调用已锁定</strong><small>不会读取参考答案生成结果；训练集参考答案仅用于最终指标计算。</small></span></div><div class="form-grid"><div class="form-field"><label for="batch-stage">比赛阶段</label><select id="batch-stage" name="stage"><option value="preliminary" ${S.stage === "preliminary" ? "selected" : ""}>初赛 · 四选一</option><option value="semi_final" ${S.stage === "semi_final" ? "selected" : ""}>复赛 · 开放问答</option></select></div><div class="form-field"><label for="batch-split">数据分区</label><select id="batch-split" name="split"><option value="train">训练集</option><option value="test">测试集</option></select></div><div class="form-field"><label for="batch-offset">起始位置</label><input class="field-input mono" id="batch-offset" name="offset" type="number" min="0" value="0"></div><div class="form-field"><label for="batch-limit">题目数量</label><input class="field-input mono" id="batch-limit" name="limit" type="number" min="1" max="200" value="10"><small>建议先用 5–10 题验证成本和限流</small></div><div class="form-field"><label for="batch-concurrency">并发数</label><input class="field-input mono" id="batch-concurrency" name="concurrency" type="number" min="1" max="4" value="1"><small>真实模型建议 1；缓存充足时可设 2</small></div><div class="form-field"><label>OCR 上下文</label><label class="switch compact-switch"><input id="batch-use-ocr" name="use_ocr" type="checkbox" checked><span class="switch-track"></span><span id="batch-ocr-label">${S.stage === "preliminary" ? "其他页面 OCR" : "缓存优先 OCR"}</span></label><small id="batch-ocr-help">${S.stage === "preliminary" ? "关闭后运行纯视觉对照批次" : "复赛优先读取分页 TXT 缓存"}</small></div><div class="form-field"><label>语义评审（额外模型调用）</label><label class="switch compact-switch"><input id="batch-evaluate-semantic" name="evaluate_semantic" type="checkbox"><span class="switch-track"></span><span>默认关闭</span></label><small>仅复赛开放问答生效；开启后每题额外调用一次 LLM 评审并产出代理分，不是官方 BERTScore。</small></div><div class="form-field"><label>输出指标</label><div class="secret-note">${icon("chart-no-axes-column")}<span id="metric-hint">${S.stage === "preliminary" ? "Accuracy" : "Exact Match / ROUGE-1 F1"}</span></div></div></div><div class="form-actions"><button class="primary-button batch-launch" id="batch-submit" type="submit">${icon("play")}启动真实评测</button></div></form></div>
      <section id="batch-live-console">${batchLiveConsole()}</section></div>
      <div class="panel batch-queue-panel"><div class="panel-head"><div><h2>任务队列</h2><p>运行状态每 2 秒自动刷新，可导出严格 JSONL</p></div><span class="tag" id="job-count">${S.jobs.length} 个任务</span></div><div id="jobs-list">${jobsList()}</div></div>
    </div></section>`;
  }

  const batchStages = [
    ["locating", "file-search", "文档定位"], ["ocr", "file-scan", "OCR"], ["indexing", "database", "索引"],
    ["retrieval", "search-code", "召回"], ["reranking", "list-filter", "重排"], ["reasoning", "brain-circuit", "VLM"],
    ["validation", "badge-check", "评测"]
  ];

  function activeBatchJob() {
    return S.jobs.find(job => ["running", "queued", "pending"].includes(field(job, ["status", "state"], ""))) || S.jobs[0] || null;
  }

  function batchDashboard() {
    const job = activeBatchJob();
    const metrics = field(job || {}, ["metrics"], {});
    const total = Number(field(job || {}, ["total"], 0));
    const completed = Number(field(job || {}, ["completed"], 0));
    const success = Number(field(job || {}, ["success"], 0));
    const failed = Number(field(job || {}, ["failed"], 0));
    const progress = Math.round(Number(field(job || {}, ["progress"], total ? completed / total : 0)) * 100);
    const successRate = Number(field(job || {}, ["success_rate"], completed ? success / completed : 0));
    const preliminary = field(job || {}, ["stage"], S.stage) === "preliminary";
    const metric = preliminary ? metrics.accuracy : metrics.semantic_proxy_score;
    const averageDuration = field(job || {}, ["average_duration_ms", "average_ms"], null);
    const metricLabel = preliminary ? "Accuracy" : "代理分（非官方）";
    const metricDetail = preliminary
      ? "随任务实时聚合"
      : `ROUGE ${formatMetric(metrics.rouge1_f1)} · EM ${formatMetric(metrics.exact_match)} · 官方综合分需离线 BERTScore`;
    const stageSummary = field(job || {}, ["stage_summary"], {});
    const errorCodes = field(job || {}, ["error_code_counts"], {});
    const stageEntries = Object.entries(stageSummary || {});
    const errorEntries = Object.entries(errorCodes || {}).sort((left, right) => Number(right[1]) - Number(left[1]));
    const stageSummaryBlock = stageEntries.length ? `<div class="batch-stage-summary"><div class="result-title"><div><span class="eyebrow">STAGE SUMMARY</span><h4>阶段汇总</h4></div><span class="tag">逐题聚合</span></div><div class="batch-stage-summary-grid">${stageEntries.map(([stage, counts]) => `<div class="batch-stage-summary-item"><strong>${escapeHtml(batchStages.find(([name]) => name === stage)?.[2] || stage)}</strong><span>完成 ${formatNumber(counts.completed)} · 跳过 ${formatNumber(counts.skipped)} · 失败 ${formatNumber(counts.failed)}</span></div>`).join("")}</div></div>` : "";
    const errorCodeBlock = errorEntries.length ? `<div class="batch-error-codes"><div class="result-title"><div><span class="eyebrow">ERROR CODES</span><h4>失败码分布</h4></div><span class="tag">${errorEntries.length} 类</span></div><ul>${errorEntries.map(([code, count]) => `<li><code>${escapeHtml(code)}</code><span>${formatNumber(count)} 次</span></li>`).join("")}</ul></div>` : "";
    return `<div class="batch-dashboard-head"><div><span class="eyebrow">EVALUATION COMMAND CENTER</span><h2>真实评测看板</h2><p>${job ? `当前聚焦任务 ${escapeHtml(field(job, ["job_id"], "--"))}` : "启动任务后，这里将展示真实执行指标"}</p></div><span class="status-text ${job ? stateClass(field(job, ["status"], "queued")) : "info"}"><span class="status-dot"></span>${job ? statusName(field(job, ["status"], "queued")) : "等待任务"}</span></div><div class="batch-kpi-grid"><article><span>整体进度</span><strong>${progress}<small>%</small></strong><p>${completed} / ${total || 0} 题完成</p></article><article><span>成功率</span><strong>${formatMetric(successRate)}</strong><p>${success} 成功 · ${failed} 失败</p></article><article><span>${metricLabel}</span><strong>${metric == null ? "--" : formatMetric(metric)}</strong><p>${metricDetail}</p></article><article><span>平均耗时</span><strong>${averageDuration == null ? "--" : formatNumber(averageDuration)}${averageDuration == null ? "" : "<small> ms</small>"}</strong><p>单题端到端耗时</p></article></div><div class="batch-master-progress"><div><span>PIPELINE PROGRESS</span><strong>${progress}%</strong></div><div class="progress-track"><div class="progress-bar" style="width:${progress}%"></div></div></div><div class="batch-dashboard-detail">${stageSummaryBlock}${errorCodeBlock}</div>`;
  }

  function batchLiveConsole() {
    const job = activeBatchJob();
    if (!job) return `<div class="panel batch-live-panel"><div class="panel-head"><div><h2>实时执行轨迹</h2><p>当前没有运行中的批次</p></div><span class="tag">IDLE</span></div>${emptyState("等待真实评测", "任务启动后将显示当前题目和 OCR、RAG、VLM 阶段。")}</div>`;
    const jobStage = field(job, ["stage"], "preliminary");
    const stagePlan = jobStage === "preliminary"
      ? batchStages.filter(([name]) => ["locating", "ocr", "reasoning", "validation"].includes(name))
      : batchStages;
    const items = field(job, ["items", "item_states", "results"], []);
    const runningItems = Array.isArray(items) ? items.filter(item => field(item, ["status"], "") === "running") : [];
    const focus = runningItems[0] || (Array.isArray(items) ? items.find(item => field(item, ["status"], "") === "queued") : null) || {};
    const currentStage = field(focus, ["current_stage"], field(job, ["current_stage"], field(job, ["status"], "queued") === "queued" ? "queued" : "locating"));
    const stageIndex = stagePlan.findIndex(([name]) => name === currentStage);
    const title = field(focus, ["stage_title", "title"], currentStage === "queued" ? "等待执行槽位" : "正在执行模型流水线");
    const detail = field(focus, ["stage_detail", "detail"], `已完成 ${field(job, ["completed"], 0)} / ${field(job, ["total"], 0)} 题`);
    const technology = field(focus, ["stage_technology", "technology"], field(job, ["mode"], "real") === "real" ? "OpenAI-compatible Model Pipeline" : "Reference Replay");
    const focusQuestion = field(focus, ["question"], "");
    const stageTrack = stagePlan.map(([name, iconName, label], index) => { const status = index < stageIndex ? "done" : index === stageIndex ? "active" : "waiting"; return `<div class="batch-stage ${status}"><span>${icon(status === "done" ? "check" : iconName)}</span><strong>${label}</strong></div>`; }).join("");
    const running = runningItems.length ? runningItems.map(item => `<span class="batch-running-item">${icon("loader-circle")}题目 #${escapeHtml(field(item, ["id", "question_id"], "--"))} · ${escapeHtml(field(item, ["stage_title", "title"], title))}</span>`).join("") : `<span class="batch-running-item muted">${icon("clock-3")}等待下一道题</span>`;
    const completedItems = Array.isArray(items) ? items.filter(item => ["completed", "failed"].includes(field(item, ["status"], ""))).slice(-4).reverse() : [];
    const auditCards = completedItems.map(item => {
      const summary = Array.isArray(field(item, ["reasoning_summary"], [])) ? field(item, ["reasoning_summary"], []) : [];
      const evidence = Array.isArray(field(item, ["evidence"], [])) ? field(item, ["evidence"], []) : [];
      const answer = field(item, ["answer"], "--");
      const errorMessage = field(field(item, ["error"], {}), ["message"], "暂无答案解释");
      return `<article class="batch-audit-card ${field(item, ["status"], "") === "failed" ? "failed" : ""}"><div class="batch-audit-head"><strong>题目 #${escapeHtml(field(item, ["id"], "--"))}</strong><span>${field(item, ["status"], "") === "failed" ? "失败" : `答案 ${escapeHtml(answer)}`}</span></div>${field(item, ["question"], "") ? `<p class="batch-audit-question">${escapeHtml(field(item, ["question"], ""))}</p>` : ""}<p class="batch-audit-explanation">${escapeHtml(field(item, ["answer_explanation"], errorMessage))}</p>${summary.length ? `<ol>${summary.slice(0, 2).map(step => `<li>${escapeHtml(step)}</li>`).join("")}</ol>` : ""}${evidence.length ? `<div class="batch-audit-evidence">${icon("quote")}${field(evidence[0], ["page"], null) ? `P.${escapeHtml(field(evidence[0], ["page"], "--"))} · ` : ""}${escapeHtml(field(evidence[0], ["quote"], "已返回证据"))}</div>` : ""}</article>`;
    }).join("");
    return `<div class="panel batch-live-panel"><div class="panel-head"><div><h2>实时执行轨迹</h2><p>${stageName(jobStage)} · ${splitName(field(job, ["split"], "train"))} · 并发 ${escapeHtml(field(job, ["concurrency"], runningItems.length || 1))}</p></div><span class="live-chip"><i></i>LIVE</span></div><div class="batch-live-body"><div class="batch-current-stage"><span class="batch-stage-emblem">${icon(stagePlan[Math.max(0, stageIndex)]?.[1] || "loader-circle")}</span><div><span class="eyebrow">CURRENT QUESTION · #${escapeHtml(field(focus, ["id"], "--"))} · ${escapeHtml(String(currentStage).toUpperCase())}</span><h3>${escapeHtml(title)}</h3>${focusQuestion ? `<p class="batch-current-question">${escapeHtml(focusQuestion)}</p>` : ""}<p>${escapeHtml(detail)}</p><small>${escapeHtml(technology)}</small></div></div><div class="batch-stage-track">${stageTrack}</div><div class="batch-running-list">${running}</div>${auditCards ? `<div class="batch-audit-console"><div class="batch-audit-console-head"><div><span class="eyebrow">AUDITABLE ANSWERS</span><h3>已完成题目 · 答案与推理</h3></div><span class="tag">${completedItems.length} 条</span></div><div class="batch-audit-grid">${auditCards}</div></div>` : ""}</div></div>`;
  }

  function jobsList(error) {
    if (error) return errorState(error, "jobs-retry");
    if (!S.jobs.length) return emptyState("暂无批量任务", "在左侧设置范围并启动一次小规模评测。" );
    return S.jobs.map(job => {
      const id = field(job, ["job_id", "id"], "--");
      const status = field(job, ["status", "state"], "pending");
      const total = Number(field(job, ["total", "total_count", "count"], 0));
      const completed = Number(field(job, ["completed", "processed", "completed_count"], 0));
      const failed = Number(field(job, ["failed", "failed_count"], 0));
      const success = Number(field(job, ["success", "success_count"], Math.max(0, completed - failed)));
      const reportedProgress = Number(field(job, ["progress"], NaN));
      const percent = Number.isFinite(reportedProgress) ? Math.min(100, Math.round(reportedProgress * 100)) : total ? Math.min(100, Math.round(completed / total * 100)) : status === "completed" ? 100 : 0;
      const metrics = field(job, ["metrics", "metric_summary"], {});
      const metricText = metrics && metrics.applicable ? [metrics.accuracy != null ? `Accuracy ${formatMetric(metrics.accuracy)}` : "", metrics.final_score != null ? `Final ${formatMetric(metrics.final_score)}` : "", metrics.semantic_f1 != null ? `语义 ${formatMetric(metrics.semantic_f1)}` : "", metrics.rouge1_f1 != null ? `ROUGE-1 ${formatMetric(metrics.rouge1_f1)}` : "", metrics.exact_match != null ? `EM ${formatMetric(metrics.exact_match)}` : ""].filter(Boolean).join(" · ") : "指标不适用";
      const ocrLabel = field(job, ["stage"], "preliminary") === "preliminary" ? field(job, ["use_ocr"], true) ? "其他页 OCR" : "纯视觉基线" : "文档 RAG/OCR";
      const currentStage = field(job, ["current_stage"], "");
      const demoBatch = field(job, ["mode"], "real") === "demo";
      const modeBanner = demoBatch
        ? `<div class="job-mode-banner demo"><span class="status-dot warning"></span><strong>演示批次</strong><span>参考答案回放，不代表真实模型性能，不可用于模型效果结论。</span></div>`
        : `<div class="job-mode-banner real"><span class="status-dot"></span><strong>真实模型</strong><span>调用已配置的 OCR / VLM / Retrieval / Reranker 服务。</span></div>`;
      return `<article class="job-row"><div class="job-main"><div class="job-title-line"><strong class="mono">${escapeHtml(id)}</strong><span class="status-text ${stateClass(status)}"><span class="status-dot ${stateClass(status) === "error" ? "error" : stateClass(status) === "warning" ? "warning" : ""}"></span>${statusName(status)}</span></div><small>${stageName(field(job, ["stage"], "preliminary"))} · ${splitName(field(job, ["split"], "train"))} · 真实模型 · ${ocrLabel} · ${metricText}</small>${modeBanner}${currentStage ? `<span class="job-stage-now">${icon("activity")}当前：${escapeHtml(batchStages.find(([name]) => name === currentStage)?.[2] || currentStage)}</span>` : ""}<div class="progress-track job-progress"><div class="progress-bar" style="width:${percent}%"></div></div><div class="progress-summary"><div><span>进度</span><strong>${percent}%</strong></div><div><span>成功</span><strong>${success}</strong></div><div><span>失败</span><strong>${failed}</strong></div><div><span>平均耗时</span><strong>${formatNumber(field(job, ["average_duration_ms", "average_ms", "avg_elapsed_ms"], 0))} ms</strong></div></div></div><div class="job-actions">${["running", "pending", "queued"].includes(status) ? `<button class="danger-button" type="button" data-cancel-job="${escapeHtml(id)}">${icon("square")}取消</button>` : ""}${failed && !["running", "pending", "queued"].includes(status) ? `<button class="secondary-button" type="button" data-retry-job="${escapeHtml(id)}">${icon("rotate-ccw")}重试失败项</button>` : ""}${status === "completed" || status === "partial" ? `<a class="secondary-button" href="${window.InduDocAPI.exportUrl(id)}" download>${icon("download")}导出 JSONL</a><a class="secondary-button" href="${window.InduDocAPI.jobSummaryUrl(id)}" download="${id}-summary.json">${icon("clipboard-list")}导出摘要 JSON</a>` : ""}</div></article>`;
    }).join("");
  }

  function runs() {
    return `<section class="page" data-page="runs">${pageHeader("运行记录", "当前服务进程内最近的单题分析与批量任务摘要", `<button class="secondary-button" id="runs-refresh" type="button">${icon("refresh-cw")}<span>刷新记录</span></button>`)}<div class="content-pad"><div class="panel"><div class="panel-head"><div><h2>最近 100 条</h2><p>密钥与 Authorization 信息不会在此显示</p></div><span class="tag">内存记录</span></div><div id="runs-list">${runsList()}</div></div></div></section>`;
  }

  function runsList(error) {
    if (error) return errorState(error, "runs-retry");
    if (!S.runs.length) return emptyState("暂无运行记录", "完成单题分析或批量评测后将在此显示。" );
    return `<div style="overflow:auto"><table class="runs-table"><thead><tr><th>时间</th><th>类型</th><th>对象 ID</th><th>阶段</th><th>状态</th><th>耗时</th><th>模式 / 模型</th></tr></thead><tbody>${S.runs.map(run => { const status = field(run, ["status"], "unknown"); const mode = field(run, ["mode"], "--"); const variant = mode === "real" && field(run, ["stage"], "preliminary") === "preliminary" ? field(run, ["ocr_enabled"], field(run, ["use_ocr"], true)) ? "OCR 增强" : "纯视觉" : mode; return `<tr><td class="mono">${formatTime(field(run, ["created_at", "timestamp", "time"]))}</td><td>${field(run, ["type", "run_type"], "single") === "batch" ? "批量" : "单题"}</td><td class="mono">${escapeHtml(field(run, ["question_id", "job_id", "id"], "--"))}</td><td>${stageName(field(run, ["stage"], "preliminary"))}</td><td><span class="status-text ${stateClass(status)}">${statusName(status)}</span></td><td class="mono">${formatNumber(field(run, ["elapsed_ms", "duration_ms"], 0))} ms</td><td>${escapeHtml(variant)} / ${escapeHtml(field(run, ["model", "model_name"], "--"))}</td></tr>`; }).join("")}</tbody></table></div>`;
  }

  function settings() {
    const data = T.unwrap(S.settings) || {};
    const ocr = data.ocr || {};
    const vlm = data.vlm || {};
    const retrieval = data.retrieval || {};
    const reranker = data.reranker || {};
    const demo = field(data, ["demo_mode"], true);
    const persistence = data.persistence || {};
    return `<section class="page" data-page="settings">${pageHeader("模型设置", "OCR、VLM、检索与重排序使用独立 OpenAI-compatible 端点，保存后写入本机配置", `<span class="mode-badge">API Key 永不回显</span>`)}<div class="content-pad"><form id="settings-form" class="settings-layout"><div class="panel"><div class="panel-head"><div><h2>运行模式</h2><p>真实模型不可用时保持演示链路可运行</p></div></div><div class="panel-body"><label class="switch"><input id="demo-mode" name="demo_mode" type="checkbox" ${demo ? "checked" : ""}><span class="switch-track"></span><span>启用演示模式</span></label><div class="secret-note" style="margin-top:14px">${icon("shield-check")}<span>API Key 会保存到服务端本机的 runtime/model-settings.json，不会写入浏览器存储，也不会通过接口回显。${persistence.enabled === false ? "当前未启用本地持久化。" : "服务重启后会自动加载。"}</span></div></div></div><div class="panel model-settings-panel"><div class="model-section ocr-section">${modelForm("ocr", "OCR 识别模型", ocr, "file-scan")}</div><div class="model-section vlm-section">${modelForm("vlm", "VLM 推理模型", vlm, "brain-circuit")}</div><div class="model-section retrieval-section">${modelForm("retrieval", "Retrieval 检索模型", retrieval, "search-code")}</div><div class="model-section reranker-section">${modelForm("reranker", "Reranker 重排序模型", reranker, "list-filter")}</div><div class="form-actions" style="padding:0 16px 16px"><button class="secondary-button" id="settings-reset" type="button">${icon("rotate-ccw")}重新加载</button><button class="primary-button" id="settings-save" type="submit">${icon("save")}保存配置</button></div></div></form></div></section>`;
  }

  function modelForm(kind, title, data, iconName) {
    const configured = field(data, ["api_key_set", "configured", "has_api_key"], false);
    const temperatureField = kind === "vlm" ? `<div class="form-field"><label for="${kind}-temperature">Temperature</label><input class="field-input mono" id="${kind}-temperature" name="${kind}_temperature" type="number" min="0" max="2" step="0.1" value="${escapeHtml(field(data, ["temperature"], 0))}"></div>` : `<div></div>`;
    const providerGuide = kind === "retrieval" ? `<div class="provider-guide">${icon("boxes")}<span><strong>硅基流动 Embedding 示例</strong><small>Base URL: https://api.siliconflow.cn/v1 · Model: Qwen/Qwen3-Embedding-4B · 调用 /embeddings</small></span></div>` : kind === "reranker" ? `<div class="provider-guide">${icon("list-filter")}<span><strong>硅基流动 Reranker 示例</strong><small>Base URL: https://api.siliconflow.cn/v1 · Model: Qwen/Qwen3-Reranker-4B · 调用 /rerank</small></span></div>` : "";
    return `<div class="model-section-title"><h3>${icon(iconName)}${title}</h3><span class="status-text ${configured ? "success" : "warning"}"><span class="status-dot ${configured ? "" : "warning"}"></span>${configured ? "已配置密钥" : "未配置密钥"}</span></div>${providerGuide}<div class="form-grid"><div class="form-field full"><label for="${kind}-url">Base URL</label><input class="field-input mono" id="${kind}-url" name="${kind}_base_url" value="${escapeHtml(field(data, ["base_url", "url"], ""))}" placeholder="https://api.example.com/v1"></div><div class="form-field"><label for="${kind}-model">Model</label><input class="field-input mono" id="${kind}-model" name="${kind}_model" value="${escapeHtml(field(data, ["model", "model_name"], ""))}" placeholder="model-name"></div><div class="form-field"><label for="${kind}-key">API Key</label><div class="key-row"><input class="field-input mono" id="${kind}-key" name="${kind}_api_key" type="password" autocomplete="new-password" placeholder="${configured ? "已配置 · 留空保持不变" : "输入服务密钥"}"><button class="icon-button" type="button" data-toggle-key="${kind}" title="显示或隐藏本次输入" aria-label="显示或隐藏本次输入">${icon("eye")}</button></div></div><div class="form-field"><label for="${kind}-timeout">Timeout（秒）</label><input class="field-input mono" id="${kind}-timeout" name="${kind}_timeout" type="number" min="5" max="600" value="${escapeHtml(field(data, ["timeout", "timeout_seconds"], 120))}"></div>${temperatureField}</div><div class="form-actions"><span class="status-text" id="${kind}-test-state"></span><button class="secondary-button" type="button" data-test-model="${kind}">${icon("plug-zap")}测试连接</button></div>`;
  }

  function formatMetric(value) {
    return value == null ? "不适用" : `${(Number(value) * 100).toFixed(1)}%`;
  }

  // The official composite score needs a real BERTScore run (offline). Online we only
  // expose an LLM-judge score plus a clearly labelled proxy, so nothing here may look
  // like an official number.
  function semanticStatusLabel(metrics) {
    const status = field(metrics, ["semantic_status"], "");
    if (status === "completed") return "已完成";
    if (status === "unavailable") return "未启用";
    if (status === "failed" || status === "error") return field(metrics, ["semantic_error"], "评审失败");
    return "不适用";
  }

  function scoreCell(label, value, note) {
    return `<div class="meta-cell"><span>${escapeHtml(label)}</span><strong>${value == null ? "未产出" : formatMetric(value)}</strong>${note ? `<small>${escapeHtml(note)}</small>` : ""}</div>`;
  }

  function hasEvaluationContract(metrics) {
    return ["semantic_status", "semantic_proxy_score", "semantic_proxy_status", "final_score_status", "bert_f1_status"]
      .some(key => metrics && metrics[key] !== undefined && metrics[key] !== null);
  }

  function evaluationMetricCells(metrics) {
    if (!hasEvaluationContract(metrics)) return "";
    const bertReason = field(metrics, ["bert_f1_reason"], "官方 BERTScore 需在离线评测中用 bert-base-chinese 计算");
    const finalStatus = field(metrics, ["final_score_status"], "不完整（缺真实 BERTScore）");
    return [
      scoreCell("代理分（非官方）", metrics.semantic_proxy_score, "0.5×语义 + 0.25×ROUGE-1 + 0.25×EM"),
      scoreCell("官方综合分", metrics.final_score, finalStatus),
      scoreCell("语义评审 F1", metrics.semantic_f1, "LLM 评审结果，非官方 BERTScore"),
      scoreCell("官方 BERTScore", metrics.bert_f1, bertReason),
      `<div class="meta-cell"><span>语义评审模型</span><strong>${escapeHtml(field(metrics, ["semantic_model"], "未标记"))}</strong></div>`,
      `<div class="meta-cell"><span>语义评审状态</span><strong>${escapeHtml(semanticStatusLabel(metrics))}</strong></div>`
    ].join("");
  }

  function ocrCorpusNote(corpus) {
    if (!corpus || typeof corpus !== "object") return "";
    if (field(corpus, ["source"], "none") === "none") return "";
    const generated = field(corpus, ["model"], "未标记模型");
    const expected = field(corpus, ["expected_model"], "未标记模型");
    const stale = Boolean(field(corpus, ["stale"], false));
    const reason = field(corpus, ["stale_reason"], "");
    const pages = field(corpus, ["completed_pages"], null);
    return `<div class="corpus-note ${stale ? "stale" : "fresh"}">${icon(stale ? "triangle-alert" : "badge-check")}<div><strong>${stale ? "预计算语料已过期，改用实时 OCR" : "预计算语料可用"}</strong><p>由 ${escapeHtml(generated)} 生成，当前配置为 ${escapeHtml(expected)}${stale ? `；${escapeHtml(reason || "指纹或模型不一致")}` : "；源 PDF 指纹一致"}${pages != null ? `；已完成 ${formatNumber(pages)} 页` : ""}</p></div></div>`;
  }

  window.InduDocViews = Object.freeze({
    overview, overviewContent, workspace, questionList, analysisPanel,
    batch, batchDashboard, batchLiveConsole, jobsList, runs, runsList, settings,
    loadingState, errorState, emptyState, escapeHtml, icon, field, statusName, analysisAuditContent, ocrCorpusNote
  });
})();
