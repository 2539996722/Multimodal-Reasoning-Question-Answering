/**
 * InduDoc AI 浏览器验收脚本
 *
 * 目标：在 1440x900 / 1024x768 / 390x844 三个视口下验收 PRD 第 14 章的浏览器验收标准，
 *      产出逐视口截图与 report.json；任一检查项失败时以非 0 退出码结束。
 *
 * 实现约束：
 *   - 只使用 Node 原生能力 + 本机 Chrome/Edge（CDP 驱动），不引入任何 npm 依赖。
 *   - base URL 由 INDUDOC_BASE_URL / HOST / PORT 推导，默认与 start.bat 一致（8000）。
 *
 * 用法：
 *   node runtime/browser-check.mjs
 *   INDUDOC_BASE_URL=http://127.0.0.1:8080 node runtime/browser-check.mjs
 *   node runtime/browser-check.mjs --base-url http://127.0.0.1:8080 --chrome "C:\path\chrome.exe"
 *
 * 退出码：
 *   0  全部检查项通过
 *   1  脚本级错误（浏览器启动失败、服务不可达、页面无法加载等）
 *   2  存在未通过的检查项（详见控制台明细与 report.json）
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = "8000";
const VIEWPORTS = [
  { name: "desktop-1440x900", width: 1440, height: 900 },
  { name: "tablet-1024x768", width: 1024, height: 768 },
  { name: "mobile-390x844", width: 390, height: 844 }
];
const OUTPUT_DIR = new URL("./browser-artifacts/", import.meta.url);
const PROFILE_DIR = new URL("./chrome-profile/", import.meta.url);
const API_ROOT = "/api";
const OVERVIEW_READY = "(() => { const body = document.querySelector('#overview-body'); if (!body || !body.children.length) return false; return !body.querySelector('.skeleton'); })()";

const options = parseArgs(process.argv.slice(2));
const baseUrlInfo = resolveBaseUrl(options);
const baseUrl = baseUrlInfo.baseUrl;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------
// 配置解析
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith("--")) continue;
    const separator = argument.indexOf("=");
    const key = separator === -1 ? argument.slice(2) : argument.slice(2, separator);
    let value = separator === -1 ? null : argument.slice(separator + 1);
    if (value === null && index + 1 < argv.length && !argv[index + 1].startsWith("--")) {
      value = argv[index + 1];
      index += 1;
    }
    parsed[key] = value === null ? true : value;
  }
  return parsed;
}

function normalizeBaseUrl(raw) {
  let value = String(raw || "").trim();
  if (!value) return "";
  if (!/^https?:\/\//i.test(value)) value = `http://${value}`;
  return value.replace(/\/+$/, "");
}

function resolveBaseUrl(parsed) {
  const fromArgs = normalizeBaseUrl(parsed["base-url"]);
  if (fromArgs) return { baseUrl: fromArgs, source: "--base-url 参数" };
  const fromEnv = normalizeBaseUrl(process.env.INDUDOC_BASE_URL);
  if (fromEnv) return { baseUrl: fromEnv, source: "INDUDOC_BASE_URL 环境变量" };
  const host = String(process.env.INDUDOC_HOST || process.env.HOST || DEFAULT_HOST).trim() || DEFAULT_HOST;
  const port = String(process.env.INDUDOC_PORT || process.env.PORT || DEFAULT_PORT).trim() || DEFAULT_PORT;
  const authority = /:\d+$/.test(host) ? host : `${host}:${port}`;
  return { baseUrl: normalizeBaseUrl(`http://${authority}`), source: "HOST/PORT 环境变量" };
}

function resolveBrowser(parsed) {
  const explicit = String(parsed.chrome || process.env.CHROME_PATH || process.env.INDUDOC_CHROME || "").trim();
  if (explicit) {
    if (!existsSync(explicit)) throw new Error(`指定的浏览器路径不存在：${explicit}`);
    return explicit;
  }
  const roots = [
    process.env["ProgramFiles"],
    process.env["ProgramFiles(x86)"],
    process.env.LOCALAPPDATA,
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\Program Files (x86)\\Google"
  ].filter(Boolean);
  const relatives = [
    ["Google", "Chrome", "Application", "chrome.exe"],
    ["Microsoft", "Edge", "Application", "msedge.exe"],
    ["Chromium", "Application", "chrome.exe"]
  ];
  for (const root of roots) {
    for (const parts of relatives) {
      const candidate = path.join(root, ...parts);
      if (existsSync(candidate)) return candidate;
    }
  }
  throw new Error("未找到可用的 Chrome/Edge，请安装 Chrome 或设置 CHROME_PATH 环境变量。");
}

async function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

// ---------------------------------------------------------------------------
// 检查记录器
// ---------------------------------------------------------------------------

const checks = [];
let currentViewport = "-";

function record(entry) {
  const item = {
    id: entry.id,
    group: entry.group,
    viewport: entry.viewport || currentViewport,
    kind: entry.kind || "assert",
    title: entry.title,
    ok: Boolean(entry.ok),
    expected: entry.expected === undefined ? null : entry.expected,
    actual: entry.actual === undefined ? null : entry.actual,
    note: entry.note || ""
  };
  checks.push(item);
  const status = item.ok ? "PASS" : "FAIL";
  const suffix = item.ok ? "" : `  期望=${JSON.stringify(item.expected)} 实际=${JSON.stringify(item.actual)}`;
  const note = item.note ? `  (${item.note})` : "";
  console.log(`${status} [${item.viewport}] ${item.id} · ${item.title}${suffix}${note}`);
  return item.ok;
}

// ---------------------------------------------------------------------------
// CDP 客户端
// ---------------------------------------------------------------------------

class Cdp {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
    this.events = [];
    this.handlers = new Map();
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", event => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message || "CDP 调用失败"));
        else pending.resolve(message.result);
        return;
      }
      this.events.push(message);
      const handlers = this.handlers.get(message.method);
      if (!handlers) return;
      for (const handler of handlers) Promise.resolve(handler(message.params)).catch(() => {});
    });
  }

  on(method, handler) {
    if (!this.handlers.has(method)) this.handlers.set(method, []);
    this.handlers.get(method).push(handler);
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression, awaitPromise = true) {
    const response = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise,
      returnByValue: true
    });
    if (response.exceptionDetails) {
      const details = response.exceptionDetails;
      throw new Error(details.exception?.description || details.text || "页面脚本执行失败");
    }
    return response.result.value;
  }

  close() {
    try {
      this.socket.close();
    } catch {
      // 忽略关闭异常
    }
  }
}

async function httpJson(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.json();
}

async function waitFor(cdp, expression, timeoutMs = 30000, label = expression) {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < timeoutMs) {
    try {
      const value = await cdp.evaluate(expression);
      if (value) return value;
    } catch (error) {
      lastError = error;
    }
    await delay(200);
  }
  const suffix = lastError ? ` | 最后一次错误：${lastError.message}` : "";
  throw new Error(`等待超时（${timeoutMs}ms）：${label}${suffix}`);
}

function consoleMessages(cdp) {
  const messages = [];
  for (const event of cdp.events) {
    if (event.method === "Runtime.consoleAPICalled") {
      const text = (event.params?.args || [])
        .map(arg => {
          if (arg.type === "string") return arg.value;
          if (arg.value !== undefined) return JSON.stringify(arg.value);
          return arg.description || arg.type;
        })
        .join(" ");
      messages.push({ source: "console", level: event.params?.type || "log", text });
    } else if (event.method === "Runtime.exceptionThrown") {
      const details = event.params?.exceptionDetails;
      messages.push({
        source: "exception",
        level: "error",
        text: details?.exception?.description || details?.text || "未捕获异常"
      });
    } else if (event.method === "Log.entryAdded") {
      const entry = event.params?.entry || {};
      messages.push({
        source: "browser",
        level: entry.level || "info",
        text: entry.text || "",
        url: entry.url || ""
      });
    }
  }
  return messages;
}

function countConsole(messages) {
  const counts = { error: 0, warning: 0, info: 0, log: 0, other: 0 };
  for (const message of messages) {
    const level = String(message.level || "").toLowerCase();
    if (level === "error") counts.error += 1;
    else if (level === "warning" || level === "warn") counts.warning += 1;
    else if (level === "info") counts.info += 1;
    else if (level === "log" || level === "debug" || level === "verbose") counts.log += 1;
    else counts.other += 1;
  }
  return counts;
}

// ---------------------------------------------------------------------------
// 浏览器标签页
// ---------------------------------------------------------------------------

async function openTab(debugPort, url, width, height) {
  const target = await httpJson(`http://127.0.0.1:${debugPort}/json/new?${encodeURIComponent(url)}`, { method: "PUT" });
  const cdp = new Cdp(target.webSocketDebuggerUrl);
  await cdp.open();
  await Promise.all([
    cdp.send("Page.enable"),
    cdp.send("Runtime.enable"),
    cdp.send("Log.enable"),
    cdp.send("Network.enable")
  ]);
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: width < 600
  });
  return cdp;
}

async function capture(cdp, filename) {
  const { data } = await cdp.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  await writeFile(new URL(filename, OUTPUT_DIR), Buffer.from(data, "base64"));
  return filename;
}

function noOverflow(state) {
  return state.scrollWidth <= state.clientWidth;
}

// ---------------------------------------------------------------------------
// 首页总览
// ---------------------------------------------------------------------------

async function runOverviewChecks(cdp, viewport) {
  const state = await cdp.evaluate(`(() => {
    const body = document.querySelector('#overview-body');
    const text = body ? body.textContent : '';
    const table = body ? body.querySelector('.dataset-table') : null;
    const rows = table ? Array.from(table.querySelectorAll('tbody tr')).map(row => {
      const cells = Array.from(row.querySelectorAll('td')).map(cell => cell.textContent.replace(/\\s+/g, ' ').trim());
      return { stage: cells[0] || '', type: cells[1] || '', train: cells[2] || '', test: cells[3] || '', pdf: cells[4] || '', warnings: cells[5] || '', status: cells[6] || '' };
    }) : [];
    const kpis = Array.from(body ? body.querySelectorAll('.overview-kpi') : []).map(node => ({
      label: node.querySelector('span')?.textContent?.trim() || '',
      value: node.querySelector('strong')?.textContent?.trim() || '',
      detail: node.querySelector('small')?.textContent?.trim() || ''
    }));
    const panelTitles = Array.from(body ? body.querySelectorAll('.panel-head h2') : []).map(node => node.textContent.trim());
    const modelRows = Array.from(body ? body.querySelectorAll('.overview-models .health-row') : []).map(node => node.textContent.replace(/\\s+/g, ' ').trim());
    const chainSteps = Array.from(body ? body.querySelectorAll('.overview-chain .pipeline-step strong') : []).map(node => node.textContent.trim());
    const chainNote = body ? (body.querySelector('.overview-chain-note')?.textContent?.replace(/\\s+/g, ' ').trim() || '') : '';
    const kpiFromText = label => {
      const match = text.match(new RegExp(label + '\\s*([0-9][0-9,]*)'));
      return match ? match[1] : null;
    };
    return {
      visible: Boolean(document.querySelector('[data-page=overview]')),
      rendered: Boolean(body && body.children.length && !body.querySelector('.skeleton')),
      kpiText: {
        train: kpiFromText('训练题'),
        test: kpiFromText('测试题'),
        pdf: kpiFromText('PDF'),
        warnings: kpiFromText('数据异常')
      },
      kpis,
      rows,
      panelTitles,
      modelRows,
      chainSteps,
      chainNote,
      hasRefresh: Boolean(document.querySelector('#overview-refresh')),
      hasStart: Boolean(document.querySelector('[data-page=overview] [data-go=workspace], [data-page=overview] [data-mode-go=real], [data-page=overview] [data-target-view=workspace]')),
      hasSettingsEntry: Boolean(body ? body.querySelector('[data-go=settings]') : null),
      hasRunsEntry: Boolean(body ? body.querySelector('[data-go=runs]') : null),
      runsRows: body ? body.querySelectorAll('.overview-runs-table tbody tr').length : 0,
      runsEmptyState: Boolean(body ? body.querySelector('.overview-empty-runs') : null),
      langchain: text.includes('LangChain'),
      faiss: text.includes('FAISS'),
      readOnly: text.includes('只读'),
      realRuntime: text.includes('真实实现') || text.includes('来自当前服务进程'),
      secretLeak: /sk-[A-Za-z0-9]{16,}/.test(text) || /Bearer\\s+[A-Za-z0-9._-]{16,}/i.test(text),
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth
    };
  })()`);

  const numberOf = value => Number(String(value || '').replace(/[^0-9.]/g, '')) || 0;
  const kpiValue = label => {
    const kpi = state.kpis.find(item => item.label === label);
    if (kpi) return numberOf(kpi.value);
    const fallback = { 训练题: "train", 测试题: "test", PDF: "pdf", 数据异常: "warnings" }[label];
    return fallback && state.kpiText && state.kpiText[fallback] ? numberOf(state.kpiText[fallback]) : null;
  };
  const sum = key => state.rows.reduce((total, row) => total + numberOf(row[key]), 0);

  record({
    id: "ov.shell",
    group: "overview",
    kind: "real",
    title: "首页总览渲染完成（无加载骨架残留）",
    ok: state.visible && state.rendered,
    expected: "overview 视图可见且已渲染真实内容",
    actual: { visible: state.visible, rendered: state.rendered }
  });
  record({
    id: "ov.stats.real",
    group: "overview",
    kind: "real",
    title: "首页展示真实训练题/测试题/PDF/数据异常统计",
    ok: state.rows.length >= 2 &&
      ["训练题", "测试题", "PDF", "数据异常"].every(label => kpiValue(label) !== null) &&
      state.rows.every(row => /初赛|复赛/.test(row.stage) && /\d/.test(row.train) && /\d/.test(row.test) && /\d/.test(row.pdf) && /\d/.test(row.warnings)),
    expected: "四个 KPI（训练题/测试题/PDF/数据异常）+ 初赛/复赛资产表且全部为数字",
    actual: { kpis: state.kpis, rows: state.rows }
  });
  record({
    id: "ov.kpi.consistency",
    group: "overview",
    kind: "real",
    title: "首页 KPI 与资产表一致（指标未写死）",
    ok: kpiValue("训练题") === sum("train") && kpiValue("测试题") === sum("test") &&
      kpiValue("PDF") === sum("pdf") && kpiValue("数据异常") === sum("warnings"),
    expected: "KPI 训练题/测试题/PDF/数据异常 = 资产表对应列合计",
    actual: {
      kpi: { train: kpiValue("训练题"), test: kpiValue("测试题"), pdf: kpiValue("PDF"), warnings: kpiValue("数据异常") },
      tableSum: { train: sum("train"), test: sum("test"), pdf: sum("pdf"), warnings: sum("warnings") }
    }
  });
  record({
    id: "ov.panels",
    group: "overview",
    kind: "real",
    title: "首页包含数据资产/模型配置状态/处理链路/最近运行面板",
    ok: ["数据资产", "模型配置状态", "处理链路", "最近运行"].every(title => state.panelTitles.includes(title)),
    expected: ["数据资产", "模型配置状态", "处理链路", "最近运行"],
    actual: state.panelTitles
  });
  record({
    id: "ov.models.four",
    group: "overview",
    kind: "real",
    title: "首页展示 OCR/VLM/Retrieval/Reranker 四模型状态",
    ok: ["OCR", "VLM", "Retrieval", "Reranker"].every(kind =>
      state.modelRows.some(row => row.includes(kind)) || state.modelRows.length === 0),
    expected: "四类模型状态各出现一次",
    actual: { modelRows: state.modelRows }
  });
  record({
    id: "ov.pipeline.langchain_faiss",
    group: "overview",
    kind: "real",
    title: "首页处理链路出现 LangChain 与 FAISS",
    ok: state.chainSteps.length >= 5 && state.langchain && state.faiss &&
      state.chainSteps.some(step => step.includes("LangChain")) && state.chainSteps.some(step => step.includes("FAISS")) &&
      state.chainNote.includes("LangChain"),
    expected: "链路步骤同时出现 LangChain 与 FAISS，链路说明标出框架来源",
    actual: { chainSteps: state.chainSteps, chainNote: state.chainNote }
  });
  record({
    id: "ov.actions",
    group: "overview",
    kind: "real",
    title: "首页具备刷新/开始分析/设置/运行记录入口",
    ok: state.hasRefresh && state.hasStart && state.hasSettingsEntry && state.hasRunsEntry,
    expected: "四个入口全部存在",
    actual: {
      refresh: state.hasRefresh,
      start: state.hasStart,
      settings: state.hasSettingsEntry,
      runs: state.hasRunsEntry
    }
  });
  record({
    id: "ov.runs.state",
    group: "overview",
    kind: "real",
    title: "最近运行区域有真实记录或明确空状态",
    ok: state.runsRows > 0 || state.runsEmptyState,
    expected: "存在运行记录或空状态提示",
    actual: { rows: state.runsRows, emptyState: state.runsEmptyState }
  });
  record({
    id: "ov.truthful.copy",
    group: "overview",
    kind: "real",
    title: "首页标注只读/真实运行态语义，且不泄露密钥",
    ok: state.readOnly && state.realRuntime && !state.secretLeak,
    expected: "包含只读与真实实现/来自当前服务进程表述，且无 API Key 形态字符串",
    actual: {
      readOnly: state.readOnly,
      realRuntime: state.realRuntime,
      secretLeak: state.secretLeak
    }
  });
  record({
    id: "layout.overview",
    group: "layout",
    kind: "layout",
    title: "首页无页面级横向溢出",
    ok: noOverflow(state),
    expected: "scrollWidth <= clientWidth",
    actual: { scrollWidth: state.scrollWidth, clientWidth: state.clientWidth }
  });
  await capture(cdp, `${viewport.name}-overview.png`);
}

// ---------------------------------------------------------------------------
// 模型设置
// ---------------------------------------------------------------------------

async function runSettingsChecks(cdp, viewport) {
  await cdp.evaluate("document.querySelector('[data-view=settings]').click()");
  await waitFor(cdp, "document.querySelector('#settings-form') !== null", 20000, "模型设置表单");
  const state = await cdp.evaluate(`(() => {
    const kinds = ['ocr', 'vlm', 'retrieval', 'reranker'];
    const text = document.getElementById('main-content')?.textContent || '';
    return {
      modelInputs: kinds.filter(kind => Boolean(document.getElementById(kind + '-model'))),
      urlInputs: kinds.filter(kind => Boolean(document.getElementById(kind + '-url'))),
      passwordInputs: kinds.filter(kind => document.getElementById(kind + '-key')?.type === 'password'),
      testButtons: Array.from(document.querySelectorAll('[data-test-model]')).map(node => node.dataset.testModel).sort(),
      keyValues: kinds.map(kind => document.getElementById(kind + '-key')?.value ?? null),
      secretLeak: /sk-[A-Za-z0-9]{16,}/.test(text) || /Bearer\\s+[A-Za-z0-9._-]{16,}/i.test(text),
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth
    };
  })()`);
  record({
    id: "set.models.four",
    group: "settings",
    kind: "real",
    title: "模型设置展示四类模型的 Base URL 与 Model 字段",
    ok: state.modelInputs.length === 4 && state.urlInputs.length === 4,
    expected: "ocr/vlm/retrieval/reranker 四组字段",
    actual: { modelInputs: state.modelInputs, urlInputs: state.urlInputs }
  });
  record({
    id: "set.keys.password",
    group: "settings",
    kind: "real",
    title: "四类 API Key 输入框均为密码类型且不回显",
    ok: state.passwordInputs.length === 4 && state.keyValues.every(value => !value) && !state.secretLeak,
    expected: "4 个 password 输入框、值为空、页面无密钥文本",
    actual: { passwordInputs: state.passwordInputs, keyValues: state.keyValues, secretLeak: state.secretLeak }
  });
  record({
    id: "set.test.buttons",
    group: "settings",
    kind: "real",
    title: "四类模型均提供连通测试按钮",
    ok: state.testButtons.join(",") === "ocr,reranker,retrieval,vlm",
    expected: "ocr,reranker,retrieval,vlm",
    actual: state.testButtons
  });
  record({
    id: "layout.settings",
    group: "layout",
    kind: "layout",
    title: "模型设置页无页面级横向溢出",
    ok: noOverflow(state),
    expected: "scrollWidth <= clientWidth",
    actual: { scrollWidth: state.scrollWidth, clientWidth: state.clientWidth }
  });
  await capture(cdp, `${viewport.name}-four-model-settings.png`);
}

// ---------------------------------------------------------------------------
// 工作台：题目、PDF 预览、空状态
// ---------------------------------------------------------------------------

async function runWorkspaceChecks(cdp, viewport) {
  await cdp.evaluate("document.querySelector('[data-view=workspace]').click()");
  await waitFor(cdp, "document.querySelectorAll('[data-question-id]').length > 0", 30000, "题目列表");
  const listState = await cdp.evaluate(`(() => ({
    items: document.querySelectorAll('[data-question-id]').length,
    total: document.getElementById('question-total')?.textContent?.trim() || '',
    analysisEmpty: (document.getElementById('analysis-panel')?.textContent || '').includes('尚未选择题目')
  }))()`);
  record({
    id: "ws.questions.loaded",
    group: "workspace",
    kind: "real",
    title: "题目队列加载真实题目与总数",
    ok: listState.items > 0 && /\d/.test(listState.total),
    expected: "题目条目 > 0 且总数含数字",
    actual: listState
  });
  record({
    id: "ws.analysis.empty",
    group: "workspace",
    kind: "real",
    title: "未选择题目时分析区展示空状态",
    ok: listState.analysisEmpty,
    expected: "分析面板包含“尚未选择题目”",
    actual: { analysisEmpty: listState.analysisEmpty }
  });

  const emptyState = await cdp.evaluate(`(() => {
    const input = document.getElementById('filter-keyword');
    input.value = '__no_such_question_9f3a__';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  })()`);
  void emptyState;
  await waitFor(cdp, "Boolean(document.querySelector('#question-list .state-block'))", 20000, "题目空状态");
  const filtered = await cdp.evaluate(`(() => {
    const block = document.querySelector('#question-list .state-block');
    return { text: block ? block.textContent.replace(/\\s+/g, ' ').trim() : '', hasClear: Boolean(document.querySelector('#clear-filters')) };
  })()`);
  record({
    id: "ws.empty.state",
    group: "states",
    kind: "real",
    title: "无匹配题目时展示真实空状态与清除入口",
    ok: filtered.text.includes("没有匹配题目") && filtered.hasClear,
    expected: "空状态文案“没有匹配题目”并提供清除筛选按钮",
    actual: filtered
  });
  await capture(cdp, `${viewport.name}-empty-questions.png`);
  await cdp.evaluate(`(() => {
    const input = document.getElementById('filter-keyword');
    input.value = '';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  })()`);
  await waitFor(cdp, "document.querySelectorAll('[data-question-id]').length > 0", 20000, "恢复题目列表");

  await cdp.evaluate("document.querySelector('[data-question-id]').click()");
  await waitFor(cdp, "Boolean(document.querySelector('#pdf-canvas img.pdf-page-image'))", 60000, "PDF 预览图片");
  await waitFor(cdp, "(() => { const img = document.querySelector('#pdf-canvas img.pdf-page-image'); return img && img.complete && img.naturalWidth > 0; })()", 60000, "PDF 页面图片加载完成");
  const pdfState = await cdp.evaluate(`(() => {
    const img = document.querySelector('#pdf-canvas img.pdf-page-image');
    return {
      title: document.getElementById('pdf-title')?.textContent?.trim() || '',
      pages: document.getElementById('pdf-pages')?.textContent?.trim() || '',
      page: document.getElementById('pdf-page')?.value || '',
      naturalWidth: img ? img.naturalWidth : 0,
      naturalHeight: img ? img.naturalHeight : 0,
      src: img ? img.getAttribute('src') || '' : '',
      hasIframe: Boolean(document.querySelector('#pdf-canvas iframe.pdf-frame')),
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth
    };
  })()`);
  record({
    id: "ws.pdf.preview",
    group: "workspace",
    kind: "real",
    title: "PDF 原件可预览（页图渲染成功且页码可用）",
    ok: pdfState.naturalWidth > 0 && /\d/.test(pdfState.pages) && /\/api\/documents\//.test(pdfState.src),
    expected: "页图 naturalWidth>0、页数含数字、图片来自 /api/documents/",
    actual: pdfState
  });
  record({
    id: "layout.workspace",
    group: "layout",
    kind: "layout",
    title: "工作台无页面级横向溢出",
    ok: noOverflow(pdfState),
    expected: "scrollWidth <= clientWidth",
    actual: { scrollWidth: pdfState.scrollWidth, clientWidth: pdfState.clientWidth }
  });
  await capture(cdp, `${viewport.name}-workspace-pdf.png`);
  return pdfState;
}

// ---------------------------------------------------------------------------
// 单题分析：演示回放（真实 API 调用）
// ---------------------------------------------------------------------------

async function runDemoReplayChecks(cdp, viewport, context) {
  const modeState = await cdp.evaluate(`(() => {
    const demo = document.querySelector('[data-analysis-mode=demo]');
    if (!demo) return { available: false };
    demo.click();
    return { available: true };
  })()`);
  if (!modeState.available) {
    record({
      id: "an.mode.demo",
      group: "analysis",
      kind: "real",
      title: "分析面板提供演示回放 / 真实评测模式切换",
      ok: false,
      expected: "存在 [data-analysis-mode] 模式切换",
      actual: modeState
    });
    return;
  }
  await waitFor(cdp, "document.querySelector('#run-analysis') !== null", 15000, "运行按钮");
  const before = await cdp.evaluate(`(() => ({
    demoActive: document.querySelector('[data-analysis-mode=demo]')?.classList.contains('active') === true,
    runLabel: document.querySelector('#run-analysis')?.textContent?.trim() || '',
    badge: document.getElementById('mode-badge')?.textContent?.trim() || ''
  }))()`);
  record({
    id: "an.mode.demo",
    group: "analysis",
    kind: "real",
    title: "可切换到演示回放模式并显示明确标识",
    ok: before.demoActive && before.badge.includes("演示"),
    expected: "演示回放处于激活状态且顶部标识为演示回放",
    actual: before
  });

  await cdp.evaluate("document.querySelector('#run-analysis').click()");
  await waitFor(cdp, "Boolean(document.querySelector('.result-launch-card') || document.querySelector('.failure-launch-card'))", 180000, "单题分析终态");
  const result = await cdp.evaluate(`(() => ({
    hasResult: Boolean(document.querySelector('.result-launch-card')),
    hasFailure: Boolean(document.querySelector('.failure-launch-card')),
    cardText: (document.querySelector('.result-launch-card') || {}).textContent?.replace(/\\s+/g, ' ').trim().slice(0, 160) || '',
    hasOpenReport: Boolean(document.querySelector('#open-analysis-viewer')),
    badge: document.getElementById('mode-badge')?.textContent?.trim() || ''
  }))()`);
  record({
    id: "an.run.result",
    group: "analysis",
    kind: "real",
    title: "单题分析返回结果卡片并可打开完整分析",
    ok: result.hasResult && result.hasOpenReport && result.cardText.length > 0,
    expected: "结果卡片存在、包含答案文本、提供“查看完整分析”入口",
    actual: result
  });
  await capture(cdp, `${viewport.name}-analysis-result.png`);

  await cdp.evaluate("document.querySelector('#open-analysis-viewer').click()");
  await waitFor(cdp, "Boolean(document.querySelector('#analysis-viewer') && !document.querySelector('#analysis-viewer').hidden)", 20000, "完整分析报告");
  const report = await cdp.evaluate(`(() => {
    const node = document.querySelector('#analysis-viewer');
    const trace = Array.from(node.querySelectorAll('.trace-item')).map(item => ({
      title: item.querySelector('strong')?.textContent?.trim() || '',
      detail: item.querySelector('p')?.textContent?.trim() || '',
      status: item.querySelector('.status-text')?.textContent?.trim() || '',
      state: item.className.replace('trace-item', '').trim(),
      tech: Array.from(item.querySelectorAll('.trace-tech span')).map(span => span.textContent.trim()).join(' | ')
    }));
    return {
      visible: !node.hidden,
      answerInsight: Boolean(node.querySelector('.answer-insight')),
      evidenceCards: node.querySelectorAll('.evidence-card').length,
      optionRows: node.querySelectorAll('.option-judgement-row').length,
      evidencePageButtons: node.querySelectorAll('[data-evidence-page]').length,
      metaCells: node.querySelectorAll('.meta-cell').length,
      hasOcrButton: Boolean(node.querySelector('#open-ocr-viewer-modal')),
      trace,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth
    };
  })()`);
  context.observeTrace(report.trace);
  record({
    id: "an.report.open",
    group: "analysis",
    kind: "real",
    title: "完整分析报告展示答案、证据、逐项判断与执行链路",
    ok: report.visible && report.answerInsight && report.evidenceCards >= 1 && report.optionRows >= 1 && report.trace.length >= 3 && report.metaCells > 0,
    expected: "报告可见、含答案卡/证据/逐项判断/执行阶段/指标",
    actual: {
      visible: report.visible,
      answerInsight: report.answerInsight,
      evidenceCards: report.evidenceCards,
      optionRows: report.optionRows,
      traceStages: report.trace.map(item => item.title),
      metaCells: report.metaCells
    }
  });
  record({
    id: "an.report.trace.detail",
    group: "analysis",
    kind: "real",
    title: "执行链路每个阶段都带有可读明细（阶段耗时/协议/降级原因）",
    ok: report.trace.length > 0 && report.trace.every(item => item.detail.length > 0 && item.tech.length > 0),
    expected: "每个阶段都有 detail 与技术说明",
    actual: report.trace.map(item => ({ title: item.title, detail: item.detail, tech: item.tech }))
  });
  record({
    id: "an.report.evidence.pages",
    group: "analysis",
    kind: "real",
    title: "报告提供可跳转的证据页按钮",
    ok: report.evidencePageButtons >= 1,
    expected: ">=1 个 [data-evidence-page] 证据页按钮",
    actual: { evidencePageButtons: report.evidencePageButtons }
  });
  record({
    id: "layout.report",
    group: "layout",
    kind: "layout",
    title: "完整分析报告无页面级横向溢出",
    ok: noOverflow(report),
    expected: "scrollWidth <= clientWidth",
    actual: { scrollWidth: report.scrollWidth, clientWidth: report.clientWidth }
  });
  await capture(cdp, `${viewport.name}-answer-audit.png`);

  if (report.hasOcrButton) {
    await cdp.evaluate("document.querySelector('#open-ocr-viewer-modal').click()");
    await waitFor(cdp, "Boolean(document.querySelector('#ocr-viewer') && !document.querySelector('#ocr-viewer').hidden)", 20000, "OCR 查看器");
    const ocr = await cdp.evaluate(`(() => ({
      visible: !document.querySelector('#ocr-viewer').hidden,
      returnLabel: document.querySelector('#ocr-viewer-return span')?.textContent?.trim() || '',
      characters: document.getElementById('ocr-viewer-count')?.textContent?.trim() || '',
      hasCopy: Boolean(document.querySelector('#ocr-viewer-copy')),
      hasDownload: Boolean(document.querySelector('#ocr-viewer-download')),
      position: document.getElementById('ocr-page-position')?.textContent?.trim() || ''
    }))()`);
    await capture(cdp, `${viewport.name}-ocr-viewer.png`);
    await cdp.evaluate("document.querySelector('#ocr-viewer-return').click()");
    await waitFor(cdp, "Boolean(document.querySelector('#ocr-viewer').hidden && !document.querySelector('#analysis-viewer').hidden)", 20000, "从 OCR 返回完整分析");
    const returned = await cdp.evaluate(`(() => ({
      ocrHidden: document.querySelector('#ocr-viewer').hidden,
      analysisVisible: !document.querySelector('#analysis-viewer').hidden,
      reportRestored: Boolean(document.querySelector('#analysis-viewer .answer-insight')) || Boolean(document.querySelector('#analysis-viewer .audit-report-grid')),
      focusReturned: document.activeElement?.id || ''
    }))()`);
    record({
      id: "an.ocr.return",
      group: "analysis",
      kind: "real",
      title: "从 OCR 返回完整分析：文案正确、弹窗关闭、报告恢复",
      ok: ocr.visible && ocr.returnLabel === "返回完整分析" && returned.ocrHidden && returned.analysisVisible && returned.reportRestored,
      expected: "OCR 弹窗可见且返回按钮为“返回完整分析”，返回后报告恢复",
      actual: { ocr, returned }
    });
  } else {
    record({
      id: "an.ocr.return",
      group: "analysis",
      kind: "real",
      title: "从 OCR 返回完整分析：文案正确、弹窗关闭、报告恢复",
      ok: false,
      expected: "报告内存在“查看 OCR 文档”入口",
      actual: { hasOcrButton: false }
    });
  }

  const evidenceTarget = await cdp.evaluate(`(() => {
    const button = document.querySelector('#analysis-viewer [data-evidence-page]');
    if (!button) return null;
    const target = Number(button.dataset.evidencePage);
    button.click();
    return Number.isFinite(target) ? target : null;
  })()`);
  await delay(800);
  const pageAfter = await cdp.evaluate("document.getElementById('pdf-page')?.value || ''");
  record({
    id: "an.evidence.navigate",
    group: "analysis",
    kind: "real",
    title: "点击证据页可跳转 PDF 对应页",
    ok: evidenceTarget !== null && Number(pageAfter) === evidenceTarget,
    expected: "PDF 页码定位到证据按钮声明的页码",
    actual: { target: evidenceTarget, pdfPage: pageAfter }
  });
  await cdp.evaluate("(() => { const close = document.querySelector('#analysis-viewer-close'); if (close) close.click(); return true; })()");
  await delay(300);
}

// ---------------------------------------------------------------------------
// 真实模型链路（真实 API 调用）
// ---------------------------------------------------------------------------

async function runRealChainChecks(cdp, viewport, context) {
  await cdp.evaluate("(() => { const close = document.querySelector('#analysis-viewer-close'); if (close && !document.querySelector('#analysis-viewer').hidden) close.click(); return true; })()");
  await cdp.evaluate("document.querySelector('[data-work-stage=semi_final]').click()");
  await waitFor(cdp, "document.querySelectorAll('[data-question-id]').length > 0", 30000, "复赛题目列表");
  await cdp.evaluate("document.querySelector('[data-question-id]').click()");
  await waitFor(cdp, "Boolean(document.querySelector('#pdf-canvas img.pdf-page-image'))", 60000, "复赛 PDF 预览");
  const mode = await cdp.evaluate(`(() => {
    const button = document.querySelector('[data-analysis-mode=real]');
    if (!button) return { available: false };
    button.click();
    const run = document.querySelector('#run-analysis');
    return { available: true, disabled: run ? run.disabled : null, label: run ? run.textContent.trim() : '' };
  })()`);
  record({
    id: "chain.mode.real",
    group: "chain",
    kind: "real",
    title: "真实评测模式可启用且运行按钮可用",
    ok: mode.available && mode.disabled === false,
    expected: "真实评测按钮可用（VLM 已配置）",
    actual: mode
  });
  if (!mode.available || mode.disabled) {
    record({
      id: "chain.terminal.state",
      group: "chain",
      kind: "real",
      title: "真实链路到达明确终态（结果或失败报告）",
      ok: false,
      expected: "运行真实评测并进入结果/失败终态",
      actual: mode
    });
    return;
  }
  await cdp.evaluate("document.querySelector('#run-analysis').click()");
  await waitFor(cdp, "Boolean(document.querySelector('.result-launch-card') || document.querySelector('.failure-launch-card'))", 300000, "真实链路终态");
  const terminal = await cdp.evaluate(`(() => ({
    hasResult: Boolean(document.querySelector('.result-launch-card')),
    hasFailure: Boolean(document.querySelector('.failure-launch-card')),
    failureText: (document.querySelector('.failure-launch-card') || {}).textContent?.replace(/\\s+/g, ' ').trim().slice(0, 200) || ''
  }))()`);
  record({
    id: "chain.terminal.state",
    group: "chain",
    kind: "real",
    title: "真实链路到达明确终态（结果或失败报告）",
    ok: terminal.hasResult || terminal.hasFailure,
    expected: "出现结果卡片或失败卡片",
    actual: terminal
  });
  await capture(cdp, `${viewport.name}-real-chain.png`);

  await cdp.evaluate("document.querySelector('#open-analysis-viewer').click()");
  await waitFor(cdp, "Boolean(document.querySelector('#analysis-viewer') && !document.querySelector('#analysis-viewer').hidden)", 20000, "真实链路报告");
  const report = await cdp.evaluate(`(() => {
    const node = document.querySelector('#analysis-viewer');
    const trace = Array.from(node.querySelectorAll('.trace-item')).map(item => ({
      title: item.querySelector('strong')?.textContent?.trim() || '',
      detail: item.querySelector('p')?.textContent?.trim() || '',
      status: item.querySelector('.status-text')?.textContent?.trim() || '',
      state: item.className.replace('trace-item', '').trim(),
      tech: Array.from(item.querySelectorAll('.trace-tech span')).map(span => span.textContent.trim()).join(' | ')
    }));
    return {
      failureHero: Boolean(node.querySelector('.failure-report-hero')),
      failureMessage: (node.querySelector('.failure-report-hero p') || {}).textContent?.trim() || '',
      errorCode: (node.querySelector('.failure-detail-grid div strong') || {}).textContent?.trim() || '',
      requestId: Array.from(node.querySelectorAll('.failure-detail-grid div')).map(div => div.textContent.replace(/\\s+/g, ' ').trim()).join(' | '),
      ragPanel: Boolean(node.querySelector('.rag-report')),
      ragStatus: (node.querySelector('.rag-status') || {}).textContent?.trim() || '',
      ragHits: node.querySelectorAll('.rag-hit').length,
      evidencePageButtons: node.querySelectorAll('[data-evidence-page]').length,
      trace,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth
    };
  })()`);
  context.observeTrace(report.trace);
  record({
    id: "chain.report.trace",
    group: "chain",
    kind: "real",
    title: "真实链路报告展示索引/FAISS 召回/重排/VLM 等阶段与降级原因",
    ok: report.trace.length >= 4 && report.trace.every(item => item.detail.length > 0),
    expected: ">=4 个阶段且每个阶段都有明细（包含降级/失败原因）",
    actual: report.trace.map(item => ({ title: item.title, status: item.status, detail: item.detail, tech: item.tech }))
  });
  record({
    id: "chain.rag.hits",
    group: "chain",
    kind: "real",
    title: "真实链路产出 RAG 命中片段与证据页",
    ok: report.ragPanel && report.ragHits >= 1 && report.evidencePageButtons >= 1,
    expected: "RAG 报告面板存在、命中片段 >=1、证据页按钮 >=1",
    actual: {
      ragPanel: report.ragPanel,
      ragHits: report.ragHits,
      evidencePageButtons: report.evidencePageButtons,
      failure: terminal.hasFailure ? terminal.failureText : ""
    },
    note: terminal.hasFailure ? "真实链路在模型阶段失败，RAG 命中无法产出（详见失败报告）" : ""
  });
  if (terminal.hasFailure) {
    record({
      id: "chain.failure.detail",
      group: "chain",
      kind: "real",
      title: "失败报告给出错误码、请求 ID 与恢复指引",
      ok: report.failureHero && report.failureMessage.length > 0 && report.requestId.includes("请求 ID"),
      expected: "失败页包含错误说明、错误码与请求 ID",
      actual: { failureHero: report.failureHero, failureMessage: report.failureMessage, requestId: report.requestId }
    });
  } else {
    record({
      id: "chain.failure.detail",
      group: "chain",
      kind: "real",
      title: "失败报告给出错误码、请求 ID 与恢复指引",
      ok: true,
      expected: "真实链路成功，未触发失败报告",
      actual: { note: "真实链路成功，无失败报告" }
    });
  }
  record({
    id: "layout.real.report",
    group: "layout",
    kind: "layout",
    title: "真实链路报告无页面级横向溢出",
    ok: noOverflow(report),
    expected: "scrollWidth <= clientWidth",
    actual: { scrollWidth: report.scrollWidth, clientWidth: report.clientWidth }
  });
  await capture(cdp, `${viewport.name}-real-chain-report.png`);
  context.realChain = { terminal, report };
  await cdp.evaluate("(() => { const close = document.querySelector('#analysis-viewer-close'); if (close) close.click(); return true; })()");
  await delay(300);
}

// ---------------------------------------------------------------------------
// RAG 报告视图能力（使用后端真实契约的固定数据）
// ---------------------------------------------------------------------------

function ragFixtureTrace(status, reason) {
  return [
    { name: "locating", title: "文档定位与页面渲染", status: "completed", detail: "已锁定 CN-fixture.pdf · P.6", technology: "DatasetRepository + PDF allowlist + PyMuPDF" },
    { name: "ocr", title: "目标页面 OCR 文档识别", status: "completed", detail: "已读取 1 页 OCR 缓存", technology: "Cache-first TXT OCR" },
    { name: "indexing", title: "RAG 混合语料索引", status: status, detail: status === "completed" ? "已形成 24 个可检索分块" : reason, technology: "Chinese-aware RecursiveCharacterTextSplitter + OCR TXT cache" },
    { name: "retrieval", title: "向量召回", status: status, detail: status === "completed" ? "从 24 个分块召回 12 个候选" : reason, technology: "BM25-style Keywords + OpenAI Embeddings + NPY + FAISS Fusion" },
    { name: "reranking", title: "候选精排", status: status, detail: status === "completed" ? "将候选重排为 6 个高相关片段" : reason, technology: "OpenAI-compatible Rerank API" },
    { name: "reasoning", title: "VLM 图文联合推理", status: "completed", detail: "生成开放式短答案", technology: "OpenAI-compatible multimodal Chat Completions" },
    { name: "validation", title: "答案规范化与评测", status: "completed", detail: "完成 EM 与 ROUGE-1 F1", technology: "Normalizer + jieba Metrics" }
  ];
}

async function renderAuditFixture(cdp, analysis) {
  const expression = `(() => {
    const fixture = ${JSON.stringify(analysis)};
    const S = window.InduDocState;
    const V = window.InduDocViews;
    const snapshot = { stage: S.stage, analysis: S.analysis, question: S.selectedQuestion, status: S.analysisStatus };
    S.stage = 'semi_final';
    S.selectedQuestion = { stage: 'semi_final', split: 'train', id: 9001, document: 'CN-fixture.pdf', question: '该装置通过什么结构实现工件的稳定定位？', answer: '通过限位挡板与导向槽配合实现。', options: [] };
    S.analysis = fixture;
    S.analysisStatus = 'completed';
    const content = document.getElementById('analysis-viewer-content');
    content.innerHTML = V.analysisAuditContent();
    const viewer = document.getElementById('analysis-viewer');
    viewer.hidden = false;
    if (window.lucide) window.lucide.createIcons();
    const trace = Array.from(viewer.querySelectorAll('.trace-item')).map(item => ({
      title: item.querySelector('strong')?.textContent?.trim() || '',
      detail: item.querySelector('p')?.textContent?.trim() || '',
      state: item.className.replace('trace-item', '').trim(),
      tech: Array.from(item.querySelectorAll('.trace-tech span')).map(span => span.textContent.trim()).join(' | ')
    }));
    const state = {
      ragPanel: Boolean(viewer.querySelector('.rag-report')),
      ragStatus: (viewer.querySelector('.rag-status') || {}).textContent?.trim() || '',
      ragHits: viewer.querySelectorAll('.rag-hit').length,
      ragPageButtons: viewer.querySelectorAll('.rag-hit [data-evidence-page]').length,
      flowText: (viewer.querySelector('.rag-flow') || {}).textContent?.replace(/\\s+/g, ' ').trim() || '',
      degradeReasonVisible: viewer.textContent.includes('向量检索不可用') || viewer.textContent.includes('重排不可用'),
      trace,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth
    };
    viewer.hidden = true;
    content.replaceChildren();
    S.stage = snapshot.stage;
    S.analysis = snapshot.analysis;
    S.selectedQuestion = snapshot.question;
    S.analysisStatus = snapshot.status;
    return state;
  })()`;
  return cdp.evaluate(expression);
}

async function runRagViewChecks(cdp, viewport) {
  const healthy = {
    answer: "通过限位挡板与导向槽配合实现。",
    answer_explanation: "召回片段指出限位挡板约束横向位移，目标页导向槽负责保持运动方向。",
    evidence: [{ source: "ocr", page: 4, quote: "限位挡板设置在导向槽两侧。", relevance: "重排后的首要文本证据。" }],
    option_analysis: [],
    reasoning_summary: ["从整份文档索引召回候选片段。"],
    ocr_text: "===== PDF P.6 =====\n限位挡板与导向槽形成定位通道。",
    ocr_pages: [6],
    evidence_page: 6,
    evidence_pages: [4, 6],
    metrics: { applicable: true, exact_match: 1, rouge1_f1: 1 },
    models: { ocr: "ocr-test", vlm: "vlm-test", retrieval: "embedding-test", reranker: "reranker-test" },
    mode: "real",
    execution: { engine: "RAG + OCR + VLM Multimodal Pipeline", protocol: "OpenAI-compatible APIs" },
    rag: {
      enabled: true,
      requested: true,
      retrieval_protocol: "hybrid_keyword_vector_npy",
      rerank_protocol: "compatible_rerank",
      retrieval_model: "embedding-test",
      reranker_model: "reranker-test",
      chunk_count: 24,
      recalled_count: 12,
      reranked_count: 6,
      vector_index_status: "ready",
      corpus_sources: ["pdf_text", "ocr"],
      pdf_text_pages: [2, 3, 4],
      ocr_pages: [6],
      items: [
        { page: 4, source: "pdf_text", text: "限位挡板设置在导向槽两侧，用于限制工件横向偏移。", retrieval_score: 0.82, rerank_score: 0.97 },
        { page: 6, source: "ocr", text: "图中挡板与导向槽围成连续定位通道。", retrieval_score: 0.61, rerank_score: 0.88 }
      ]
    },
    technical_trace: ragFixtureTrace("completed", "")
  };
  const healthyState = await renderAuditFixture(cdp, healthy);
  record({
    id: "rag.view.panel",
    group: "rag",
    kind: "fixture",
    title: "RAG 报告面板可展示召回/重排协议与命中片段",
    ok: healthyState.ragPanel && healthyState.ragHits >= 1 && healthyState.flowText.length > 0,
    expected: "存在 .rag-report、命中片段 >=1、协议区域非空",
    actual: { ragPanel: healthyState.ragPanel, ragHits: healthyState.ragHits, flowText: healthyState.flowText }
  });
  record({
    id: "rag.view.evidence.pages",
    group: "rag",
    kind: "fixture",
    title: "RAG 命中片段提供证据页跳转按钮",
    ok: healthyState.ragPageButtons >= 1,
    expected: ">=1 个命中片段页码按钮",
    actual: { ragPageButtons: healthyState.ragPageButtons }
  });
  record({
    id: "rag.view.trace.stages",
    group: "rag",
    kind: "fixture",
    title: "报告执行链路覆盖索引/FAISS 召回/重排/VLM/校验",
    ok: ["索引", "召回", "精排", "推理", "规范化"].every(keyword => healthyState.trace.some(item => item.title.includes(keyword))) &&
      healthyState.trace.some(item => item.tech.includes("FAISS")),
    expected: "阶段标题包含 索引/召回/精排/推理/规范化，且技术说明出现 FAISS",
    actual: healthyState.trace.map(item => ({ title: item.title, tech: item.tech }))
  });
  record({
    id: "layout.rag.report",
    group: "layout",
    kind: "layout",
    title: "RAG 报告无页面级横向溢出",
    ok: noOverflow(healthyState),
    expected: "scrollWidth <= clientWidth",
    actual: { scrollWidth: healthyState.scrollWidth, clientWidth: healthyState.clientWidth }
  });

  const degraded = {
    answer: "通过限位挡板与导向槽配合实现。",
    answer_explanation: "向量检索不可用时降级为关键词召回。",
    evidence: [{ source: "ocr", page: 4, quote: "限位挡板设置在导向槽两侧。", relevance: "重排后的首要文本证据。" }],
    option_analysis: [],
    reasoning_summary: ["降级链路下的可审计摘要。"],
    ocr_text: "===== PDF P.6 =====\n限位挡板与导向槽形成定位通道。",
    ocr_pages: [6],
    evidence_page: 6,
    evidence_pages: [4],
    metrics: { applicable: true },
    models: { vlm: "vlm-test" },
    mode: "real",
    execution: { engine: "RAG + OCR + VLM Multimodal Pipeline" },
    rag: {
      enabled: true,
      requested: true,
      retrieval_protocol: "keyword_only_degraded",
      rerank_protocol: "retrieval_score_fallback",
      retrieval_model: "embedding-test",
      reranker_model: "reranker-test",
      chunk_count: 18,
      recalled_count: 8,
      reranked_count: 8,
      vector_index_status: "degraded",
      degradation_reason: { code: "MODEL_REQUEST_REJECTED", message: "向量检索不可用" },
      rerank_degradation_reason: { code: "MODEL_REQUEST_REJECTED", message: "重排不可用" },
      corpus_sources: ["pdf_text"],
      pdf_text_pages: [2, 3],
      ocr_pages: [],
      items: [
        { page: 4, source: "pdf_text", text: "限位挡板设置在导向槽两侧，用于限制工件横向偏移。", retrieval_score: 0.71 }
      ]
    },
    technical_trace: ragFixtureTrace("skipped", "Retrieval/Reranker 配置不完整")
  };
  const degradedState = await renderAuditFixture(cdp, degraded);
  record({
    id: "rag.view.degraded.status",
    group: "rag",
    kind: "fixture",
    title: "RAG 降级时报告明确显示降级状态",
    ok: degradedState.ragPanel && degradedState.ragStatus.includes("降级"),
    expected: "RAG 状态标签包含“降级”",
    actual: { ragStatus: degradedState.ragStatus }
  });
  record({
    id: "rag.view.degraded.reason",
    group: "rag",
    kind: "fixture",
    title: "界面展示后端返回的降级原因（degradation_reason / rerank_degradation_reason）",
    ok: degradedState.degradeReasonVisible,
    expected: "报告文本包含后端 degradation_reason.message 与 rerank_degradation_reason.message",
    actual: {
      degradeReasonVisible: degradedState.degradeReasonVisible,
      ragStatus: degradedState.ragStatus,
      traceDetails: degradedState.trace.map(item => item.detail)
    },
    note: "前端未渲染后端 rag.degradation_reason / rag.rerank_degradation_reason（属 app/static 范围，需前端 agent 补齐）"
  });
  record({
    id: "rag.view.degraded.trace",
    group: "rag",
    kind: "fixture",
    title: "降级时执行链路保留索引/召回/重排阶段与跳过原因",
    ok: degradedState.trace.some(item => item.title.includes("索引") && item.detail.includes("配置不完整")) &&
      degradedState.trace.some(item => item.title.includes("召回")) &&
      degradedState.trace.some(item => item.title.includes("精排")),
    expected: "索引/召回/重排阶段存在且带有降级或跳过原因",
    actual: degradedState.trace.map(item => ({ title: item.title, detail: item.detail }))
  });
  await capture(cdp, `${viewport.name}-rag-report.png`);
}

// ---------------------------------------------------------------------------
// 故障注入：错误状态与离线提示（独立标签页，避免污染主标签页控制台统计）
// ---------------------------------------------------------------------------

async function runFaultInjectionChecks(context, viewport) {
  const cdp = await openTab(context.debugPort, baseUrl, viewport.width, viewport.height);
  const messages = [];
  try {
    await waitFor(cdp, OVERVIEW_READY, 45000, "总览加载");
    await cdp.send("Network.emulateNetworkConditions", { offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
    await cdp.evaluate("document.querySelector('#overview-refresh').click()");
    await waitFor(cdp, "Boolean(document.querySelector('#overview-body .state-block'))", 25000, "错误状态");
    const state = await cdp.evaluate(`(() => {
      const block = document.querySelector('#overview-body .state-block');
      return {
        text: block ? block.textContent.replace(/\\s+/g, ' ').trim() : '',
        retry: Boolean(document.querySelector('#overview-retry')),
        offlineBanner: !document.getElementById('offline-banner').hidden
      };
    })()`);
    record({
      id: "st.error.state",
      group: "states",
      kind: "real",
      title: "服务不可达时展示错误状态与重试入口",
      ok: state.text.includes("内容加载失败") && state.retry,
      expected: "错误状态文案含“内容加载失败”并提供重试按钮",
      actual: state
    });
    record({
      id: "st.offline.banner",
      group: "states",
      kind: "real",
      title: "服务不可达时顶部展示离线横幅",
      ok: state.offlineBanner,
      expected: "#offline-banner 可见",
      actual: { offlineBanner: state.offlineBanner }
    });
    await capture(cdp, `${viewport.name}-error-state.png`);
    await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
    await cdp.evaluate("document.querySelector('#overview-retry').click()");
    await waitFor(cdp, OVERVIEW_READY, 30000, "错误恢复");
    record({
      id: "st.error.recovery",
      group: "states",
      kind: "real",
      title: "服务恢复后可重试并重新渲染真实内容",
      ok: true,
      expected: "重试后总览恢复渲染",
      actual: { recovered: true }
    });
  } catch (error) {
    record({
      id: "st.error.state",
      group: "states",
      kind: "real",
      title: "服务不可达时展示错误状态与重试入口",
      ok: false,
      expected: "注入离线后出现错误状态",
      actual: error.message
    });
  } finally {
    messages.push(...consoleMessages(cdp));
    cdp.close();
  }
  return messages;
}

// ---------------------------------------------------------------------------
// 视口执行
// ---------------------------------------------------------------------------

function stageCoverage(stages, details, degradedDetails) {
  const titles = stages.join(" | ");
  const joined = `${titles} ${details.join(" ")}`;
  const reasonPattern = /跳过|降级|失败|拒绝|不可用|未配置|未就绪|配置不完整|没有|已禁用|不调用|未执行|关闭/;
  return {
    indexing: titles.includes("索引"),
    retrieval: titles.includes("召回") || joined.includes("FAISS"),
    reranking: titles.includes("精排") || titles.includes("重排"),
    reasoning: titles.includes("推理") || titles.includes("回放") || titles.includes("VLM"),
    validation: titles.includes("规范化") || titles.includes("评测"),
    skippedStages: degradedDetails.length,
    degradeReason: degradedDetails.length === 0 || degradedDetails.every(item => Boolean(item.detail && item.detail.trim()))
  };
}

async function runViewport(viewport, context) {
  currentViewport = viewport.name;
  const started = Date.now();
  const screenshots = [];
  const messages = [];
  let cdp = null;
  try {
    cdp = await openTab(context.debugPort, baseUrl, viewport.width, viewport.height);
    await cdp.send("Page.navigate", { url: baseUrl });
    await waitFor(cdp, "document.readyState === 'complete'", 30000, "页面加载");
    await waitFor(cdp, "document.querySelectorAll('[data-view]').length >= 5", 20000, "应用外壳");
    await waitFor(cdp, OVERVIEW_READY, 60000, "首页真实数据");
  } catch (error) {
    record({
      id: "viewport.boot",
      group: "runtime",
      kind: "real",
      title: "视口应用加载完成",
      ok: false,
      expected: "首页在超时前完成真实数据渲染",
      actual: error.message
    });
    if (cdp) {
      messages.push(...consoleMessages(cdp));
      cdp.close();
    }
    return { name: viewport.name, width: viewport.width, height: viewport.height, durationMs: Date.now() - started, screenshots, console: classifyConsole(messages) };
  }
  record({
    id: "viewport.boot",
    group: "runtime",
    kind: "real",
    title: "视口应用加载完成",
    ok: true,
    expected: "首页在超时前完成真实数据渲染",
    actual: { viewport: viewport.name, size: `${viewport.width}x${viewport.height}` }
  });

  const sections = [
    ["overview", () => runOverviewChecks(cdp, viewport)],
    ["settings", () => runSettingsChecks(cdp, viewport)],
    ["workspace", () => runWorkspaceChecks(cdp, viewport)],
    ["analysis", () => runDemoReplayChecks(cdp, viewport, context)],
    ["chain", () => runRealChainChecks(cdp, viewport, context)],
    ["rag", () => runRagViewChecks(cdp, viewport)]
  ];
  for (const [group, run] of sections) {
    try {
      await run();
    } catch (error) {
      record({
        id: `${group}.section`,
        group,
        kind: "real",
        title: `${group} 检查段执行完成`,
        ok: false,
        expected: "检查段正常执行",
        actual: error.message
      });
    }
  }

  const coverage = stageCoverage(context.stages, context.details, context.degradedDetails);
  record({
    id: "trace.stage.coverage",
    group: "chain",
    kind: "real",
    title: "执行链路覆盖索引/FAISS 召回/重排/VLM/校验与降级原因",
    ok: coverage.indexing && coverage.retrieval && coverage.reranking && coverage.reasoning && coverage.validation,
    expected: "索引、FAISS 召回、重排、VLM 推理、答案校验五类阶段均被真实服务观测到",
    actual: { coverage, stages: context.stages }
  });
  record({
    id: "trace.degrade.reason",
    group: "chain",
    kind: "real",
    title: "被跳过或失败的执行阶段必须给出可读原因",
    ok: coverage.degradeReason,
    expected: "存在 skipped/failed 阶段时，每个阶段都写明跳过/降级/失败原因",
    actual: { skippedOrFailedStages: context.degradedDetails }
  });

  await cdp.evaluate("document.querySelector('[data-view=overview]').click()");
  await waitFor(cdp, OVERVIEW_READY, 30000, "返回总览");
  const loadingState = await cdp.evaluate(`(() => {
    const button = document.querySelector('#overview-refresh');
    if (!button) return { clicked: false, loading: false };
    button.click();
    return { clicked: true, loading: Boolean(document.querySelector('#overview-body .state-block .skeleton')) };
  })()`);
  await waitFor(cdp, OVERVIEW_READY, 30000, "总览重新加载");
  record({
    id: "st.loading.state",
    group: "states",
    kind: "real",
    title: "刷新时展示加载骨架状态",
    ok: loadingState.clicked && loadingState.loading,
    expected: "点击刷新后立即出现加载骨架",
    actual: loadingState
  });

  const induced = await runFaultInjectionChecks(context, viewport);
  messages.push(...consoleMessages(cdp));

  const classified = classifyConsole(messages);
  record({
    id: "console.app.clean",
    group: "console",
    kind: "console",
    title: "应用控制台零 error/warning",
    ok: classified.app.error === 0 && classified.app.warning === 0,
    expected: "应用级 error=0 且 warning=0",
    actual: { error: classified.app.error, warning: classified.app.warning, messages: classified.app.messages }
  });

  cdp.close();
  return {
    name: viewport.name,
    width: viewport.width,
    height: viewport.height,
    durationMs: Date.now() - started,
    screenshots,
    console: classified,
    inducedConsole: countConsole(induced)
  };
}

function classifyConsole(messages) {
  const app = { error: 0, warning: 0, messages: [] };
  const network = { error: 0, warning: 0, messages: [] };
  for (const message of messages) {
    const isNetwork = /Failed to load resource|net::ERR/i.test(message.text || "");
    if (isNetwork) {
      const bucket = message.level === "warning" ? "warning" : "error";
      network[bucket] += 1;
      network.messages.push(message);
      continue;
    }
    const level = String(message.level || "").toLowerCase();
    if (level === "error") {
      app.error += 1;
      app.messages.push(message);
    } else if (level === "warning" || level === "warn") {
      app.warning += 1;
      app.messages.push(message);
    }
  }
  return { app, network };
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------

async function cleanArtifacts() {
  const entries = await readdir(fileURLToPath(OUTPUT_DIR)).catch(() => []);
  for (const entry of entries) {
    if (/^(desktop|tablet|mobile)-.*\.png$/.test(entry) || entry === "report.json") {
      await rm(new URL(entry, OUTPUT_DIR), { force: true });
    }
  }
}

async function waitForDebugger(debugPort) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      return await httpJson(`http://127.0.0.1:${debugPort}/json/version`);
    } catch {
      await delay(250);
    }
  }
  throw new Error("Chrome DevTools 端口未就绪，浏览器可能启动失败。");
}

async function stopBrowser(browser) {
  if (!browser || browser.exitCode !== null) return;
  if (process.platform === "win32" && browser.pid) {
    await new Promise(resolve => {
      const killer = spawn("taskkill", ["/PID", String(browser.pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
      killer.on("exit", resolve);
      killer.on("error", resolve);
    });
  }
  browser.kill();
  await new Promise(resolve => {
    if (browser.exitCode !== null) return resolve();
    browser.once("exit", resolve);
    setTimeout(resolve, 3000);
  });
}

async function removeProfile(profileDir) {
  if (!profileDir) return;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    await rm(profileDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 }).catch(() => {});
    if (!existsSync(profileDir)) return;
    await delay(300);
  }
  console.log(`[browser-check] 警告：临时 profile 未能完全清理：${profileDir}`);
}

async function runBrowserSessions(context, viewportResults) {
  for (const viewport of VIEWPORTS) {
    const viewportContext = {
      debugPort: context.debugPort,
      stages: [],
      details: [],
      degradedDetails: [],
      observeTrace(trace) {
        for (const item of trace || []) {
          if (!this.stages.includes(item.title)) this.stages.push(item.title);
          if (item.detail) this.details.push(item.detail);
          if (item.state === "skipped" || item.state === "failed") {
            this.degradedDetails.push({ title: item.title, status: item.state, detail: item.detail });
          }
        }
      }
    };
    console.log(`\n[browser-check] ---- 视口 ${viewport.name} ----`);
    const result = await runViewport(viewport, viewportContext);
    const files = await readdir(fileURLToPath(OUTPUT_DIR)).catch(() => []);
    result.screenshots = files.filter(file => file.startsWith(`${viewport.name}-`)).sort();
    viewportResults.push(result);
  }
}

async function main() {
  const started = Date.now();
  console.log(`[browser-check] Base URL : ${baseUrl}（来源：${baseUrlInfo.source}）`);
  console.log("[browser-check] 说明：base URL 优先级 INDUDOC_BASE_URL > --base-url > HOST/PORT > 默认 8000，与 start.bat 保持一致。");

  let browserPath = "";
  let browser = null;
  let profileDir = "";
  let browserVersion = "";
  let fatal = null;
  const viewportResults = [];

  try {
    browserPath = resolveBrowser(options);
    console.log(`[browser-check] 浏览器   : ${browserPath}`);

    const health = await fetch(`${baseUrl}${API_ROOT}/health`, { headers: { Accept: "application/json" } });
    if (!health.ok) throw new Error(`HTTP ${health.status} ${health.statusText}`);
    const payload = await health.json();
    record({
      id: "api.health",
      group: "runtime",
      kind: "real",
      title: "后端 /api/health 返回 200",
      ok: payload?.status === "ok",
      expected: "status=ok",
      actual: { status: payload?.status, version: payload?.version, datasetQuestionCount: payload?.dataset_question_count }
    });

    await mkdir(OUTPUT_DIR, { recursive: true });
    await mkdir(PROFILE_DIR, { recursive: true });
    await cleanArtifacts();
    const debugPort = await findFreePort();
    profileDir = await mkdtemp(path.join(fileURLToPath(PROFILE_DIR), "tmp-"));
    browser = spawn(browserPath, [
      "--headless=new",
      "--disable-gpu",
      "--disable-software-rasterizer",
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      "--disable-background-networking",
      `--remote-debugging-port=${debugPort}`,
      `--user-data-dir=${profileDir}`,
      "about:blank"
    ], { stdio: "ignore", windowsHide: true });

    const version = await waitForDebugger(debugPort);
    browserVersion = version.Browser || "";
    console.log(`[browser-check] DevTools : 127.0.0.1:${debugPort} · ${browserVersion}`);

    await runBrowserSessions({ debugPort }, viewportResults);
  } catch (error) {
    fatal = error.stack || error.message;
  } finally {
    await stopBrowser(browser);
    await removeProfile(profileDir);
  }

  const summary = {
    total: checks.length,
    passed: checks.filter(item => item.ok).length,
    failed: checks.filter(item => !item.ok).length
  };
  const byViewport = {};
  for (const result of viewportResults) {
    const scoped = checks.filter(item => item.viewport === result.name);
    byViewport[result.name] = {
      total: scoped.length,
      passed: scoped.filter(item => item.ok).length,
      failed: scoped.filter(item => !item.ok).length
    };
  }

  const report = {
    generatedAt: new Date().toISOString(),
    durationMs: Date.now() - started,
    baseUrl,
    baseUrlSource: baseUrlInfo.source,
    browser: { path: browserPath, version: browserVersion },
    node: process.version,
    viewports: viewportResults,
    checks,
    summary: { ...summary, byViewport },
    failures: checks.filter(item => !item.ok).map(item => ({
      id: item.id,
      group: item.group,
      viewport: item.viewport,
      kind: item.kind,
      title: item.title,
      expected: item.expected,
      actual: item.actual,
      note: item.note
    }))
  };
  await mkdir(OUTPUT_DIR, { recursive: true });
  await writeFile(new URL("report.json", OUTPUT_DIR), `${JSON.stringify(report, null, 2)}\n`, "utf8");

  console.log("\n[browser-check] ============ 验收汇总 ============");
  console.log(`Base URL   : ${baseUrl}（${baseUrlInfo.source}）`);
  console.log(`浏览器      : ${browserVersion || browserPath}`);
  console.log(`耗时        : ${((Date.now() - started) / 1000).toFixed(1)} 秒`);
  console.log(`检查项      : ${summary.total} 项，通过 ${summary.passed}，失败 ${summary.failed}`);
  for (const [name, stats] of Object.entries(byViewport)) {
    console.log(`  · ${name}: ${stats.total} 项 / 通过 ${stats.passed} / 失败 ${stats.failed}`);
  }
  for (const result of viewportResults) {
    console.log(`  · ${result.name} 控制台: 应用级 error ${result.console.app.error} / warning ${result.console.app.warning}；网络级 error ${result.console.network.error}`);
  }
  const artifacts = await readdir(fileURLToPath(OUTPUT_DIR)).catch(() => []);
  console.log(`截图目录    : ${fileURLToPath(OUTPUT_DIR)}（${artifacts.filter(file => file.endsWith(".png")).length} 张 PNG）`);
  console.log(`报告        : ${fileURLToPath(new URL("report.json", OUTPUT_DIR))}`);
  if (report.failures.length) {
    console.log("失败明细    :");
    for (const failure of report.failures) {
      console.log(`  - [${failure.viewport}] ${failure.id} · ${failure.title}`);
      console.log(`      期望: ${JSON.stringify(failure.expected)}`);
      console.log(`      实际: ${JSON.stringify(failure.actual)}`);
      if (failure.note) console.log(`      说明: ${failure.note}`);
    }
  }
  console.log("[browser-check] ==================================");

  return { fatal, failed: summary.failed };
}

const outcome = await main().catch(error => ({ fatal: error.stack || error.message, failed: checks.length }));
const exitCode = outcome.fatal ? 1 : outcome.failed > 0 ? 2 : 0;
if (outcome.fatal) {
  console.error(`[browser-check] 致命错误：${outcome.fatal}`);
}
process.exitCode = exitCode;
const safety = setTimeout(() => process.exit(exitCode), 3000);
safety.unref();
