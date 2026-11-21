import { demoMessages, demoSearch, mailboxes as demoMailboxes } from "./demo.js";
import { assembleAnswer, highlightIds, insufficient, planQuestion } from "./assistant.js";

const byId = id => document.getElementById(id);
const queryInput = byId("query");
const messageList = byId("messages");
const letter = byId("letter");
const mailboxSelect = byId("mailbox");
const folderNames = { inbox: "Входящие", starred: "Избранное", archive: "Архив", reminders: "Напоминания" };
const fullDate = new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
const shortDate = new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "short", timeZone: "UTC" });
const reminderDate = new Intl.DateTimeFormat("ru-RU", { dateStyle: "medium", timeStyle: "short" });
const when = (date, formatter) => date ? formatter.format(new Date(date + "T12:00:00Z")) : "";
const PAGE_SIZE = 60;
const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
};
const icon = name => {
  const node = make("span", "icon", name);
  node.setAttribute("aria-hidden", "true");
  return node;
};

const state = {
  live: false, mailboxes: demoMailboxes, mailbox: demoMailboxes[0], messages: [],
  folder: "inbox", mode: "semantic", query: "", results: null, date: null, selected: null,
  version: 0, pending: false, error: "", loadingMailbox: false, limit: PAGE_SIZE,
  mailboxVersion: 0, details: new Map(), view: "search", applied: false,
  filters: {}, appliedFilters: {}, orderIds: [], evidence: [], reminders: null,
  reminderMail: null, fromCitation: false,
};

function localFlags() {
  try {
    const parsed = JSON.parse(localStorage.getItem("pismo.messageFlags.v1"));
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}
const flags = localFlags();
const flagKey = mail => (state.live ? "corpus:" : "demo:") + mail.mailbox + ":" + mail.id;
const flag = (mail, name) => flags[flagKey(mail)]?.[name] ?? Boolean(mail[name]);
function setFlag(mail, name, value) {
  const key = flagKey(mail);
  flags[key] = { ...flags[key], [name]: value };
  try {
    localStorage.setItem("pismo.messageFlags.v1", JSON.stringify(flags));
  } catch {
    // Mail actions remain available in this tab.
  }
}

function initials(sender) {
  return sender.split(/\s+/u).slice(0, 2).map(word => word[0]).join("").toUpperCase();
}
function avatar(mail) {
  const node = make("span", "avatar", initials(mail.sender || mail.category || "Письмо"));
  node.dataset.tone = mail.tone ?? 0;
  node.setAttribute("aria-hidden", "true");
  return node;
}
function messageCount(n) {
  return n + " " + { one: "письмо", few: "письма", many: "писем", other: "писем" }[new Intl.PluralRules("ru").select(n)];
}
function visibleMessages() {
  if (state.applied) return (state.results || []).map(id => state.messages.find(mail => mail.id === id)).filter(Boolean);
  if (state.folder === "reminders") return (state.reminders || []).map(entry => entry.message);
  return state.messages.filter(mail => {
    if (state.folder === "starred") return flag(mail, "starred");
    if (state.folder === "archive") return flag(mail, "archived");
    return !flag(mail, "archived");
  });
}
function highlighted(node, text) {
  for (const [index, part] of highlightIds(text, state.orderIds).entries()) {
    node.append(index % 2 ? make("mark", "", part) : document.createTextNode(part));
  }
  return node;
}

function reminderFor(mail) {
  return state.reminders?.find(item => item.message.id === mail.id);
}

function reminderText(item) {
  if (!item) return "";
  const label = Date.parse(item.dueAt) <= Date.now() ? "Срок наступил" : "Напомнить";
  return label + " · " + reminderDate.format(new Date(item.dueAt));
}
function blank(symbol, title, description) {
  const container = make("div", "empty");
  container.append(icon(symbol), make("strong", "", title), make("p", "", description));
  return container;
}

function renderReader(mail, index, count) {
  const tools = byId("letter-tools");
  tools.hidden = !mail;
  byId("position").textContent = mail && state.applied ? "Результат " + (index + 1) + " из " + count : "Письмо";
  if (!mail) {
    letter.replaceChildren(blank("mail", "Выберите письмо", "Письмо откроется здесь."));
    return;
  }
  const subject = highlighted(make("h2"), mail.subject || "Без темы");
  const header = make("div", "envelope");
  const sender = make("div", "identity");
  sender.append(make("span", "name", mail.sender || "Отправитель не указан"));
  if (mail.address) sender.append(make("small", "", mail.address));
  const time = make("time", "", [when(mail.date, fullDate), mail.time].filter(Boolean).join(" · "));
  if (mail.date) time.dateTime = mail.date;
  header.append(avatar(mail), sender, time);
  const body = make("div", "body-text");
  const detail = state.details.get(mail.id);
  const paragraphs = (detail?.body ?? mail.body ?? "").split(/\n\s*\n/u).filter(Boolean);
  body.append(...paragraphs.map(text => highlighted(make("p"), text)));
  if (state.live && !detail?.body && detail?.failed) {
    const retry = make("button", "more", "Загрузить письмо целиком");
    retry.addEventListener("click", () => {
      state.details.delete(mail.id);
      loadLetter(mail);
    });
    body.append(retry);
  }
  const elements = [subject, header, body];
  if (mail.category) {
    const category = make("span", "category");
    category.textContent = mail.category;
    elements.unshift(category);
  }
  if (mail.details) {
    const details = make("dl", "details");
    for (const [key, value] of Object.entries(mail.details)) {
      const item = make("div");
      item.append(make("dt", "", key), make("dd", "", value));
      details.append(item);
    }
    elements.push(details);
  }
  const reminder = reminderFor(mail);
  if (reminder) elements.push(make("p", "reminder-status", reminderText(reminder)));
  letter.replaceChildren(...elements);
  const starred = flag(mail, "starred");
  const archived = flag(mail, "archived");
  byId("star").setAttribute("aria-pressed", String(starred));
  setTool("star", starred ? "Убрать из избранного" : "В избранное");
  setTool("archive", archived ? "Вернуть во входящие" : "В архив", archived ? "unarchive" : "archive");
  setTool("read-toggle", flag(mail, "unread") ? "Отметить прочитанным" : "Отметить непрочитанным");
  byId("remind").hidden = state.reminders === null;
  setTool("remind", reminder ? "Изменить напоминание" : "Напомнить о письме");
}

function setTool(id, label, glyph) {
  const button = byId(id);
  button.title = label;
  button.setAttribute("aria-label", label);
  if (glyph) button.firstElementChild.textContent = glyph;
}

function renderAnswer() {
  if (state.view !== "ask") return;
  const answer = byId("answer");
  const scroll = answer.scrollTop;
  const nodes = [];
  if (state.query) nodes.push(make("h3", "question-title", state.query));
  if (state.pending) nodes.push(make("p", "answer-label", "Ищем письма для ответа…"));
  else if (state.error) nodes.push(make("p", "", "Не удалось получить ответ. Попробуйте ещё раз."));
  else if (!state.applied) nodes.push(blank("auto_awesome", "Спросите о письмах", "Например, когда согласовали доставку или что ответили покупателю."));
  else if (!state.evidence.length) nodes.push(make("p", "", insufficient));
  else {
    nodes.push(make("p", "answer-label", "Выдержки из писем"));
    const citations = make("ol", "evidence");
    for (const [index, item] of state.evidence.entries()) {
      const mail = state.messages.find(candidate => candidate.id === item.id);
      if (!mail) continue;
      const row = make("li");
      const cite = make("button", "citation", [index + 1, mail.subject || "Без темы", when(mail.date, shortDate)].filter(Boolean).join(" · "));
      cite.type = "button";
      cite.dataset.cite = mail.id;
      if (mail.id === state.selected && document.body.classList.contains("source-open")) cite.setAttribute("aria-current", "true");
      cite.addEventListener("click", () => openMessage(mail, true));
      row.append(highlighted(make("blockquote"), item.text), cite);
      citations.append(row);
    }
    nodes.push(citations);
    if (state.evidence.length === 1 && state.reminders !== null) {
      const mail = state.messages.find(candidate => candidate.id === state.evidence[0].id);
      if (mail) {
        const button = make("button", "secondary remind-from-answer", reminderFor(mail) ? "Изменить напоминание" : "Напомнить о письме");
        button.type = "button";
        button.addEventListener("click", () => openReminder(mail));
        nodes.push(button);
      }
    }
  }
  answer.replaceChildren(...nodes);
  answer.scrollTop = scroll;
}

function renderFilters() {
  const area = byId("active-filters");
  const { sender, from, to } = state.appliedFilters;
  const labels = [sender && "От " + sender,
    from && to ? when(from, shortDate) + " – " + when(to, shortDate) :
      from ? "С " + when(from, shortDate) : to ? "По " + when(to, shortDate) : ""].filter(Boolean);
  area.replaceChildren(...labels.map(label => {
    const button = make("button", "", label);
    button.type = "button";
    button.addEventListener("click", openFilters);
    return button;
  }));
  area.hidden = !labels.length;
}

function render() {
  const rows = visibleMessages();
  if (!rows.some(mail => mail.id === state.selected)) state.selected = rows[0]?.id ?? null;
  if (rows.findIndex(mail => mail.id === state.selected) >= state.limit) state.selected = rows[0]?.id ?? null;
  const scroll = messageList.scrollTop;
  const focusId = document.activeElement?.dataset?.message;
  const rendered = rows.slice(0, state.limit).map(mail => {
    const selected = mail.id === state.selected;
    const unread = flag(mail, "unread");
    const button = make("button", "mail" + (unread ? " unread" : ""));
    button.type = "button";
    button.dataset.message = mail.id;
    button.setAttribute("aria-current", String(selected));
    const sender = make("span", "sender");
    if (unread) {
      const mark = make("span", "unread-mark");
      mark.title = "Непрочитанное";
      sender.append(mark);
    }
    sender.append(make("span", "clip", mail.sender || mail.category || "Письмо"));
    const time = make("time", "", mail.date === state.mailbox.asOf && mail.time ? mail.time : when(mail.date, shortDate));
    if (mail.date) time.dateTime = mail.date;
    const reminder = state.folder === "reminders" ? reminderFor(mail) : null;
    const snippet = reminder ? reminderText(reminder) : (mail.body || "").replace(/\s+/gu, " ");
    button.append(avatar(mail), sender, time, highlighted(make("span", "subject"), mail.subject || "Без темы"),
      highlighted(make("span", "snippet" + (reminder ? " reminder-row" : "")), snippet));
    return button;
  });
  if (rows.length > state.limit) {
    const more = make("button", "more", "Показать ещё");
    more.type = "button";
    more.addEventListener("click", () => {
      const previous = state.limit;
      state.limit += PAGE_SIZE;
      render();
      messageList.children[previous]?.focus();
    });
    rendered.push(more);
  }
  if (!rendered.length) {
    if (state.error) rendered.push(blank("info", "Не удалось загрузить письма", "Попробуйте повторить поиск или переключить ящик."));
    else if (state.pending) rendered.push(blank("search", "Ищем письма", ""));
    else if (state.loadingMailbox) rendered.push(blank("inbox", "Загружаем письма", ""));
    else if (state.applied) rendered.push(blank("search", "Письма не найдены", "Попробуйте другие слова или измените фильтры."));
    else if (state.folder === "reminders") rendered.push(blank("inbox", "Напоминаний пока нет", "Откройте письмо и нажмите «Напомнить о письме»."));
    else rendered.push(blank("inbox", "В папке пока пусто", ""));
  }
  messageList.replaceChildren(...rendered);
  messageList.scrollTop = scroll;
  if (focusId) [...messageList.children].find(row => row.dataset.message === focusId)?.focus({ preventScroll: true });
  byId("list-title").textContent = state.applied ? "Результаты поиска" : folderNames[state.folder];
  byId("list-count").textContent = state.pending || state.loadingMailbox ? "Загрузка…" : state.error ? "Ошибка загрузки" : messageCount(rows.length);
  byId("breadcrumb").textContent = state.view === "ask" ? "Спросить" : state.applied ? "Поиск" : folderNames[state.folder];
  byId("unread-count").textContent = state.messages.filter(mail => flag(mail, "unread") && !flag(mail, "archived")).length || "";
  byId("date-filter").hidden = !state.date;
  byId("date-filter").lastElementChild.textContent = when(state.date, shortDate);
  byId("clear").hidden = !queryInput.value && !state.applied;
  byId("show-answer").hidden = state.view !== "ask" || !state.applied;
  document.body.classList.toggle("asking", state.view === "ask");
  document.querySelector('[data-folder="reminders"]').hidden = state.reminders === null;
  updateReminderCount();
  for (const button of document.querySelectorAll("[data-folder]")) {
    const active = state.view === "search" && !state.applied && button.dataset.folder === state.folder;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  for (const button of document.querySelectorAll("[data-mode]")) {
    button.setAttribute("aria-pressed", String(state.mode === button.dataset.mode));
  }
  const index = rows.findIndex(mail => mail.id === state.selected);
  renderReader(rows[index], index, rows.length);
  if (state.live && index >= 0 && (state.view === "search" || document.body.classList.contains("source-open"))) loadLetter(rows[index]);
  renderAnswer();
  renderFilters();
}

async function api(path, options, timeout = 15_000) {
  const response = await fetch("./api/" + path, { ...options, signal: AbortSignal.timeout(timeout) });
  if (!response.ok || !response.headers.get("content-type")?.includes("application/json")) throw new Error("API unavailable");
  return response.json();
}
function jsonRequest(method, data) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) };
}

function updateReminderCount() {
  byId("reminder-count").textContent = state.reminders?.filter(item => Date.parse(item.dueAt) <= Date.now()).length || "";
}

async function refreshReminders() {
  const mailbox = state.mailbox.id;
  try {
    const data = await api("reminders?" + new URLSearchParams({ mailbox }), undefined, 3500);
    if (state.mailbox.id !== mailbox || !Array.isArray(data.reminders)) return;
    state.reminders = data.reminders;
    render();
  } catch {
    // A plain static preview can run the search without a reminder store.
  }
}

function localDateTime(date) {
  const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return shifted.toISOString().slice(0, 16);
}

function openReminder(mail) {
  const item = reminderFor(mail);
  state.reminderMail = mail;
  byId("reminder-subject").textContent = mail.subject || "Без темы";
  byId("reminder-time").value = localDateTime(item ? new Date(item.dueAt) : new Date(Date.now() + 60 * 60_000));
  byId("reminder-zone").textContent = "Часовой пояс " + Intl.DateTimeFormat().resolvedOptions().timeZone;
  byId("remove-reminder").hidden = !item;
  byId("reminder-error").hidden = true;
  byId("reminder-dialog").showModal();
}

let noticeTimer;
function notify(text) {
  byId("live").textContent = text;
  const notice = byId("notice");
  notice.hidden = false;
  notice.textContent = text;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice.hidden = true; }, 3500);
}

async function saveReminder(remove = false) {
  const mail = state.reminderMail;
  const time = new Date(byId("reminder-time").value);
  const error = byId("reminder-error");
  if (!remove && (!Number.isFinite(time.getTime()) || time.getTime() <= Date.now())) {
    error.textContent = "Выберите время в будущем.";
    error.hidden = false;
    return;
  }
  const controls = [...byId("reminder-form").querySelectorAll("button")];
  controls.forEach(button => { button.disabled = true; });
  const data = { mailbox: mail.mailbox, messageId: mail.id };
  if (!remove) data.dueAt = time.toISOString();
  try {
    const result = await api("reminders", jsonRequest(remove ? "DELETE" : "PUT", data));
    if (state.mailbox.id === mail.mailbox) state.reminders = result.reminders;
    byId("reminder-dialog").close();
    render();
    notify(remove ? "Напоминание удалено" : "Напоминание сохранено");
  } catch {
    error.textContent = "Не удалось сохранить изменение. Попробуйте ещё раз.";
    error.hidden = false;
  } finally {
    controls.forEach(button => { button.disabled = false; });
  }
}

async function loadLetter(mail) {
  if (state.details.has(mail.id)) return;
  const details = state.details;
  details.set(mail.id, {});
  try {
    const result = await api("message?" + new URLSearchParams({ id: mail.id, mailbox: mail.mailbox }));
    details.set(mail.id, result.message);
  } catch {
    details.set(mail.id, { failed: true });
  }
  if (state.details === details && state.selected === mail.id) {
    const rows = visibleMessages();
    renderReader(mail, rows.findIndex(row => row.id === mail.id), rows.length);
  }
}
function resetSearch() {
  state.version++;
  state.query = "";
  state.applied = false;
  state.filters = {};
  state.appliedFilters = {};
  state.orderIds = [];
  state.evidence = [];
  state.results = null;
  state.date = null;
  state.pending = false;
  state.error = "";
  state.limit = PAGE_SIZE;
  state.selected = null;
  queryInput.value = "";
  messageList.scrollTop = 0;
  closeReader();
}
function closeReader() {
  document.body.classList.remove("reader-open", "source-open", "answer-open");
}
function openMessage(mail, fromCitation = false) {
  if (state.selected !== mail.id) letter.scrollTop = 0;
  state.selected = mail.id;
  state.fromCitation = fromCitation;
  setFlag(mail, "unread", false);
  document.body.classList.add(state.view === "ask" ? "source-open" : "reader-open");
  document.body.classList.remove("answer-open");
  render();
  if (state.view === "ask" || matchMedia("(max-width: 820px)").matches) byId("back").focus();
}
function setMenu(open) {
  byId("sidebar").classList.toggle("open", open);
  byId("menu").setAttribute("aria-expanded", String(open));
}
function showExamples() {
  const suggestions = byId("suggestions");
  let examples = state.snapshot === state.mailbox.asOf ? state.mailbox.examples || [] : [];
  if (!state.live && state.view === "ask") examples = state.mailbox.id === "anna"
    ? ["Когда обещали ответить по возврату заказа № 78431?", "Что случилось с оплатой заказа № 78452?"]
    : ["Во сколько разгрузили контейнер MSKU 210478?", "Когда приедет машина на склад?"];
  suggestions.hidden = !examples.length;
  suggestions.replaceChildren(make("span", "", "Попробуйте"), ...examples.map(text => {
    const button = make("button", "", text);
    button.type = "button";
    button.addEventListener("click", () => {
      queryInput.value = text;
      search();
    });
    return button;
  }));
}
async function selectMailbox(id, asOf) {
  resetSearch();
  setMenu(false);
  state.folder = "inbox";
  state.reminders = null;
  state.mailbox = state.mailboxes.find(mailbox => mailbox.id === id);
  state.snapshot = asOf || state.mailbox.asOf;
  byId("account-mark").textContent = state.mailbox.initials;
  byId("today").textContent = when(state.snapshot, fullDate);
  byId("snapshot-date").value = state.snapshot;
  showExamples();
  refreshReminders();
  if (!state.live) {
    state.messages = demoMessages.filter(mail => mail.mailbox === id);
    render();
    return;
  }
  state.messages = [];
  state.details = new Map();
  state.loadingMailbox = true;
  const version = ++state.mailboxVersion;
  render();
  try {
    const payload = await api("messages?" + new URLSearchParams({ mailbox: id, asOf: state.snapshot }));
    if (state.mailboxVersion !== version) return;
    state.messages = payload.messages;
  } catch {
    if (state.mailboxVersion !== version) return;
    if (!state.applied) state.error = "messages";
  }
  state.loadingMailbox = false;
  render();
}
async function search() {
  const query = queryInput.value.trim();
  if (!query && !Object.values(state.filters).some(Boolean)) {
    resetSearch();
    render();
    return;
  }
  const version = ++state.version;
  const plan = planQuestion(query, state.snapshot, [...new Set(state.messages.map(mail => mail.sender).filter(Boolean))], state.filters);
  const filters = state.view === "ask" ? plan.filters : state.filters;
  state.query = query;
  state.applied = true;
  state.appliedFilters = filters;
  state.orderIds = plan.ids;
  state.evidence = [];
  state.selected = null;
  state.results = null;
  state.date = null;
  state.pending = true;
  state.error = "";
  closeReader();
  if (state.view === "ask" && matchMedia("(max-width: 820px)").matches) document.body.classList.add("answer-open");
  messageList.scrollTop = 0;
  render();
  try {
    const searchText = state.view === "ask" ? plan.query : query;
    const request = { mailbox: state.mailbox.id, query: searchText, mode: state.mode, asOf: state.snapshot,
      filters, orderIds: plan.ids };
    const demoCandidates = state.live || !plan.ids.length ? state.messages : state.messages.filter(mail => {
      const text = mail.subject + " " + mail.body;
      return plan.ids.some(id => highlightIds(text, [id]).length > 1);
    });
    const result = state.live ? await api("search", jsonRequest("POST", request), 120_000)
      : demoSearch(demoCandidates, searchText, state.snapshot, state.mode, filters);
    if (state.version !== version) return;
    if (result.messages) {
      const emails = new Map(state.messages.map(mail => [mail.id, mail]));
      for (const mail of result.messages) emails.set(mail.id, mail);
      state.messages = [...emails.values()];
    }
    state.results = result.ids;
    state.date = result.date;
    if (state.view === "ask") {
      const top = visibleMessages().slice(0, 6);
      const sources = state.live ? await Promise.all(top.map(mail => api("message?" + new URLSearchParams({
        mailbox: mail.mailbox, id: mail.id,
      })).then(payload => payload.message).catch(() => null))) : top;
      if (state.version !== version) return;
      state.evidence = assembleAnswer(query, sources.filter(Boolean));
    }
  } catch {
    if (state.version !== version) return;
    state.error = "search";
  }
  state.pending = false;
  render();
}

const filtersForm = byId("filters-form");
let draftMode;
function openFilters() {
  const { sender = "", from = "", to = "" } = state.applied ? state.appliedFilters : state.filters;
  filtersForm.elements.namedItem("sender").value = sender;
  filtersForm.elements.namedItem("from").value = from;
  filtersForm.elements.namedItem("to").value = to;
  filtersForm.elements.namedItem("to").setCustomValidity("");
  draftMode = state.mode;
  byId("filters-dialog").showModal();
}

byId("filter-toggle").addEventListener("click", openFilters);
byId("reset-filters").addEventListener("click", () => {
  filtersForm.reset();
  filtersForm.elements.namedItem("to").setCustomValidity("");
});
filtersForm.addEventListener("input", () => filtersForm.elements.namedItem("to").setCustomValidity(""));
filtersForm.addEventListener("submit", event => {
  event.preventDefault();
  const filters = Object.fromEntries(new FormData(filtersForm));
  const to = filtersForm.elements.namedItem("to");
  if (filters.from && filters.to && filters.to < filters.from) {
    to.setCustomValidity("Конец периода должен быть не раньше начала.");
    to.reportValidity();
    return;
  }
  filters.sender = filters.sender.trim();
  state.filters = filters;
  state.mode = draftMode;
  byId("filters-dialog").close();
  search();
});
for (const button of document.querySelectorAll("[data-close]")) {
  button.addEventListener("click", () => byId(button.dataset.close).close());
}
for (const button of document.querySelectorAll("[data-preset]")) {
  button.addEventListener("click", () => {
    const preset = button.dataset.preset;
    const target = new Date(Date.now() + (preset === "tomorrow" ? 0 : Number(preset) * 60_000));
    if (preset === "tomorrow") {
      target.setDate(target.getDate() + 1);
      target.setHours(9, 0, 0, 0);
    }
    byId("reminder-time").value = localDateTime(target);
  });
}
byId("reminder-form").addEventListener("submit", event => {
  event.preventDefault();
  saveReminder();
});
byId("remove-reminder").addEventListener("click", () => saveReminder(true));
byId("remind").addEventListener("click", () => {
  const mail = visibleMessages().find(candidate => candidate.id === state.selected);
  if (mail) openReminder(mail);
});
setInterval(() => {
  if (state.reminders === null) return;
  const previous = byId("reminder-count").textContent;
  updateReminderCount();
  if (previous !== byId("reminder-count").textContent && state.folder === "reminders") render();
}, 15_000);

function setView(view) {
  state.view = view;
  for (const button of document.querySelectorAll("[data-view]")) {
    button.setAttribute("aria-pressed", String(button.dataset.view === view));
  }
  byId("submit-query").textContent = view === "ask" ? "Спросить" : "Найти";
  queryInput.placeholder = view === "ask" ? "Что нужно узнать из писем?" : "Опишите письмо или введите номер заказа";
  showExamples();
}
for (const button of document.querySelectorAll("[data-view]")) {
  button.addEventListener("click", () => {
    if (button.dataset.view === state.view) return;
    setView(button.dataset.view);
    closeReader();
    if (state.applied) search();
    else render();
  });
}
byId("show-answer").addEventListener("click", () => {
  document.body.classList.add("answer-open");
  byId("answer-back").focus();
});
byId("answer-back").addEventListener("click", () => {
  document.body.classList.remove("answer-open");
  byId("show-answer").focus();
});
byId("search-form").addEventListener("submit", event => {
  event.preventDefault();
  search();
});
queryInput.addEventListener("input", () => {
  byId("clear").hidden = !queryInput.value && !state.applied;
  if (!queryInput.value && state.applied && !Object.values(state.filters).some(Boolean)) {
    resetSearch();
    render();
  }
});
byId("clear").addEventListener("click", () => { resetSearch(); render(); queryInput.focus(); });
mailboxSelect.addEventListener("change", () => selectMailbox(mailboxSelect.value));
byId("snapshot-date").addEventListener("change", event => {
  const input = event.currentTarget;
  if (input.value && input.validity.valid) selectMailbox(mailboxSelect.value, input.value);
  else input.value = state.snapshot;
});
for (const button of document.querySelectorAll("[data-folder]")) {
  button.addEventListener("click", () => {
    resetSearch();
    state.folder = button.dataset.folder;
    setView("search");
    setMenu(false);
    render();
  });
}
for (const button of document.querySelectorAll("[data-mode]")) {
  button.addEventListener("click", () => {
    draftMode = button.dataset.mode;
    for (const choice of document.querySelectorAll("[data-mode]")) {
      choice.setAttribute("aria-pressed", String(choice.dataset.mode === draftMode));
    }
  });
}
messageList.addEventListener("click", event => {
  const row = event.target.closest("[data-message]");
  if (!row) return;
  const mail = visibleMessages().find(candidate => candidate.id === row.dataset.message);
  if (mail) openMessage(mail);
});
byId("back").addEventListener("click", () => {
  closeReader();
  if (state.view === "ask" && state.fromCitation) {
    if (matchMedia("(max-width: 820px)").matches) document.body.classList.add("answer-open");
    renderAnswer();
    [...byId("answer").querySelectorAll("[data-cite]")].find(button => button.dataset.cite === state.selected)?.focus();
  } else messageList.querySelector('[aria-current="true"]')?.focus();
});
byId("menu").addEventListener("click", () => setMenu(!byId("sidebar").classList.contains("open")));
document.addEventListener("click", event => {
  if (!event.target.closest("#sidebar, #menu")) setMenu(false);
});
document.addEventListener("keydown", event => {
  const searchShortcut = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k";
  if (searchShortcut) {
    event.preventDefault();
    closeReader();
    setMenu(false);
    queryInput.focus();
  }
  if (event.key === "Escape") {
    if (byId("sidebar").classList.contains("open")) setMenu(false);
    else if (document.body.classList.contains("source-open")) byId("back").click();
    else if (document.body.classList.contains("answer-open")) byId("answer-back").click();
    else if (document.body.classList.contains("reader-open") && matchMedia("(max-width: 820px)").matches) byId("back").click();
    else if (document.activeElement === queryInput) { resetSearch(); render(); }
  }
});
for (const [id, name, added, removed] of [
  ["star", "starred", "Письмо добавлено в избранное", "Письмо убрано из избранного"],
  ["archive", "archived", "Письмо перемещено в архив", "Письмо возвращено во входящие"],
  ["read-toggle", "unread", "Письмо отмечено непрочитанным", "Письмо отмечено прочитанным"],
]) {
  byId(id).addEventListener("click", () => {
    const mail = visibleMessages().find(candidate => candidate.id === state.selected);
    if (!mail) return;
    const value = !flag(mail, name);
    setFlag(mail, name, value);
    if (!visibleMessages().some(item => item.id === mail.id)) closeReader();
    byId("live").textContent = value ? added : removed;
    render();
  });
}

try {
  const payload = await api("mailboxes", undefined, 3000);
  if (payload.mailboxes?.length && payload.engine) {
    state.live = true;
    state.mailboxes = payload.mailboxes;
    byId("source-label").textContent = "Корпус хакатона";
    byId("today").hidden = true;
    byId("snapshot").hidden = false;
    const semantic = document.querySelector('[data-mode="semantic"]');
    semantic.title = "Векторный поиск E5 и переранжирование BGE";
    if (payload.engine === "bm25") {
      state.mode = "lexical";
      semantic.hidden = true;
      byId("search-description").textContent = "Введите слова из темы или текста письма.";
    }
  }
} catch {
  // Sites serves the self-contained fictional mailbox.
}
mailboxSelect.replaceChildren(...state.mailboxes.map(mailbox => {
  const option = make("option", "", mailbox.name);
  option.value = mailbox.id;
  return option;
}));
await selectMailbox(state.mailboxes[0].id);
