import { demoMessages, demoSearch, mailboxes as demoMailboxes } from "./demo.js";
import { assembleAnswer, continuesSearch, highlightIds, insufficient, planQuestion, routeQuestion } from "./assistant.js";

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
  filters: {}, appliedFilters: {}, orderIds: [], reminders: null,
  supportsConditions: false, conditions: [], conditionVersion: 0, openedConfirmation: null,
  reminderMail: null, fromCitation: false, serverAssistant: false,
  entryMode: "auto", chat: [], context: null, scrollChat: false,
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
  if (state.folder === "reminders") {
    const entries = [...state.conditions, ...(state.reminders || [])];
    return [...new Map(entries.map(entry => [entry.message.id, entry.message])).values()];
  }
  return state.messages.filter(mail => {
    if (state.folder === "starred") return flag(mail, "starred");
    if (state.folder === "archive") return flag(mail, "archived");
    return !flag(mail, "archived");
  });
}

function selectedMessage() {
  return visibleMessages().find(mail => mail.id === state.selected)
    ?? state.messages.find(mail => mail.id === state.selected);
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
function conditionFor(mail) {
  return state.conditions.find(item => item.message.id === mail.id);
}

function conditionText(item) {
  const label = { pending: "Ждём подтверждения", fulfilled: "Подтверждено письмом", due: "Срок наступил" }[item.status];
  return item.status === "fulfilled" ? label : label + " · " + reminderDate.format(new Date(item.dueAt));
}

function renderCondition(item, showFact) {
  const details = make("section", "condition-status");
  details.append(make("p", "reminder-status", conditionText(item)));
  if (showFact) details.append(make("p", "", item.condition));
  if (item.evidence) {
    const source = make("button", "citation", "Открыть подтверждение");
    source.type = "button";
    source.addEventListener("click", () => {
      state.openedConfirmation = { mail: item.evidence, origin: item.message.id, fromCitation: state.fromCitation };
      openMessage(item.evidence, true);
      document.body.classList.add("source-open");
      byId("back").focus();
    });
    details.append(make("blockquote", "", item.quote), source);
  } else if (item.checkUnavailable) {
    details.append(make("p", "answer-label", "Автопроверка пока недоступна."));
  }
  return details;
}
function blank(symbol, title, description) {
  const container = make("div", "empty");
  container.append(icon(symbol), make("strong", "", title), make("p", "", description));
  return container;
}

function renderReader(mail, index, count) {
  const tools = byId("letter-tools");
  tools.hidden = !mail;
  byId("position").textContent = mail && index >= 0 && state.applied ? "Результат " + (index + 1) + " из " + count : "Письмо";
  if (!mail) {
    letter.replaceChildren(blank("mail", "Выберите письмо", "Письмо откроется здесь."));
    return;
  }
  const condition = conditionFor(mail);
  const showFactAsTitle = condition && state.folder === "reminders";
  const subject = highlighted(make("h2"), showFactAsTitle ? condition.condition : mail.subject || "Без темы");
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
  const elements = [subject, header];
  if (condition) elements.push(renderCondition(condition, !showFactAsTitle));
  elements.push(body);
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
  byId("remind").hidden = state.reminders === null && !state.supportsConditions;
  setTool("remind", reminder || condition ? "Изменить напоминание" : "Напомнить о письме");
}

function setTool(id, label, glyph) {
  const button = byId(id);
  button.title = label;
  button.setAttribute("aria-label", label);
  if (glyph) button.firstElementChild.textContent = glyph;
}

function openDraft(draft) {
  const input = byId("draft-text");
  input.value = draft.text;
  input.addEventListener("input", () => { draft.text = input.value; }, { signal: draftEditor.signal });
  byId("draft-dialog").showModal();
}

let draftEditor;
byId("draft-dialog").addEventListener("close", () => draftEditor?.abort());

function sourceButton(id, label, select = mail => openMessage(mail, true)) {
  const mail = state.messages.find(candidate => candidate.id === id);
  if (!mail) return null;
  const subject = mail.subject || "Письмо";
  const compact = subject.length > 62 ? subject.slice(0, 59).trimEnd() + "…" : subject;
  const button = make("button", "citation", label ?? [compact, when(mail.date, shortDate)].filter(Boolean).join(" · "));
  button.type = "button";
  button.title = subject;
  button.dataset.cite = mail.id;
  if (mail.id === state.selected && document.body.classList.contains("source-open")) button.setAttribute("aria-current", "true");
  button.addEventListener("click", () => select(mail));
  return button;
}

function renderConditionalDraft(card, draft) {
  card.append(make("p", "answer-status", draft.condition));
  const sources = draft.sources;
  if (!sources.length) {
    card.append(make("p", "", "Сначала найди нужное письмо или укажи его номер, затем повтори просьбу."));
    return card;
  }
  if (sources.length > 1) {
    card.append(make("p", "", "Выбери исходное письмо, чтобы я подставил его ключи."));
    const links = make("ul", "chat-results");
    for (const source of sources) {
      const button = sourceButton(source.id, null, mail => openReminder(mail, null, { ...draft, anchors: source.anchors }));
      if (button) {
        const row = make("li");
        row.append(button);
        links.append(row);
      }
    }
    card.append(links);
  } else {
    const [source] = sources;
    const original = sourceButton(source.id);
    const mail = state.messages.find(item => item.id === source.id);
    if (original && mail) {
      card.append(original);
      const button = make("button", "secondary remind-from-answer", "Проверить и сохранить");
      button.type = "button";
      button.addEventListener("click", () => openReminder(mail, null, { ...draft, anchors: source.anchors }));
      const actions = make("div", "chat-actions");
      actions.append(button);
      card.append(actions);
    }
  }
  return card;
}

function renderTurn(turn) {
  const card = make("section", "chat-turn");
  card.append(make("h3", "question-title", turn.query));
  if (turn.pending || turn.error) {
    card.append(make("p", "answer-label", turn.error ? "Не удалось получить результат. Попробуйте ещё раз." : "Ищем письма…"));
    return card;
  }
  const { result, evidence, count } = turn;
  if (result?.conditionalDraft) return renderConditionalDraft(card, result.conditionalDraft);
  if (turn.routing?.router === "jev") {
    const path = turn.routing.route === "search" ? "Поиск" : "Ассистент";
    card.append(make("p", "answer-label", "Jev · " + path));
  }
  if (turn.searchIds) {
    card.append(make("p", "answer-status", count ? "Найдено: " + messageCount(count) : "Письма не найдены."));
    const links = make("ul", "chat-results");
    for (const id of turn.searchIds.slice(0, 3)) {
      const link = sourceButton(id);
      if (link) {
        const row = make("li");
        row.append(link);
        links.append(row);
      }
    }
    if (links.childElementCount) card.append(links);
    if (count) {
      const button = make("button", "secondary chat-show-results", "К списку писем");
      button.type = "button";
      button.addEventListener("click", () => byId("answer-back").click());
      card.append(button);
    }
    return card;
  }
  if (result?.refund && !result.actionOnly) {
    card.append(make("p", "answer-status", result.confirmed
      ? "В просмотренных письмах есть подтверждение зачисления."
      : "В просмотренных письмах зачисление не подтверждено."));
  }
  if (result?.timeline?.length) {
    card.append(make("h4", "chat-subtitle", "Статусы в письмах по номеру"));
    const timeline = make("ol", "timeline");
    for (const event of result.timeline) {
      const row = make("li");
      const source = sourceButton(event.id, "Открыть письмо");
      row.append(make("span", "timeline-date", [when(event.date, shortDate), event.time.slice(0, 5)].filter(Boolean).join(" · ")),
        make("strong", "", event.status));
      if (source) row.append(source);
      timeline.append(row);
    }
    card.append(timeline);
  }
  if (evidence.length && !result?.actionOnly) {
    const provider = result?.provider === "deepseek" ? "Источники подобрал DeepSeek" : "Выдержки из писем";
    const coverage = result?.reviewed ? " · просмотрено " + result.reviewed + " из " + count : "";
    card.append(make("p", "answer-label", provider + coverage));
    const citations = make("ol", "evidence");
    for (const item of evidence) {
      const row = make("li");
      const cite = sourceButton(item.id);
      if (!cite) continue;
      if (item.kind === "confirmed") row.append(make("p", "evidence-kind", "Подтверждение"));
      if (item.kind === "promised") row.append(make("p", "evidence-kind", "Обещание или оформление"));
      row.append(highlighted(make("blockquote"), item.text), cite);
      citations.append(row);
    }
    card.append(citations);
  } else if (!result?.actionOnly && !result?.refund && !result?.timeline?.length) {
    card.append(make("p", "", insufficient));
  }

  const actions = make("div", "chat-actions");
  const proposal = result?.reminder;
  const reminderId = proposal?.id ?? (!result && evidence.length === 1 ? evidence[0].id : null);
  const mail = reminderId && state.messages.find(candidate => candidate.id === reminderId);
  if (mail && state.reminders !== null) {
    const button = make("button", "secondary remind-from-answer", reminderFor(mail) ? "Изменить напоминание" : "Напомнить о письме");
    button.type = "button";
    button.addEventListener("click", () => openReminder(mail, proposedTime(proposal)));
    actions.append(button);
  }
  if (result?.draft) {
    const button = make("button", "secondary", "Черновик ответа");
    button.type = "button";
    button.addEventListener("click", () => {
      draftEditor?.abort();
      draftEditor = new AbortController();
      openDraft(result.draft);
    });
    actions.append(button);
  }
  if (actions.childElementCount) {
    if (result?.actionOnly) {
      const source = sourceButton(proposal?.id ?? result.draft?.id);
      if (source) card.append(source);
    }
    card.append(actions);
  } else if (result?.actionOnly) {
    card.append(make("p", "", result.confirmed
      ? "Нашлось подтверждение зачисления. Условие для напоминания не выполнено."
      : "Не удалось подготовить действие по этим письмам."));
  }
  return card;
}

function renderAnswer() {
  if (state.view !== "ask") return;
  const answer = byId("answer");
  const scroll = answer.scrollTop;
  const currentFocus = document.activeElement?.closest?.(".citation");
  const nodes = state.chat.map(renderTurn);
  if (!nodes.length) nodes.push(blank("auto_awesome", "Спросите о письмах", state.context?.ids.length
    ? "Уточнения будут опираться на найденные письма."
    : "Например, когда согласовали доставку или что ответили покупателю."));
  answer.replaceChildren(...nodes);
  if (state.scrollChat && nodes.length) {
    const latest = nodes[nodes.length - 1].getBoundingClientRect().top;
    answer.scrollTop += latest - answer.getBoundingClientRect().top - 20;
  } else answer.scrollTop = scroll;
  state.scrollChat = false;
  byId("chat-submit").disabled = state.pending;
  byId("new-chat").hidden = !state.context && !state.chat.length;
  if (currentFocus) {
    [...answer.querySelectorAll(".citation")].find(node => node.dataset.cite === currentFocus.dataset.cite)?.focus({ preventScroll: true });
  }
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
  const proof = state.openedConfirmation?.mail;
  const fromHistory = (state.view === "ask" && document.body.classList.contains("source-open")) || Boolean(proof);
  if (!fromHistory && !rows.some(mail => mail.id === state.selected)) state.selected = rows[0]?.id ?? null;
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
    const conditional = state.folder === "reminders" && conditionFor(mail);
    const sender = make("span", "sender");
    if (unread) {
      const mark = make("span", "unread-mark");
      mark.title = "Непрочитанное";
      sender.append(mark);
    }
    sender.append(make("span", "clip", conditional ? "Условное напоминание" : mail.sender || mail.category || "Письмо"));
    const time = make("time", "", mail.date === state.mailbox.asOf && mail.time ? mail.time : when(mail.date, shortDate));
    if (mail.date) time.dateTime = mail.date;
    const reminder = state.folder === "reminders" && reminderFor(mail);
    const snippet = conditional ? conditionText(conditional) : reminder ? reminderText(reminder) :
      (mail.body || "").replace(/\s+/gu, " ");
    button.append(avatar(mail), sender, time,
      highlighted(make("span", "subject"), conditional ? conditional.condition : mail.subject || "Без темы"),
      highlighted(make("span", "snippet" + (reminder || conditional ? " reminder-row" : "")), snippet));
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
  byId("show-answer").hidden = state.view !== "ask";
  byId("ask-results").hidden = state.view !== "search" || !state.context?.ids.length;
  document.body.classList.toggle("asking", state.view === "ask");
  document.querySelector('[data-folder="reminders"]').hidden = state.reminders === null && !state.supportsConditions;
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
  const selected = rows[index] ?? proof ?? (fromHistory ? state.messages.find(mail => mail.id === state.selected) : null);
  renderReader(selected, index, rows.length);
  if (state.live && selected && (state.view === "search" || fromHistory)) loadLetter(selected);
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
  const ordinary = state.reminders?.filter(item => Date.parse(item.dueAt) <= Date.now()).length || 0;
  byId("reminder-count").textContent = (ordinary + state.conditions.filter(item => item.status === "due").length) || "";
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

async function refreshConditions() {
  if (!state.supportsConditions) return;
  const version = ++state.conditionVersion;
  const params = new URLSearchParams({ mailbox: state.mailbox.id, asOf: state.snapshot });
  try {
    const data = await api("conditional-reminders?" + params, undefined, 35_000);
    if (state.conditionVersion === version) {
      state.conditions = data.reminders;
      render();
    }
  } catch {
    if (state.conditionVersion === version && state.folder === "reminders") notify("Не удалось проверить условия.");
  }
}

function localDateTime(date) {
  const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return shifted.toISOString().slice(0, 16);
}

function proposedTime(proposal) {
  if (!proposal?.time) return null;
  const [hour, minute] = proposal.time.split(":").map(Number);
  const date = new Date();
  date.setDate(date.getDate() + proposal.days);
  date.setHours(hour, minute, 0, 0);
  return date > new Date() ? date : null;
}

function openReminder(mail, proposed = null, draft = null) {
  const item = reminderFor(mail);
  const savedCondition = conditionFor(mail);
  const conditional = draft || (!proposed && savedCondition);
  state.reminderMail = mail;
  byId("reminder-subject").textContent = mail.subject || "Без темы";
  byId("condition-toggle").hidden = !state.supportsConditions;
  byId("reminder-conditional").checked = Boolean(conditional);
  byId("reminder-condition").value = conditional?.condition ?? "";
  byId("condition-anchors").value = (savedCondition?.anchors ?? conditional?.anchors ??
    state.details.get(mail.id)?.conditionAnchors ?? []).join(", ");
  setReminderMode(Boolean(conditional), proposed ?? (item ? new Date(item.dueAt) : null));
  if (draft) byId("reminder-time").value = draft.dueAt ?? "";
  byId("reminder-error").hidden = true;
  byId("reminder-dialog").showModal();
}

function setReminderMode(conditional, ordinaryTime = null) {
  const mail = state.reminderMail;
  const item = conditional ? conditionFor(mail) : reminderFor(mail);
  byId("condition-fields").hidden = !conditional;
  byId("reminder-condition").required = conditional;
  byId("reminder-presets").hidden = conditional;
  byId("remove-reminder").hidden = !item;
  if (conditional) {
    const tomorrow = new Date(Date.parse(state.snapshot + "T12:00Z") + 86_400_000).toISOString().slice(0, 10);
    byId("reminder-time").value = item?.dueAt ?? (tomorrow + "T09:00");
    byId("reminder-zone").textContent = "Дата берётся из архива. Подтверждение проверяется в новых письмах.";
  } else {
    byId("reminder-time").value = localDateTime(ordinaryTime ?? (item ? new Date(item.dueAt) : new Date(Date.now() + 60 * 60_000)));
    byId("reminder-zone").textContent = "Часовой пояс " + Intl.DateTimeFormat().resolvedOptions().timeZone;
  }
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
  const rawTime = byId("reminder-time").value;
  const time = new Date(rawTime);
  const conditional = state.supportsConditions && byId("reminder-conditional").checked;
  const error = byId("reminder-error");
  if (!remove && (!Number.isFinite(time.getTime()) || (conditional ? rawTime.slice(0, 10) <= state.snapshot : time.getTime() <= Date.now()))) {
    error.textContent = conditional ? "Выберите дату после текущего среза архива." : "Выберите время в будущем.";
    error.hidden = false;
    return;
  }
  const controls = [...byId("reminder-form").querySelectorAll("button")];
  controls.forEach(button => { button.disabled = true; });
  const data = { mailbox: mail.mailbox, messageId: mail.id };
  if (!remove) data.dueAt = conditional ? rawTime : time.toISOString();
  if (conditional) {
    data.asOf = state.snapshot;
    data.condition = byId("reminder-condition").value.trim();
    data.anchors = byId("condition-anchors").value.split(",").map(value => value.trim()).filter(Boolean);
    if (!remove && (!data.condition || data.anchors.length > 6 || data.anchors.some(value => value.length < 2 || value.length > 100))) {
      error.textContent = "Укажите факт и до шести ключей длиной от 2 до 100 символов.";
      error.hidden = false;
      controls.forEach(button => { button.disabled = false; });
      return;
    }
  }
  try {
    const result = await api(conditional ? "conditional-reminders" : "reminders", jsonRequest(remove ? "DELETE" : "PUT", data));
    if (state.mailbox.id === mail.mailbox) {
      if (conditional) refreshConditions();
      else state.reminders = result.reminders;
    }
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
  state.chat = [];
  state.context = null;
  byId("chat-query").value = "";
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
  state.openedConfirmation = null;
  document.body.classList.remove("reader-open", "source-open", "answer-open");
}
function openMessage(mail, fromCitation = false) {
  if (state.openedConfirmation?.mail.id !== mail.id) state.openedConfirmation = null;
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
  const stayInReminders = state.folder === "reminders" && state.mailbox.id === id && asOf;
  resetSearch();
  setMenu(false);
  state.folder = stayInReminders ? "reminders" : "inbox";
  state.reminders = null;
  state.conditions = [];
  state.mailbox = state.mailboxes.find(mailbox => mailbox.id === id);
  state.snapshot = asOf || state.mailbox.asOf;
  byId("account-mark").textContent = state.mailbox.initials;
  byId("today").textContent = when(state.snapshot, fullDate);
  byId("snapshot-date").value = state.snapshot;
  showExamples();
  refreshReminders();
  refreshConditions();
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
async function search({ followup = false } = {}) {
  const input = followup ? byId("chat-query") : queryInput;
  const query = input.value.trim();
  if (!query && (followup || !Object.values(state.filters).some(Boolean))) {
    if (!followup) {
      resetSearch();
      render();
    }
    return;
  }
  if (followup && state.pending) return;
  const version = ++state.version;
  const mode = state.entryMode;
  const route = mode === "auto" ? routeQuestion(query) : mode;
  const continuation = continuesSearch(query);
  const useContext = followup || (mode !== "search" && continuation);
  if (!useContext) {
    state.chat = [];
    state.context = null;
  }
  if (followup) input.value = "";
  const inChat = followup || mode !== "search";
  state.view = inChat ? "ask" : "search";
  const ask = route === "ask" || route === "auto-ask";
  const autoRoute = mode === "auto" && query ? (ask ? "agent" : "search") : "";
  const plan = planQuestion(query, state.snapshot, [...new Set(state.messages.map(mail => mail.sender).filter(Boolean))], state.filters);
  const previous = useContext ? state.context : null;
  const previousIds = new Set(previous?.orderIds.map(id => id.toLowerCase()));
  const explicitNewScope = plan.ids.some(id => !previousIds.has(id.toLowerCase()));
  const newSearch = mode === "auto" && !ask && !continuation
    && /^(?:найди|покажи|выведи|перечисли|подбери)\s/iu.test(query);
  const scope = previous && !explicitNewScope && !newSearch ? previous : null;
  const focusedId = useContext && !explicitNewScope && document.body.classList.contains("source-open")
    ? state.selected : null;
  const filters = scope?.filters ?? (mode === "search" ? state.filters : plan.filters);
  const orderIds = plan.ids.length ? plan.ids : scope?.orderIds ?? [];
  const turn = { query, pending: true, evidence: [] };
  if (inChat) {
    state.chat.push(turn);
    state.scrollChat = true;
  }
  state.query = query;
  state.applied = true;
  state.appliedFilters = filters;
  state.orderIds = orderIds;
  state.selected = null;
  state.results = null;
  state.date = scope?.date ?? null;
  state.pending = true;
  state.error = "";
  closeReader();
  if (inChat && matchMedia("(max-width: 820px)").matches) document.body.classList.add("answer-open");
  messageList.scrollTop = 0;
  render();
  try {
    const searchText = ask || autoRoute ? plan.query : query;
    const request = { mailbox: state.mailbox.id, query: searchText, mode: state.mode, asOf: state.snapshot, filters, orderIds };
    const demoCandidates = !orderIds.length ? state.messages : state.messages.filter(mail =>
      orderIds.some(id => highlightIds(mail.subject + " " + mail.body, [id]).length > 1));
    const agent = state.serverAssistant && (ask || Boolean(autoRoute)) && Boolean(query);
    let result;
    if (agent) {
      const context = scope ? [...new Set([scope.root, scope.latest])].filter(Boolean) : [];
      result = await api("ask", jsonRequest("POST", {
        ...request, question: query, context, autoRoute, sourceIds: scope?.ids ?? [], focusedId,
      }), 120_000);
    } else if (scope?.ids.length) {
      result = { ids: scope.ids, date: scope.date };
    } else if (state.live) {
      result = await api("search", jsonRequest("POST", request), 120_000);
    } else {
      result = demoSearch(demoCandidates, searchText, state.snapshot, state.mode, filters);
    }
    if (state.version !== version) return;
    if (result.messages) {
      const emails = new Map(state.messages.map(mail => [mail.id, mail]));
      for (const mail of result.messages) emails.set(mail.id, mail);
      state.messages = [...emails.values()];
    }
    state.results = result.ids;
    state.date = result.date ?? scope?.date ?? null;
    state.context = { root: scope?.root ?? query, latest: query, ids: result.ids, orderIds, filters, date: state.date };
    turn.count = result.ids.length;
    turn.routing = result.routing;
    if (result.answer) {
      turn.result = result.answer;
      turn.evidence = result.answer.evidence;
    } else if (ask && !agent) {
      const top = visibleMessages().slice(0, 8);
      const sources = state.live ? await Promise.all(top.map(mail => api("message?" + new URLSearchParams({
        mailbox: mail.mailbox, id: mail.id,
      })).then(payload => payload.message).catch(() => null))) : top;
      if (state.version !== version) return;
      turn.evidence = assembleAnswer(query, sources.filter(Boolean));
    } else turn.searchIds = result.ids;
  } catch {
    if (state.version !== version) return;
    state.error = "search";
    turn.error = true;
  }
  turn.pending = false;
  state.pending = false;
  if (inChat) state.scrollChat = true;
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
byId("reminder-conditional").addEventListener("change", event => setReminderMode(event.target.checked));
byId("remove-reminder").addEventListener("click", () => saveReminder(true));
byId("remind").addEventListener("click", () => {
  const mail = selectedMessage();
  if (mail) openReminder(mail);
});
setInterval(() => {
  if (state.reminders === null) return;
  const previous = byId("reminder-count").textContent;
  updateReminderCount();
  if (previous !== byId("reminder-count").textContent && state.folder === "reminders") render();
}, 15_000);

function setMode(mode) {
  state.entryMode = mode;
  state.view = mode === "ask" ? "ask" : "search";
  for (const button of document.querySelectorAll("[data-view]")) {
    button.setAttribute("aria-pressed", String(button.dataset.view === mode));
  }
  byId("submit-query").textContent = mode === "ask" ? "Спросить" : "Найти";
  queryInput.placeholder = mode === "ask" ? "Что нужно узнать из писем?" : "Опишите письмо или введите номер заказа";
  showExamples();
}

function openChat() {
  closeReader();
  state.view = "ask";
  if (!state.chat.length && state.context?.root) {
    const { root, ids } = state.context;
    state.chat.push({ query: root, count: ids.length, searchIds: ids });
  }
  document.body.classList.add("answer-open");
  render();
  byId("chat-query").focus();
}

for (const button of document.querySelectorAll("[data-view]")) {
  button.addEventListener("click", () => {
    if (button.dataset.view !== "search") {
      setMode(button.dataset.view);
      return openChat();
    }
    closeReader();
    setMode(button.dataset.view);
    render();
  });
}

byId("ask-results").addEventListener("click", () => {
  setMode("ask");
  openChat();
});
byId("chat-form").addEventListener("submit", event => {
  event.preventDefault();
  search({ followup: true });
});
byId("new-chat").addEventListener("click", () => {
  resetSearch();
  openChat();
});
byId("copy-draft").addEventListener("click", async () => {
  const input = byId("draft-text");
  try {
    await navigator.clipboard.writeText(input.value);
    notify("Черновик скопирован");
    byId("draft-dialog").close();
  } catch {
    input.select();
  }
});

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
    setMode("auto");
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
  const opening = state.openedConfirmation;
  closeReader();
  if (opening) {
    const origin = state.conditions.find(item => item.message.id === opening.origin)?.message;
    if (origin) openMessage(origin, opening.fromCitation);
    return;
  }
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
    const mail = selectedMessage();
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
    state.serverAssistant = payload.assistant === true;
    state.supportsConditions = payload.conditionalReminders === true;
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
