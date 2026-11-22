import { normalize, words } from "./demo.js";

const orderPattern = /(?:№\s*|(?:заказ[а-яё]*|рейс[а-яё]*|парти[а-яё]*)\s*#?\s*)(\d{2,16})(?!\d)/giu;
const archiveReference = /\bSYN-[0-9a-f]{6,24}\b/giu;

export function routeQuestion(query) {
  const text = normalize(query).trim();
  if (/(?:^|[^\p{L}])(?:напомни|черновик[аи]?|объясни|сравни|подытожь|резюмируй|сводк[уа]|хронологи[яюи])(?!\p{L})/u.test(text)
      || /(?:подготовь|составь|напиши)\s+.{0,35}(?:ответ|письмо)/u.test(text)) return "ask";
  const rows = /^(?:найди|покажи|выведи|перечисли|подбери)(?!\p{L})/u;
  const question = /(?:^|[^\p{L}])(?:когда|почему|сколько|кто|что ответил\p{L}*|что изменил\p{L}*|есть ли|вернули ли|пришли ли|подтвердили ли|в каком статусе|какой статус)(?!\p{L})/u;
  if (!rows.test(text)) return text.endsWith("?") || question.test(text) ? "ask" : "search";
  if (!text.includes("?") && !question.test(text)) return "search";
  const clauses = text.split(/[.!?;]\s+|,\s+(?:а\s+|и\s+)/u);
  return clauses.slice(1).some(part => part.endsWith("?") || question.test(part)) ? "auto-ask" : "auto-search";
}

export function continuesSearch(query) {
  const text = normalize(query).trim().replace(/^найди письма[.!?;,\s]+/u, "");
  return /^(?:а\s|тогда\s|и\s+что\s|что изменил|напомни\s|подготовь\s|составь\s)|(?<!\p{L})по\s+(?:этим письмам|этому номеру)(?!\p{L})/u.test(text);
}
const addressPattern = /[\w.+-]+@[\w.-]+\.[a-z]{2,}/iu;
const sendingPattern = /(?:^|\s)(?:от|отправитель)(?=\s|:)\s*:?\s*/iu;
const intentPatterns = {
  promised: /обещ|сообщ|ответ|подтверд/u,
  time: /\d{1,2}:\d{2}|\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)|сегодня|завтра|вчера|понедельник|вторник|сред[ауе]|четверг|пятниц|суббот|воскресен/u,
  amount: /\d[\d,. \u00a0]*\s*(?:₽|руб(?:лей|ля|ль|\.)?|евро|доллар)/u,
};

const day = (asOf, offset) => {
  const date = new Date(asOf + "T12:00:00Z");
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().slice(0, 10);
};

export function planQuestion(question, asOf, senderNames, filters) {
  const derived = { ...filters };
  let query = question;
  const address = question.match(addressPattern)?.[0];
  const senderPrefix = question.match(sendingPattern);
  if (!derived.sender && address) {
    derived.sender = address;
    query = query.replace(address, "").replace(sendingPattern, " ");
  } else if (!derived.sender && senderPrefix) {
    const offset = senderPrefix.index + senderPrefix[0].length;
    const tail = question.slice(offset);
    const matching = senderNames.filter(name => normalize(tail).startsWith(normalize(name)));
    if (matching.length === 1) {
      derived.sender = matching[0];
      query = question.slice(0, senderPrefix.index) + " " + tail.slice(matching[0].length);
    }
  }
  if (!derived.from && !derived.to) {
    const lastWeek = /за прошлую неделю|на прошлой неделе/iu;
    const lastDays = /за последние\s+(\d{1,2})\s+(?:дней|дня|день)/iu;
    const match = query.match(lastDays);
    if (lastWeek.test(query)) {
      const weekday = new Date(asOf + "T12:00:00Z").getUTCDay();
      const through = -((weekday + 6) % 7) - 1;
      derived.from = day(asOf, through - 6);
      derived.to = day(asOf, through);
      query = query.replace(lastWeek, "");
    } else if (match && Number(match[1]) > 0) {
      derived.from = day(asOf, 1 - Number(match[1]));
      derived.to = asOf;
      query = query.replace(lastDays, "");
    }
  }
  return {
    query: query.trim(),
    filters: derived,
    ids: [...new Set([
      ...[...question.matchAll(orderPattern)].map(match => match[1]), ...(question.match(archiveReference) ?? []),
    ])],
  };
}

export function highlightIds(text, ids) {
  if (!ids.length) return [text];
  const pattern = new RegExp("(?<![\\p{L}\\p{N}])(" + ids.join("|") + ")(?![\\p{L}\\p{N}])", "giu");
  return text.split(pattern);
}

function evidenceIntent(question) {
  const normalized = normalize(question);
  if (/когда|во сколько|к какому сроку|срок|до скольк/u.test(normalized)) {
    return /обещ|ответ|сообщ/u.test(normalized) ? "promised" : "time";
  }
  if (/сколько сто|какая сумма|сколько.*(?:руб|денег|верну|оплат)/u.test(normalized)) return "amount";
  return "summary";
}

export const insufficient = "В найденных письмах не нашлось ответа на этот вопрос.";

export function assembleAnswer(question, documents) {
  const intent = evidenceIntent(question);
  const tokens = new Set(words(question).filter(token => token.length > 2));
  const scored = [];
  for (const [index, mail] of documents.entries()) {
    const body = mail.body || "";
    const sentences = body.match(/[^.!?\n]+(?:[.!?]+|$)/gu) || body.split("\n");
    for (const [position, sentence] of sentences.entries()) {
      const text = sentence.trim();
      const normalized = normalize(text);
      if (text.length < 14 || text.length > 390 || /^(?:привет|здравствуйте|добрый (?:день|вечер)|спасибо|с уважением)(?!\p{L})/iu.test(text) || /\?$/u.test(text)) continue;
      let strength = 0;
      if (intent === "time") strength = Number(intentPatterns.time.test(normalized));
      if (intent === "promised") strength = Number(intentPatterns.time.test(normalized) && intentPatterns.promised.test(normalized));
      if (intent === "amount") strength = Number(intentPatterns.amount.test(normalized));
      const overlap = new Set(words(text).filter(token => tokens.has(token))).size;
      if (intent !== "summary" && !strength) continue;
      if (intent === "summary" && index >= 2 && !overlap) continue;
      if (/^(?:напомните|уточните|подскажите|подтвердите)(?!\p{L})/iu.test(text)) continue;
      scored.push({ id: mail.id, text, score: strength * 10 + overlap * 3 - index * 1.5 - position * 0.2 });
    }
    if (intent === "amount") {
      for (const [label, value] of Object.entries(mail.details || {})) {
        if (intentPatterns.amount.test(value)) scored.push({ id: mail.id, text: label + " — " + value, score: 17 - index });
      }
    }
  }
  scored.sort((a, b) => b.score - a.score);
  const chosen = [];
  for (const item of scored) {
    if (chosen.some(previous => previous.id === item.id || previous.text === item.text)) continue;
    chosen.push(item);
    if (chosen.length === (intent === "summary" ? 3 : 2)) break;
  }
  return chosen;
}
