export const demoDate = "2026-11-21";
export const mailboxes = [
  { id: "anna", name: "Анна Королёва", initials: "АК", asOf: demoDate, examples: [
    "Покупатель хочет вернуть деньги",
    "Где не прошла оплата",
    "Что вчера писали про доставку",
  ] },
  { id: "max", name: "Максим Орлов", initials: "МО", asOf: demoDate, examples: [
    "Когда разгрузили контейнер",
    "Какая машина приедет завтра",
    "Где повреждён груз",
  ] },
];

const entry = (id, mailbox, sender, address, subject, date, time, category, tone, body, extra) =>
  ({ id, mailbox, sender, address, subject, date, time, category, tone, body, ...extra });

export const demoMessages = [
  entry("a1", "anna", "Мария Котова", "maria.kotova@example.com",
    "Заявка на возврат заказа № 78431", "2026-11-21", "10:42", "Возвраты", 0,
    "Анна, добрый день!\n\nКлиент передал заказ № 78431 в пункт выдачи. Склад подтвердил приём, состояние товара проверили. Можно оформить перечисление на карту, с которой оплачивали покупку.\n\nПодтвердите, пожалуйста, когда платёж уйдёт. Клиенту обещали сообщить срок сегодня до 16:00.\n\nМария Котова\nСлужба поддержки",
    { details: { "Заказ": "№ 78431", "Сумма": "4 890 ₽", "Статус": "Подтверждён складом" }, topics: ["refund"], unread: true }),
  entry("a2", "anna", "Евгений Литвинов", "e.litvinov@example.com",
    "Сбой эквайринга · заказы 78452 и 78453", "2026-11-21", "09:18", "Оплата", 1,
    "Анна, два платежа за утро получили отказ от банка. У заказов 78452 и 78453 деньги со счетов клиентов не списаны.\n\nПлатёжный шлюз уже снова отвечает. Отправьте покупателям новую ссылку на оплату, когда они подтвердят заказ.",
    { topics: ["payment"], unread: true }),
  entry("a3", "anna", "Ирина Волкова", "irina.volkova@example.com",
    "Новый интервал доставки для заказа № 78120", "2026-11-20", "17:35", "Доставка", 2,
    "Курьер свяжется с получателем в понедельник. Новое окно доставки с 14:00 до 18:00. Адрес остался прежним.\n\nОбнови, пожалуйста, дату в карточке заказа.",
    { topics: ["delivery"], unread: true }),
  entry("a4", "anna", "Денис Ким", "denis.kim@example.com",
    "Обмен товара или компенсация · заказ № 77904", "2026-11-20", "14:06", "Возвраты", 3,
    "Покупатель получил чайник с повреждённой ручкой. От обмена отказался и просит компенсировать полную стоимость на исходный способ оплаты. Фото вложил в заявку, проверка завершена.",
    { topics: ["refund", "damage"] }),
  entry("a5", "anna", "Ольга Громова", "olga.gromova@example.com",
    "Закрывающие документы за октябрь", "2026-11-20", "11:24", "Документы", 4,
    "Акты и счета за октябрь согласованы, оригиналы передали курьеру. Сверь, пожалуйста, сумму по договору до понедельника.",
    { topics: ["documents"], starred: true }),
  entry("a6", "anna", "Роман Савельев", "r.saveliev@example.com",
    "Задержка на маршруте № 42", "2026-11-19", "16:10", "Доставка", 1,
    "На маршруте № 42 сломался фургон. Доставку заказов перенесли на следующий рабочий день. Покупатели получат уведомление после 18:00.",
    { topics: ["delivery"], unread: true }),
  entry("a7", "anna", "Мария Котова", "maria.kotova@example.com",
    "Отмена заказа № 77318", "2026-11-19", "12:12", "Возвраты", 0,
    "Заказ отменили до передачи в доставку. 2 390 ₽ уже перечислили обратно на карту клиента. Банк подтвердил операцию.",
    { topics: ["refund"] }),
  entry("a8", "anna", "Денис Ким", "denis.kim@example.com",
    "График поддержки на выходные", "2026-11-18", "15:03", "Команда", 3,
    "В субботу обращения до 14:00 разбирает Ирина, затем смену принимает Роман. В воскресенье дежурю я.",
    { topics: [] }),
  entry("m1", "max", "Алёна Миронова", "a.mironova@example.com",
    "Контейнер MSKU 210478 · приёмка завершена", "2026-11-21", "11:20", "Склад", 0,
    "Контейнер MSKU 210478 вскрыли в 09:40. В 10:55 выгрузили последнюю палету, сверили накладную. Приняли 18 мест без расхождений.\n\nДокументы оставила на стойке приёмки.",
    { details: { "Контейнер": "MSKU 210478", "Мест": "18", "Выгрузка": "10:55" }, topics: ["unload"], unread: true }),
  entry("m2", "max", "Виктор Павлов", "v.pavlov@example.com",
    "График машин на воскресенье", "2026-11-21", "10:30", "Транспорт", 1,
    "22 ноября на склад приедет тягач М742КМ. Время подачи машины 08:30, ворота № 3. Водителю отправили схему проезда.",
    { topics: ["truck"], unread: true }),
  entry("m3", "max", "Яна Семёнова", "y.semenova@example.com",
    "Замечания к палете из партии 2574", "2026-11-20", "16:45", "Контроль качества", 4,
    "На палете из партии 2574 обнаружили продавленные коробки и сорванную пломбу. Товар временно разместили в зоне карантина. Ждём решения перевозчика по акту.",
    { topics: ["damage"], unread: true }),
  entry("m4", "max", "Алёна Миронова", "a.mironova@example.com",
    "Контейнер TLLU 640019 · очередь на рампе", "2026-11-20", "15:31", "Склад", 0,
    "Контейнер TLLU 640019 ждёт свободную рампу. Выгрузку запланировали на субботу, за фактическим временем обращайтесь к мастеру смены.",
    { topics: ["unload"] }),
  entry("m5", "max", "Виктор Павлов", "v.pavlov@example.com",
    "Путевой лист рейса № 513", "2026-11-19", "09:15", "Документы", 1,
    "Путевой лист закрыт. Пробег по рейсу составил 286 км, показания сверили. Оригинал передан в бухгалтерию.",
    { topics: ["documents"], starred: true }),
];

const topicPatterns = {
  refund: /возврат|верн[уёе]|обратно|компенс|отмен.*заказ|отказ.*товар/u,
  payment: /эквайр|сбой.*плат|отказ.*банк|не\s+прош|отклон|не\s+спис|не\s+оплат/u,
  delivery: /достав|курьер|маршрут|привез|опаздыва|перенос/u,
  unload: /разгруз|выгруз|разобрал.*контейнер|распаковал|контейнер|палет.*рамп/u,
  truck: /машин|тягач|фургон|грузовик|приед|транспорт/u,
  damage: /поврежд|дефект|сломан|продавлен|испорч|брак|помят/u,
  documents: /документ|акты|накладн|закрывающ|сч[её]т|бухгалтер/u,
};

const stopWords = new Set("а без в где для до за и из к как какие когда ли мне на найди о об от по покажи про с со у что чтобы это хочу хочет письмо письма писали наш все".split(" "));
const relative = /(?<![\p{L}\p{N}])(?:\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?|вчера|позавчера|сегодня)(?![\p{L}\p{N}])/giu;
const months = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(" ");

export const normalize = text => text.toLowerCase().replaceAll("ё", "е");

export function words(text) {
  return (normalize(text).match(/[\p{L}\p{N}]+/gu) || [])
    .filter(word => word.length > 1 && !stopWords.has(word));
}

function dateInQuery(text, asOf) {
  const matches = [...text.matchAll(relative)];
  if (matches.length !== 1) return null;
  const value = matches[0][0].toLowerCase();
  const offset = { вчера: -1, позавчера: -2, сегодня: 0 }[value];
  const anchor = new Date(asOf + "T12:00:00Z");
  if (offset !== undefined) {
    anchor.setUTCDate(anchor.getUTCDate() + offset);
    return anchor.toISOString().slice(0, 10);
  }
  const [day, month, year = asOf.slice(0, 4)] = value.split(/\s+/u);
  const iso = [year, String(months.indexOf(month) + 1).padStart(2, "0"), day.padStart(2, "0")].join("-");
  const date = new Date(iso + "T12:00:00Z");
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === iso ? iso : null;
}

export function demoSearch(messages, query, asOf, mode, filters = {}) {
  const targetDate = dateInQuery(query, asOf);
  const text = targetDate ? query.replace(relative, "") : query;
  const terms = [...new Set(words(text))];
  const topics = Object.entries(topicPatterns).filter(([, pattern]) => pattern.test(text)).map(([topic]) => topic);
  const useTopics = mode === "semantic" && topics.length > 0;
  const candidates = messages.filter(mail =>
    (!targetDate || mail.date === targetDate) &&
    (!filters.from || mail.date && mail.date >= filters.from) &&
    (!filters.to || mail.date && mail.date <= filters.to) &&
    (!filters.sender || normalize(mail.sender + " " + mail.address).includes(normalize(filters.sender)))
  );
  const ranked = candidates.map(mail => {
    const subject = words(mail.subject);
    const body = words(mail.sender + " " + mail.body);
    const keywords = terms.reduce((sum, term) => sum + Number(subject.includes(term)) * 3 + Number(body.includes(term)), 0);
    const concepts = topics.filter(topic => mail.topics.includes(topic)).length;
    return { id: mail.id, score: useTopics ? (concepts ? keywords + concepts * 9 : 0) : keywords };
  });
  const filterOnly = !query || !terms.length && (targetDate || Object.values(filters).some(Boolean));
  return { ids: ranked.filter(item => item.score > 0 || filterOnly)
    .sort((a, b) => b.score - a.score).map(item => item.id), date: targetDate };
}
