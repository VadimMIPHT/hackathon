// The build prepends the public demo assets and their fictional mailbox metadata.
const json = (value, status = 200) => Response.json(value, {
  status, headers: { "cache-control": "no-store" },
});
const mailKey = (mailbox, messageId) => mailbox + "/" + messageId;

async function listReminders(db, owner, mailbox) {
  const { results } = await db.prepare(
    "SELECT message_id, due_at FROM reminders WHERE owner_id = ? AND mailbox_id = ? ORDER BY due_at, message_id"
  ).bind(owner, mailbox).all();
  return results.flatMap(row => {
    const message = demoMail[mailKey(mailbox, row.message_id)];
    return message ? [{ message, dueAt: row.due_at }] : [];
  });
}

async function reminders(request, env, url) {
  const owner = request.headers.get("oai-authenticated-user-id");
  if (!owner) return json({ error: "Sign in required" }, 401);
  if (!env.DB) return json({ error: "Reminder storage unavailable" }, 503);
  if (request.method === "GET") {
    const mailbox = url.searchParams.get("mailbox");
    if (!mailboxIds.includes(mailbox)) return json({ error: "Unknown mailbox" }, 400);
    return json({ reminders: await listReminders(env.DB, owner, mailbox) });
  }
  if (!["PUT", "DELETE"].includes(request.method)) return json({ error: "Method not allowed" }, 405);
  const origin = request.headers.get("Origin");
  if (origin && origin !== url.origin) return json({ error: "Same-origin requests only" }, 403);
  if (!request.headers.get("Content-Type")?.startsWith("application/json")) return json({ error: "Invalid content type" }, 415);
  let data;
  try {
    const payload = await request.text();
    if (payload.length > 2048) return json({ error: "Payload too large" }, 413);
    data = JSON.parse(payload);
  } catch {
    return json({ error: "Invalid JSON" }, 400);
  }
  const { mailbox, messageId, dueAt } = data || {};
  if (typeof mailbox !== "string" || typeof messageId !== "string" || !Object.hasOwn(demoMail, mailKey(mailbox, messageId))) {
    return json({ error: "Unknown message" }, 400);
  }
  if (request.method === "DELETE") {
    await env.DB.prepare(
      "DELETE FROM reminders WHERE owner_id = ? AND mailbox_id = ? AND message_id = ?"
    ).bind(owner, mailbox, messageId).run();
  } else {
    const time = typeof dueAt === "string" ? Date.parse(dueAt) : NaN;
    const now = Date.now();
    if (!Number.isFinite(time) || time <= now || time > now + 365 * 24 * 60 * 60_000) {
      return json({ error: "Choose a time in the next year" }, 400);
    }
    await env.DB.prepare(
      "INSERT INTO reminders (owner_id, mailbox_id, message_id, due_at) VALUES (?, ?, ?, ?) " +
      "ON CONFLICT (owner_id, mailbox_id, message_id) DO UPDATE SET due_at = excluded.due_at"
    ).bind(owner, mailbox, messageId, new Date(time).toISOString()).run();
  }
  return json({ reminders: await listReminders(env.DB, owner, mailbox) });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/reminders") {
      try {
        return await reminders(request, env, url);
      } catch {
        return json({ error: "Reminder storage unavailable" }, 503);
      }
    }
    if (url.pathname.startsWith("/api/")) return json({ error: "Unknown endpoint" }, 404);
    const path = url.pathname === "/" ? "/index.html" : url.pathname;
    if (!Object.hasOwn(assets, path)) return new Response("Not found", { status: 404 });
    if (!["GET", "HEAD"].includes(request.method)) return new Response("Method not allowed", { status: 405 });
    const { mediaType, data } = assets[path];
    const body = request.method === "HEAD" ? null : Uint8Array.from(atob(data), part => part.charCodeAt(0));
    return new Response(body, { headers: {
      "content-type": mediaType,
      "cache-control": "private, no-cache",
      "x-content-type-options": "nosniff",
    } });
  },
};
