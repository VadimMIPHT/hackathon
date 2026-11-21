import { readFile, mkdir } from "node:fs/promises";
import { createServer } from "node:http";
import { DatabaseSync } from "node:sqlite";

import worker from "../dist/server/index.js";

const port = Number(process.argv[2] || 8941);
await mkdir(".preview", { recursive: true });
const db = new DatabaseSync(".preview/reminders.sqlite3");
if (!db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'reminders'").get()) {
  db.exec(await readFile("drizzle/0000_reminders.sql", "utf8"));
}
const DB = {
  prepare(sql) {
    const statement = db.prepare(sql);
    return { bind: (...values) => ({
      all: () => ({ results: statement.all(...values) }),
      run: () => statement.run(...values),
    }) };
  },
};

createServer(async (incoming, outgoing) => {
  try {
    const origin = "http://127.0.0.1:" + port;
    const headers = new Headers(incoming.headers);
    headers.set("oai-authenticated-user-id", "local-preview");
    const options = { method: incoming.method, headers };
    if (!["GET", "HEAD"].includes(incoming.method)) Object.assign(options, { body: incoming, duplex: "half" });
    const response = await worker.fetch(new Request(origin + incoming.url, options), { DB });
    outgoing.writeHead(response.status, Object.fromEntries(response.headers));
    outgoing.end(Buffer.from(await response.arrayBuffer()));
  } catch {
    outgoing.writeHead(500).end("Preview request failed");
  }
}).listen(port, "127.0.0.1", () => process.stdout.write("http://127.0.0.1:" + port + "/\n"));
