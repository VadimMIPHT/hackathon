import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";

import { demoMessages, mailboxes } from "../dist/demo.js";

const mediaTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".woff2": "font/woff2",
};
async function assetsIn(directory) {
  const files = [];
  const entries = await readdir(join("dist", directory), { withFileTypes: true });
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== "server" && !entry.name.startsWith(".")) files.push(...await assetsIn(path));
    } else if (entry.isFile() && Object.hasOwn(mediaTypes, extname(path))) {
      files.push(["/" + path, { mediaType: mediaTypes[extname(path)], data: await readFile(join("dist", path), "base64") }]);
    }
  }
  return files;
}

const assets = Object.fromEntries(await assetsIn(""));
const demoMail = Object.fromEntries(demoMessages.map(mail => [mail.mailbox + "/" + mail.id, mail]));
const worker = await readFile("worker/index.js", "utf8");
await mkdir("dist/server", { recursive: true });
const source = [
  "const assets = " + JSON.stringify(assets) + ";",
  "const demoMail = " + JSON.stringify(demoMail) + ";",
  "const mailboxIds = " + JSON.stringify(mailboxes.map(mailbox => mailbox.id)) + ";",
  worker,
].join("\n");
await writeFile("dist/server/index.js", source);
