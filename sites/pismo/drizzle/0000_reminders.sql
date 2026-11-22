CREATE TABLE reminders (
    owner_id TEXT NOT NULL,
    mailbox_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    due_at TEXT NOT NULL,
    PRIMARY KEY (owner_id, mailbox_id, message_id)
);
