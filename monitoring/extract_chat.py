"""Extract the human-readable conversation from a Claude Code session transcript.

KEEPS   user prose, assistant visible responses, and any compaction summary.
DROPS   tool_use / tool_result blocks (every Bash, Write, Edit, workflow call),
        thinking blocks, system reminders, IDE notices, slash-command plumbing,
        and the non-message entry types (attachment, ai-title, queue-operation,
        file-history-*, mode, last-prompt, atis-latch).
"""
import json
import re
import sys

SRC = sys.argv[1]
OUT = sys.argv[2]

# Wrapper tags that carry harness plumbing rather than anything the user wrote.
STRIP_BLOCKS = [
    r"<system-reminder>.*?</system-reminder>",
    r"<local-command-caveat>.*?</local-command-caveat>",
    r"<command-name>.*?</command-name>",
    r"<command-message>.*?</command-message>",
    r"<command-args>.*?</command-args>",
    r"<local-command-stdout>.*?</local-command-stdout>",
    r"<ide_opened_file>.*?</ide_opened_file>",
    r"<ide_selection>.*?</ide_selection>",
    r"<task-notification>.*?</task-notification>",
    r"<functions>.*?</functions>",
]


def clean(t):
    if not isinstance(t, str):
        return ""
    for pat in STRIP_BLOCKS:
        t = re.sub(pat, "", t, flags=re.DOTALL)
    # Collapse runs of blank lines left behind by the strips.
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def texts(msg):
    """Visible text blocks only."""
    out = []
    c = msg.get("content")
    if isinstance(c, str):
        out.append(c)
    elif isinstance(c, list):
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
    return out


turns = []
summaries = []
n_lines = 0

with open(SRC, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        n_lines += 1
        try:
            r = json.loads(line)
        except Exception:
            continue

        et = r.get("type")

        # Compaction summaries live outside the normal message flow.
        if et == "summary" or (et == "system" and "summary" in str(r.get("subtype", ""))):
            s = clean(json.dumps(r.get("summary") or r.get("content") or ""))
            if len(s) > 200:
                summaries.append(s)
            continue

        if et not in ("user", "assistant"):
            continue
        msg = r.get("message")
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        body = "\n\n".join(t for t in (clean(x) for x in texts(msg)) if t)
        if not body:
            continue
        # Drop residual harness echoes that survive as bare text.
        if body.startswith("Caveat: The messages below were generated"):
            continue
        turns.append((role, body))

# Merge consecutive same-role turns (streamed assistant messages arrive split).
merged = []
for role, body in turns:
    if merged and merged[-1][0] == role:
        merged[-1] = (role, merged[-1][1] + "\n\n" + body)
    else:
        merged.append((role, body))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("# Session transcript — conversation only\n\n")
    f.write("Extracted from the Claude Code session JSONL. Contains the user's messages and\n")
    f.write("the assistant's visible responses. **Excludes** every tool call and result\n")
    f.write("(Bash, Write, Edit, workflows), internal reasoning, system reminders and IDE\n")
    f.write("notices — so this is the discussion and the conclusions, not the mechanics.\n\n")
    f.write(f"Source: `{SRC}` ({n_lines:,} JSONL entries)\n\n")
    if summaries:
        f.write(f"Includes {len(summaries)} compaction summary block(s) from earlier in the "
                f"session, marked below.\n\n")
    f.write("---\n\n")

    for s in summaries:
        f.write("## ⟨earlier context — compaction summary⟩\n\n")
        f.write(s + "\n\n---\n\n")

    n_u = n_a = 0
    for role, body in merged:
        if role == "user":
            n_u += 1
            f.write(f"## ▸ User [{n_u}]\n\n{body}\n\n")
        else:
            n_a += 1
            f.write(f"### Claude [{n_a}]\n\n{body}\n\n---\n\n")

print(f"JSONL entries read : {n_lines:,}")
print(f"turns kept         : {len(merged)}  ({n_u} user, {n_a} assistant)")
print(f"compaction blocks  : {len(summaries)}")
