#!/usr/bin/env python3
"""Project-wide recovery search across pages and messages of all sessions.

usage: cp_search.py QUERY [--offset N]

Returns compact references only — ids and short excerpts, never full
contents; --offset pages through further results (8 pages + 8 messages
per result page). Load detail with cp_around.py or cp_page_in.py --id N.
"""
import sys

import cpcommon as cp

PER_SECTION = 8


def parse(argv):
    offset = 0
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--offset":
            i += 1
            offset = int(argv[i])
        else:
            positional.append(a)
        i += 1
    if len(positional) != 1:
        sys.exit(__doc__.strip())
    return positional[0], offset


def main():
    needle, offset = parse(sys.argv[1:])
    like = "%" + needle + "%"
    con = cp.connect()
    pages = con.execute(
        "SELECT p.id, p.kind, p.content, p.created_at, s.key AS session_key,"
        " t.name AS topic FROM pages p"
        " JOIN sessions s ON s.id = p.session_id"
        " LEFT JOIN topics t ON t.id = p.topic_id"
        " WHERE p.content LIKE ? ORDER BY p.id DESC LIMIT ? OFFSET ?",
        (like, PER_SECTION + 1, offset),
    ).fetchall()
    messages = con.execute(
        "SELECT m.id, m.role, m.content, m.created_at, s.key AS session_key"
        " FROM messages m JOIN sessions s ON s.id = m.session_id"
        " WHERE m.content LIKE ? ORDER BY m.id DESC LIMIT ? OFFSET ?",
        (like, PER_SECTION + 1, offset),
    ).fetchall()
    if not pages and not messages:
        print("no matches for %r (offset %d)" % (needle, offset))
        return
    if pages:
        tags = cp.page_tags(con, [r["id"] for r in pages[:PER_SECTION]])
        print("pages:")
        for r in pages[:PER_SECTION]:
            print(
                "  page %d [%s] %s session=%s topic=%s tags=%s"
                % (r["id"], r["kind"], r["created_at"], r["session_key"], r["topic"] or "-",
                   ",".join(tags.get(r["id"], [])) or "-")
            )
            print("    " + cp.excerpt(r["content"], needle))
        if len(pages) > PER_SECTION:
            print("  more: cp_search.py %s --offset %d" % (needle, offset + PER_SECTION))
    if messages:
        print("messages:")
        for r in messages[:PER_SECTION]:
            print("  #%d %s %s session=%s" % (r["id"], r["role"], r["created_at"], r["session_key"]))
            print("    " + cp.excerpt(r["content"], needle))
        if len(messages) > PER_SECTION:
            print("  more: cp_search.py %s --offset %d" % (needle, offset + PER_SECTION))


if __name__ == "__main__":
    main()
