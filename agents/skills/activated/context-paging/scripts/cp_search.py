#!/usr/bin/env python3
"""Project-wide recovery search across pages and messages of all sessions.

usage: cp_search.py QUERY
"""
import sys

import cpcommon as cp


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: cp_search.py QUERY")
    needle = sys.argv[1]
    like = "%" + needle + "%"
    con = cp.connect()
    pages = con.execute(
        "SELECT p.id, p.kind, p.content, p.created_at, s.key AS session_key,"
        " t.name AS topic FROM pages p"
        " JOIN sessions s ON s.id = p.session_id"
        " LEFT JOIN topics t ON t.id = p.topic_id"
        " WHERE p.content LIKE ? ORDER BY p.id DESC LIMIT 8",
        (like,),
    ).fetchall()
    messages = con.execute(
        "SELECT m.id, m.role, m.content, m.created_at, s.key AS session_key"
        " FROM messages m JOIN sessions s ON s.id = m.session_id"
        " WHERE m.content LIKE ? ORDER BY m.id DESC LIMIT 8",
        (like,),
    ).fetchall()
    if not pages and not messages:
        print("no matches for %r" % needle)
        return
    if pages:
        tags = cp.page_tags(con, [r["id"] for r in pages])
        print("pages:")
        for r in pages:
            print(
                "  page %d [%s] %s session=%s topic=%s tags=%s"
                % (r["id"], r["kind"], r["created_at"], r["session_key"], r["topic"] or "-",
                   ",".join(tags.get(r["id"], [])) or "-")
            )
            print("    " + cp.excerpt(r["content"], needle))
    if messages:
        print("messages:")
        for r in messages:
            print("  #%d %s %s session=%s" % (r["id"], r["role"], r["created_at"], r["session_key"]))
            print("    " + cp.excerpt(r["content"], needle))


if __name__ == "__main__":
    main()
