#!/usr/bin/env python3
"""Reload pages of the current session: recent, matching a query, or by id.

usage: cp_page_in.py [QUERY] [--id N] [--all] [--full] [--limit N]

--all searches every session in this project; --full prints whole contents;
default limit is 10 entries.
"""
import sys

import cpcommon as cp


def parse(argv):
    opts = {"id": None, "all": False, "full": False, "limit": 10}
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--id", "--limit"):
            i += 1
            opts[a[2:]] = int(argv[i])
        elif a == "--all":
            opts["all"] = True
        elif a == "--full":
            opts["full"] = True
        else:
            positional.append(a)
        i += 1
    return (positional[0] if positional else None), opts


def show(con, rows, full):
    tags = cp.page_tags(con, [r["id"] for r in rows])
    for r in rows:
        print(
            "page %d [%s] %s session=%s topic=%s tags=%s"
            % (
                r["id"],
                r["kind"],
                r["created_at"],
                r["session_key"],
                r["topic"] or "-",
                ",".join(tags.get(r["id"], [])) or "-",
            )
        )
        print("  " + (r["content"] if full else cp.excerpt(r["content"], "", 400)))


def main():
    query, opts = parse(sys.argv[1:])
    con = cp.connect()
    if opts["id"] is not None:
        rows = con.execute(
            "SELECT p.*, s.key AS session_key, t.name AS topic FROM pages p"
            " JOIN sessions s ON s.id = p.session_id"
            " LEFT JOIN topics t ON t.id = p.topic_id WHERE p.id = ?",
            (opts["id"],),
        ).fetchall()
        if not rows:
            sys.exit("no page %d" % opts["id"])
        show(con, rows, True)
        return
    sid, key, _ = cp.ensure_session(con)
    sql = (
        "SELECT p.*, s.key AS session_key, t.name AS topic FROM pages p"
        " JOIN sessions s ON s.id = p.session_id"
        " LEFT JOIN topics t ON t.id = p.topic_id"
    )
    where, params = [], []
    if not opts["all"]:
        where.append("p.session_id = ?")
        params.append(sid)
    if query:
        where.append("p.content LIKE ?")
        params.append("%" + query + "%")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY p.id DESC LIMIT ?"
    params.append(opts["limit"])
    rows = con.execute(sql, params).fetchall()
    scope = "all sessions" if opts["all"] else "session %d key=%s" % (sid, key)
    if not rows:
        print("no pages (%s)" % scope)
        return
    print("%d page(s), %s:" % (len(rows), scope))
    show(con, rows, opts["full"])


if __name__ == "__main__":
    main()
