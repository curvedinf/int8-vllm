#!/usr/bin/env python3
"""Chronological project-wide recovery search across pages and messages.

usage: cp_search.py QUERY [--since DUR] [--offset N] [--role R,R]

Oldest-first references only — ids and short excerpts, never full
contents. --since bounds the lookup window: 30m, 1h (the default), 7d,
or any; --offset pages through further results (8 pages + 8 messages per
result page); --role restricts messages to the given roles (user,
thinking, assistant, tool_call, tool) and suppresses pages. The matches
line reports per-role and per-kind counts so follow-up searches can
target one type. Load detail with cp_around.py or cp_page_in.py --id N.
"""
import sys

import cpcommon as cp

PER_SECTION = 8


def time_cond(modifier, col):
    if modifier is None:
        return "", []
    return " AND %s >= datetime('now', ?)" % col, [modifier]


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


def counts(con, sql, params):
    return con.execute(sql, params).fetchall()


def tally(rows):
    return ", ".join("%d %s" % (r[1], r[0]) for r in rows)


def older(con, sql, params, modifier):
    if modifier is None:
        return 0
    return con.execute(sql, params).fetchone()[0]


def main():
    argv, modifier, label = cp.parse_since(sys.argv[1:])
    argv, roles = cp.parse_role(argv)
    needle, offset = parse(argv)
    like = "%" + needle + "%"
    role_cond, role_params = "", []
    if roles:
        role_cond = " AND m.role IN (%s)" % ",".join("?" * len(roles))
        role_params = roles
    con = cp.connect()
    print(cp.filters_line(scope="project", query=needle, since=label,
                          offset=offset or None,
                          role=",".join(roles) if roles else None))

    tc, tp = time_cond(modifier, "p.created_at")
    pages = [] if roles else con.execute(
        "SELECT p.id, p.kind, p.content, p.created_at, s.key AS session_key,"
        " t.name AS topic FROM pages p"
        " JOIN sessions s ON s.id = p.session_id"
        " LEFT JOIN topics t ON t.id = p.topic_id"
        " WHERE p.content LIKE ?" + tc +
        " ORDER BY p.id ASC LIMIT ? OFFSET ?",
        [like] + tp + [PER_SECTION + 1, offset],
    ).fetchall()
    mc, mp = time_cond(modifier, "m.created_at")
    messages = con.execute(
        "SELECT m.id, m.role, m.content, m.created_at, s.key AS session_key"
        " FROM messages m JOIN sessions s ON s.id = m.session_id"
        " WHERE m.content LIKE ?" + mc + role_cond +
        " ORDER BY m.id ASC LIMIT ? OFFSET ?",
        [like] + mp + role_params + [PER_SECTION + 1, offset],
    ).fetchall()

    op = 0 if roles else older(
        con, "SELECT COUNT(*) FROM pages WHERE content LIKE ?", [like], modifier)
    om = older(
        con, "SELECT COUNT(*) FROM messages m WHERE m.content LIKE ?" + role_cond,
        [like] + role_params, modifier)
    if not pages and not messages:
        line = "no matches for %r" % needle
        if op or om:
            line += "; %d page(s), %d message(s) match beyond the window (--since any)" % (op, om)
        print(line)
        return

    msg_counts = counts(
        con,
        "SELECT m.role, COUNT(*) FROM messages m WHERE m.content LIKE ?" + mc + role_cond +
        " GROUP BY m.role ORDER BY 2 DESC, 1",
        [like] + mp + role_params)
    msg_total = sum(r[1] for r in msg_counts)
    parts = []
    if msg_total:
        parts.append("%d message(s): %s" % (msg_total, tally(msg_counts)))
    if not roles:
        page_counts = counts(
            con,
            "SELECT p.kind, COUNT(*) FROM pages p WHERE p.content LIKE ?" + tc +
            " GROUP BY p.kind ORDER BY 2 DESC, 1",
            [like] + tp)
        page_total = sum(r[1] for r in page_counts)
        if page_total:
            parts.append("%d page(s): %s" % (page_total, tally(page_counts)))
    print("[matches: %s]" % "; ".join(parts))

    if pages or op:
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
            print("  more: cp_search.py %s --since %s --offset %d"
                  % (needle, label, offset + PER_SECTION))
        if op:
            print("  (%d page(s) older beyond the window: --since any)" % op)
    if messages or om:
        print("messages:")
        for r in messages[:PER_SECTION]:
            print("  #%d %s %s session=%s" % (r["id"], r["role"], r["created_at"], r["session_key"]))
            print("    " + cp.excerpt(r["content"], needle))
        if len(messages) > PER_SECTION:
            print("  more: cp_search.py %s --since %s%s --offset %d"
                  % (needle, label, " --role " + ",".join(roles) if roles else "",
                     offset + PER_SECTION))
        if om:
            print("  (%d message(s) older beyond the window: --since any)" % om)


if __name__ == "__main__":
    main()
