#!/usr/bin/env python3
"""List the current session's messages, newest last.

usage: cp_message_list.py [QUERY] [LIMIT]   (default limit 20)
"""
import sys

import cpcommon as cp


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else None
    limit = 20
    if len(sys.argv) > 2:
        try:
            limit = int(sys.argv[2])
        except ValueError:
            sys.exit("usage: cp_message_list.py [QUERY] [LIMIT]")
    con = cp.connect()
    sid, key, _ = cp.ensure_session(con)
    sql = "SELECT id, role, content, created_at FROM messages WHERE session_id = ?"
    params = [sid]
    if query:
        sql += " AND content LIKE ?"
        params.append("%" + query + "%")
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = con.execute(sql, params).fetchall()
    print("session %d key=%s" % (sid, key))
    for r in reversed(rows):
        print("#%d %s %s: %s" % (r["id"], r["role"], r["created_at"], cp.excerpt(r["content"], query or "")))


if __name__ == "__main__":
    main()
