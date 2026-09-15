#!/usr/bin/env python
"""Write a file from stdin to a path. Usage: python write_file.py PATH < content"""
import sys
path = sys.argv[1]
content = sys.stdin.read()
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Wrote {len(content)} bytes to {path}")