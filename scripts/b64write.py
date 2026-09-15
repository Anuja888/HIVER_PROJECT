#!/usr/bin/env python
"""Helper: write a file from a base64-encoded payload to avoid shell escaping issues."""
import base64, sys
path = sys.argv[1]
payload_b64 = sys.argv[2]
content = base64.b64decode(payload_b64).decode("utf-8")
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Wrote {len(content)} bytes to {path}")