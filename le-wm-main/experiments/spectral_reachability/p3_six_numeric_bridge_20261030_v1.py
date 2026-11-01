"""Six-slot bounded chunk transport; transport integrity is not provenance."""
import base64
import hashlib

from experiments.spectral_reachability import p3_matched_order_feedback_numeric_bridge_codex_20261029 as old

need = old.need
MAX_EVENT = old.MAX_EVENT
CHUNK = old.CHUNK
RESOURCE = "p3_six_numeric_chunk_v1"


class Bridge:
    def __init__(self, store, slot):
        need(type(slot) is int and 0 <= slot < 6, "fixed six-slot identity")
        self.store, self.slot, self.count = store, slot, 0

    def event(self, payload):
        raw = old.storage.base.raw(payload)
        need(0 < len(raw) <= MAX_EVENT, "bounded native event")
        digest = hashlib.sha256(raw).hexdigest()
        parts = [raw[i:i + CHUNK] for i in range(0, len(raw), CHUNK)]
        for index, part in enumerate(parts):
            self.store.event("numerical", "record", RESOURCE, self.store.index,
                             dict(slot=self.slot, sequence=self.count, index=index,
                                  total=len(parts), bytes=len(raw), sha256=digest,
                                  data=base64.b64encode(part).decode("ascii")))
        self.count += 1


def reconstruct(events, slot):
    need(type(slot) is int and 0 <= slot < 6, "fixed six-slot identity")
    chunks = [e["detail"] for e in events if e["origin"] == "numerical" and
              e["kind"] == "record" and e["resource"] == RESOURCE and e["detail"].get("slot") == slot]
    output, i = [], 0
    while i < len(chunks):
        first = chunks[i]
        n, size, digest = first["total"], first["bytes"], first["sha256"]
        need(type(n) is int and n > 0 and type(size) is int and 0 < size <= MAX_EVENT and
             n == (size + CHUNK - 1) // CHUNK and i + n <= len(chunks), "complete chunks")
        parts = []
        for j in range(n):
            d = chunks[i + j]
            need(set(d) == {"slot", "sequence", "index", "total", "bytes", "sha256", "data"} and
                 d["slot"] == slot and type(d["sequence"]) is int and d["sequence"] == len(output) and
                 type(d["index"]) is int and d["index"] == j and d["total"] == n and
                 d["bytes"] == size and d["sha256"] == digest, "chunk identity and order")
            try:
                part = base64.b64decode(d["data"], validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid chunk encoding") from exc
            need(len(part) == (CHUNK if j < n - 1 else size - (n - 1) * CHUNK), "chunk length")
            parts.append(part)
        raw = b"".join(parts)
        need(hashlib.sha256(raw).hexdigest() == digest, "chunk hash")
        value = old.storage.base.decode(raw)
        need(type(value) is dict, "event dictionary")
        output.append(value)
        i += n
    return output
