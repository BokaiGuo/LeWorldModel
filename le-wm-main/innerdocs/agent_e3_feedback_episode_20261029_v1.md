# E3 single-session sequencing primitive

Fixture callbacks demonstrate the physical ACK → current RGB encoding →
authenticated-history join → later plan → next packet sequence in one session.
Early stop prevents subsequent encode and plan. This is not a native physical
adapter, has no production CLI, no durable ledger/consumer and cannot establish
the authenticity of callbacks or observation. Caller must implement original
setup, time limits, source/model/precision guards, partials and cleanup. Partial
packets and early-stop native data require a richer real ABI before production.
No imagined latent is substituted; this is not an admitted result.
