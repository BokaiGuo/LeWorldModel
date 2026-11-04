# P3 new-venv binding: CPU preparation

This is a new-environment identity record, not historical runtime equivalence
or permission to launch GPU work.

- Binding: `le-wm-main/results/p3_venv_migration_binding_20261102_codexexclusive_PREP01.json`
- Binding SHA256: `0569d430d8fdf798580318144d9fd6534623dc19dd812ef8a4b94db4ef81a930`
- External-SHA reconstruction: `new_environment_binding_reconstructed`.
- Project `.venv` uses Python 3.12.3 and identifies torch 2.12.0,
  torchvision 0.27.0, transformers 5.9.0, stable-worldmodel 0.1.0,
  stable-pretraining 0.1.7 and numpy 2.4.6. Key CPU imports succeeded with
  `CUDA_VISIBLE_DEVICES=''`.
- The checkpoint config and weights byte hashes match their rows in the old
  host prepare. The current compatibility loader also strictly loaded the
  checkpoint on CPU, froze parameters and produced tensor SHA256
  `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`;
  192 checkpoint keys were remapped. This matches the old recorded tensor
  hash, but is not a CUDA numerical replay or environment equivalence claim.
- One preliminary CPU check called the old `frozen` helper and failed because
  that helper explicitly requires CUDA. A separate CPU-specific eval/frozen
  check then succeeded; no CUDA check was bypassed for a host run.

The binding hashes selected files, not the complete Python/native dependency
closure. CPU tensor verification above is a separate console check and is not
represented as part of the binding JSON. The original launcher still calls
the old source-closure reconstruction and remains blocked. Task/bank
reconstruction, execution binding, GPU UUID preflight, saved-consumer
integration and numerical parity are not established by this record.
