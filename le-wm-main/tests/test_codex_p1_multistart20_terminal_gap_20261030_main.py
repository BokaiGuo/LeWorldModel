from experiments.spectral_reachability import codex_p1_multistart20_terminal_gap_20261030_main as subject


def test_fixed_images_and_censored_roster():
    entries = subject.inputs()
    assert len(entries) == 24
    assert [x["index"] for x in entries] == list(range(24))
    complete = [x for x in entries if x["status"] == "probe_completed"]
    stopped = [x for x in entries if x["status"] == "native_stopped"]
    assert len(complete) == 22 and len(stopped) == 2
    assert all(x["pixels"].shape == (224, 224, 3) for x in complete)
    assert all("pixels" not in x for x in stopped)
