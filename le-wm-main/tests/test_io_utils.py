import pytest

from experiments.io_utils import prepare_empty_chunk_dir, prepare_resumable_chunk_dir


def test_prepare_empty_chunk_dir_allows_empty_directory(tmp_path):
    chunk_dir = prepare_empty_chunk_dir(tmp_path / "chunks")

    assert chunk_dir.exists()


def test_prepare_empty_chunk_dir_rejects_existing_chunks(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_000000.npz").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_empty_chunk_dir(chunk_dir)


def test_prepare_resumable_chunk_dir_returns_next_index(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_000000.npz").write_bytes(b"a")
    (chunk_dir / "chunk_000001.npz").write_bytes(b"b")

    path, existing, next_idx = prepare_resumable_chunk_dir(chunk_dir, resume=True)

    assert path == chunk_dir
    assert [p.name for p in existing] == ["chunk_000000.npz", "chunk_000001.npz"]
    assert next_idx == 2


def test_prepare_resumable_chunk_dir_rejects_gaps(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_000000.npz").write_bytes(b"a")
    (chunk_dir / "chunk_000002.npz").write_bytes(b"c")

    with pytest.raises(ValueError, match="contiguous"):
        prepare_resumable_chunk_dir(chunk_dir, resume=True)
