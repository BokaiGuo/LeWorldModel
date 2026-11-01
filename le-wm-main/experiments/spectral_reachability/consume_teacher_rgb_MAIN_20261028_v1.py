"""Complete teacher RGB receipt, using the unmodified saved-only v2 validator."""
from experiments.spectral_reachability import agent_p1_teacher_library40_encode_20261028_v2 as source


def main():
    output = source.ROOT / 'results/agent_p1_teacher_library40_encode_20261028_v2_cuda_MAIN01'
    expected = 'cb08d580b3ee2e8d15d2cc83065a8dc68a74672df0e2b602479a836469cb15f3'
    receipt = source.ROOT / 'results/agent_p1_teacher_library40_encode_20261028_v2_consume_MAIN01.json'
    source.require(not receipt.exists() and not receipt.is_symlink(), 'NEW complete receipt')
    result = source.consume(output, expected)
    source.write_new(receipt, source.canonical(result))
    source.require(source.read(receipt) == result, 'all40 row receipt roundtrip')
    print(source.canonical(dict(receipt=str(receipt), sha256=source.sha(receipt),
                                status=result['status'], encoded=result['encoded'],
                                missing=result['missing'], full40_released=False,
                                P1_admission=False)).decode())


if __name__ == '__main__':
    main()
