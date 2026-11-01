"""Persist strict saved-only native JEPA action-word receipt."""
import json
import os

from experiments.spectral_reachability import agent_p5_native_action_words_20261028_v2 as producer


def main():
    manifest = producer.ROOT/'results/agent_p5_native_action_words_20261028_v2_prepare_MAIN01/manifest.json'
    report = producer.ROOT/'results/agent_p5_native_action_words_20261028_v2_cuda_MAIN01/report.json'
    result = producer.consume(producer.read_json(manifest,
        '1515cb228c6ab0a4fe826074324fd566781b06812a1e5163c41652492518a905'), report,
        'cdbccbdf894d8d6e7c078e02f7078bd8cd26e802c9a00d59421ad0e4b0883bbd')
    destination = producer.ROOT/'results/consume_p5_words_MAIN_20261028_v1_MAIN01.json'
    with destination.open('xb') as stream:
        body = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        if stream.write(body) != len(body):
            raise OSError('short saved receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(destination), sha256=producer.sha(destination), result=result)))


if __name__ == '__main__':
    main()
