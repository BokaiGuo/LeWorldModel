"""Consume the saved full-bank CUDA replay using the shared array verifier."""
from . import p1_full_bank_cuda_replay_20261030_main as full
from . import p1_diverse_cuda_replay_consume_20261030_main as consumer


if __name__ == "__main__":
    full.configure()
    consumer.main()
