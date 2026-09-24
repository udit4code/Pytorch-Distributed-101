"""Print each worker's identity and a rank-local Python object."""

from phase0.distributed import cleanup_process_group, setup_process_group
from phase0.rank_info import get_rank_info


def main() -> None:
    try:
        setup_process_group()
        rank_info = get_rank_info()
        local_state = {"counter": 0}
        local_state["counter"] += rank_info.rank
        print(f"{rank_info} local_state={local_state}")
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
