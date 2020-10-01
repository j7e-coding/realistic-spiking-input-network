#!/usr/bin/env python

import src.network as nw
import src.logging as log


log.ex.add_config("default.json")
log.ex.add_source_file("src/network.py")
log.ex.add_source_file("src/logging.py")
log.ex.add_source_file("run.py")


@log.ex.automain
def main():
    profile = False
    if profile:
        raise NotImplementedError(
            "TODO: Implement using the Profile method runcall")
        import cProfile
        profile_name = "results/profile.txt"
        result = cProfile.run(
            "import src.network as nw; nw.simulate()",
            profile_name,
        )
        import pstats
        from pstats import SortKey
        p = pstats.Stats(profile_name)
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(40)
        return result
    else:
        return nw.simulate()

#
# if __name__ == "__main__":
#     main()
