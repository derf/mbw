# Memory BandWidth benchmark

MBW determines the read, write, and copy bandwidth available to single- and multi-threaded userspace programs, optionally taking NUMA placement into account.
It can either utilize simple for loops and memcpy, or run (experimental!) AVX512 read / write / copy benchmarks.

This is an extended version of the benchmark originally developepd by Andras Horvath et al., available at <http://github.com/raas/mbw> / <https://github.com/Willian-Zhang/mbw>.
Multi-threading, NUMA, and read/write support were not present in the original versions.

## References

Mirrors of the MBW repository are available at the following locations.

 * [ESS](https://ess.cs.uos.de/git/software/smaug/mbw)
 * [Finalrewind](https://git.finalrewind.org/derf/mbw)
 * [GitHub](https://github.com/derf/mbw)
