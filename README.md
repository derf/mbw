# Memory BandWidth benchmark

MBW determines the read, write, and transfer bandwidth available to single- and multi-threaded userspace programs, optionally taking NUMA placement into account.
It is not tuned to extremes, instead relying on simple read/write operations and memcpy.

This is an extended version of the benchmark originally developepd by Andras Horvath et al.
Multi-threading, NUMA, and read/write support have been added by Birte Friesel.
The original README follows.

---

MBW determines the "copy" memory bandwidth available to userspace programs. Its simplistic approach models that of real applications. It is not tuned to extremes and it is not aware of hardware architecture, just like your average software package.

2006, 2012 Andras.Horvath atnospam gmail.com
2013 j.m.slocum atnospam gmail.com
2022 Willian.Zhang

http://github.com/raas/mbw
https://github.com/Willian-Zhang/mbw

'mbw 1000' to run copy memory test on all methods with 1 GiB memory.
'mbw -h' for help

watch out for swap usage (or turn off swap)
