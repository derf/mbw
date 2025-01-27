/*
 * vim: ai ts=4 sts=4 sw=4 cinoptions=>4 expandtab
 */
#define _GNU_SOURCE

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <err.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#ifdef HAVE_AVX512
#include <stdint.h>
#include <immintrin.h>
#endif

#ifdef MULTITHREADED
#include <pthread.h>
#include <semaphore.h>
#endif

#ifdef NUMA
#include <numaif.h>
#include <numa.h>
#endif

/* how many runs to average by default */
#define DEFAULT_NR_LOOPS 40

/* default block size for test 2, in bytes */
#define DEFAULT_BLOCK_SIZE 262144

/* test types */
#define TEST_MEMCPY 0
#define TEST_PLAIN 1
#define TEST_MCBLOCK 2
#define TEST_AVX512 3
#define TEST_READ_PLAIN 4
#define TEST_WRITE_PLAIN 5
#define TEST_READ_AVX512 6
#define TEST_WRITE_AVX512 7
#define MAX_TESTS 8

/* version number */
#define VERSION "1.5+smaug"

/*
 * MBW memory bandwidth benchmark
 *
 * 2006, 2012 Andras.Horvath@gmail.com
 * 2013 j.m.slocum@gmail.com
 * (Special thanks to Stephen Pasich)
 *
 * http://github.com/raas/mbw
 *
 * compile with:
 *			gcc -O -o mbw mbw.c
 *
 * run with eg.:
 *
 *			./mbw 300
 *
 * or './mbw -h' for help
 *
 * watch out for swap usage (or turn off swap)
 */

#ifdef MULTITHREADED
unsigned long num_threads = 1;
volatile unsigned int done = 0;
pthread_t *threads;
sem_t start_sem, stop_sem, sync_sem;
#endif

long *arr_a = NULL;
long *arr_b = NULL; /* the two arrays to be copied from/to */
unsigned long long arr_size=0; /* array size (elements in array) */
unsigned int test_type;
/* fixed memcpy block size for -t2 */
unsigned long long block_size=DEFAULT_BLOCK_SIZE;

int sanity_check = 0;
long arr_a_sum = 0;
long *partial_sum;

#ifdef NUMA
void* mp_pages[1];
int mp_status[1];
int mp_nodes[1];
int numa_node_a = -1;
int numa_node_b = -1;
int numa_node_cpu = -1;
struct bitmask* bitmask_a = NULL;
struct bitmask* bitmask_b = NULL;
#endif

#ifdef HAVE_AVX512

/**
 * AVX512 implementation taken from
 * <https://lore.kernel.org/all/1453086314-30158-4-git-send-email-zhihong.wang@intel.com/>
 */

/**
 * Copy 16 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov16(uint8_t *dst, const uint8_t *src)
{
	__m128i xmm0;

	xmm0 = _mm_loadu_si128((const __m128i *)src);
	_mm_storeu_si128((__m128i *)dst, xmm0);
}

/**
 * Copy 32 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov32(uint8_t *dst, const uint8_t *src)
{
	__m256i ymm0;

	ymm0 = _mm256_loadu_si256((const __m256i *)src);
	_mm256_storeu_si256((__m256i *)dst, ymm0);
}

/**
 * Copy 64 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov64(uint8_t *dst, const uint8_t *src)
{
	__m512i zmm0;

	zmm0 = _mm512_loadu_si512((const void *)src);
	_mm512_storeu_si512((void *)dst, zmm0);
}

/**
 * Copy 128 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov128(uint8_t *dst, const uint8_t *src)
{
	rte_mov64(dst + 0 * 64, src + 0 * 64);
	rte_mov64(dst + 1 * 64, src + 1 * 64);
}

/**
 * Copy 256 bytes from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov256(uint8_t *dst, const uint8_t *src)
{
	rte_mov64(dst + 0 * 64, src + 0 * 64);
	rte_mov64(dst + 1 * 64, src + 1 * 64);
	rte_mov64(dst + 2 * 64, src + 2 * 64);
	rte_mov64(dst + 3 * 64, src + 3 * 64);
}

/**
 * Copy 128-byte blocks from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov128blocks(uint8_t *dst, const uint8_t *src, size_t n)
{
	__m512i zmm0, zmm1;

	while (n >= 128) {
		zmm0 = _mm512_loadu_si512((const void *)(src + 0 * 64));
		n -= 128;
		zmm1 = _mm512_loadu_si512((const void *)(src + 1 * 64));
		src = src + 128;
		_mm512_storeu_si512((void *)(dst + 0 * 64), zmm0);
		_mm512_storeu_si512((void *)(dst + 1 * 64), zmm1);
		dst = dst + 128;
	}
}

/**
 * Copy 512-byte blocks from one location to another,
 * locations should not overlap.
 */
static inline void
rte_mov512blocks(uint8_t *dst, const uint8_t *src, size_t n)
{
	__m512i zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
	const uint8_t *end = src + n;

	while (src < end) {
		zmm0 = _mm512_load_si512((const void *)(src + 0 * 64));
		zmm1 = _mm512_load_si512((const void *)(src + 1 * 64));
		zmm2 = _mm512_load_si512((const void *)(src + 2 * 64));
		zmm3 = _mm512_load_si512((const void *)(src + 3 * 64));
		zmm4 = _mm512_load_si512((const void *)(src + 4 * 64));
		zmm5 = _mm512_load_si512((const void *)(src + 5 * 64));
		zmm6 = _mm512_load_si512((const void *)(src + 6 * 64));
		zmm7 = _mm512_load_si512((const void *)(src + 7 * 64));
		_mm512_store_si512((void *)(dst + 0 * 64), zmm0);
		_mm512_store_si512((void *)(dst + 1 * 64), zmm1);
		_mm512_store_si512((void *)(dst + 2 * 64), zmm2);
		_mm512_store_si512((void *)(dst + 3 * 64), zmm3);
		_mm512_store_si512((void *)(dst + 4 * 64), zmm4);
		_mm512_store_si512((void *)(dst + 5 * 64), zmm5);
		_mm512_store_si512((void *)(dst + 6 * 64), zmm6);
		_mm512_store_si512((void *)(dst + 7 * 64), zmm7);
		src += 512;
		dst += 512;
	}
}

static inline void *
rte_memcpy(void *dst, const void *src, size_t n)
{
	uintptr_t dstu = (uintptr_t)dst;
	uintptr_t srcu = (uintptr_t)src;
	void *ret = dst;
	size_t dstofss;
	size_t bits;

	/**
	 * Copy less than 16 bytes
	 */
	if (n < 16) {
		if (n & 0x01) {
			*(uint8_t *)dstu = *(const uint8_t *)srcu;
			srcu = (uintptr_t)((const uint8_t *)srcu + 1);
			dstu = (uintptr_t)((uint8_t *)dstu + 1);
		}
		if (n & 0x02) {
			*(uint16_t *)dstu = *(const uint16_t *)srcu;
			srcu = (uintptr_t)((const uint16_t *)srcu + 1);
			dstu = (uintptr_t)((uint16_t *)dstu + 1);
		}
		if (n & 0x04) {
			*(uint32_t *)dstu = *(const uint32_t *)srcu;
			srcu = (uintptr_t)((const uint32_t *)srcu + 1);
			dstu = (uintptr_t)((uint32_t *)dstu + 1);
		}
		if (n & 0x08)
			*(uint64_t *)dstu = *(const uint64_t *)srcu;
		return ret;
	}

	/**
	 * Fast way when copy size doesn't exceed 512 bytes
	 */
	if (n <= 32) {
		rte_mov16((uint8_t *)dst, (const uint8_t *)src);
		rte_mov16((uint8_t *)dst - 16 + n,
				  (const uint8_t *)src - 16 + n);
		return ret;
	}
	if (n <= 64) {
		rte_mov32((uint8_t *)dst, (const uint8_t *)src);
		rte_mov32((uint8_t *)dst - 32 + n,
				  (const uint8_t *)src - 32 + n);
		return ret;
	}
	if (n <= 512) {
		if (n >= 256) {
			n -= 256;
			rte_mov256((uint8_t *)dst, (const uint8_t *)src);
			src = (const uint8_t *)src + 256;
			dst = (uint8_t *)dst + 256;
		}
		if (n >= 128) {
			n -= 128;
			rte_mov128((uint8_t *)dst, (const uint8_t *)src);
			src = (const uint8_t *)src + 128;
			dst = (uint8_t *)dst + 128;
		}
COPY_BLOCK_128_BACK63:
		if (n > 64) {
			rte_mov64((uint8_t *)dst, (const uint8_t *)src);
			rte_mov64((uint8_t *)dst - 64 + n,
					  (const uint8_t *)src - 64 + n);
			return ret;
		}
		if (n > 0)
			rte_mov64((uint8_t *)dst - 64 + n,
					  (const uint8_t *)src - 64 + n);
		return ret;
	}

	/**
	 * Make store aligned when copy size exceeds 512 bytes
	 */
	dstofss = ((uintptr_t)dst & 0x3F);
	if (dstofss > 0) {
		dstofss = 64 - dstofss;
		n -= dstofss;
		rte_mov64((uint8_t *)dst, (const uint8_t *)src);
		src = (const uint8_t *)src + dstofss;
		dst = (uint8_t *)dst + dstofss;
	}

	/**
	 * Copy 512-byte blocks.
	 * Use copy block function for better instruction order control,
	 * which is important when load is unaligned.
	 */
	rte_mov512blocks((uint8_t *)dst, (const uint8_t *)src, n);
	bits = n;
	n = n & 511;
	bits -= n;
	src = (const uint8_t *)src + bits;
	dst = (uint8_t *)dst + bits;

	/**
	 * Copy 128-byte blocks.
	 * Use copy block function for better instruction order control,
	 * which is important when load is unaligned.
	 */
	if (n >= 128) {
		rte_mov128blocks((uint8_t *)dst, (const uint8_t *)src, n);
		bits = n;
		n = n & 127;
		bits -= n;
		src = (const uint8_t *)src + bits;
		dst = (uint8_t *)dst + bits;
	}

	/**
	 * Copy whatever left
	 */
	goto COPY_BLOCK_128_BACK63;
}
#endif

void usage()
{
    printf("mbw memory benchmark v%s, https://github.com/raas/mbw\n", VERSION);
    printf("Usage: mbw [options] array_size_in_MiB\n");
    printf("Options:\n");
    printf("	-n: number of runs per test (0 to run forever)\n");
    printf("	-a: Don't display average\n");
    printf("	-C: enable sanity checks\n");
    printf("	-t%d: memcpy test\n", TEST_MEMCPY);
    printf("	-t%d: plain (b[i]=a[i] style) test\n", TEST_PLAIN);
    printf("	-t%d: memcpy test with fixed block size\n", TEST_MCBLOCK);
#ifdef HAVE_AVX512
    printf("	-t%d: AVX512 copy test\n", TEST_AVX512);
#endif
    printf("	-t%d: plain read test (sum)\n", TEST_READ_PLAIN);
    printf("	-t%d: plain write test (const fill)\n", TEST_WRITE_PLAIN);
#ifdef HAVE_AVX512
    printf("	-t%d: AVX512 read test (sum)\n", TEST_READ_AVX512);
    printf("	-t%d: AVX512 write test (const fill)\n", TEST_WRITE_AVX512);
#endif
    printf("	-b <size>: block size in bytes for -t2 (default: %d)\n", DEFAULT_BLOCK_SIZE);
    printf("	-q: quiet (print statistics only)\n");
#ifdef NUMA
    printf("	-a <node>: allocate source array on NUMA node\n");
    printf("	-b <node>: allocate target array on NUMA node\n");
    printf("	-c <node>: schedule task/threads on NUME node\n");
#endif
    printf("(will then use two arrays, watch out for swapping)\n");
    printf("'Bandwidth' is amount of data copied over the time this operation took.\n");
    printf("\nThe default is to run all tests available.\n");
}

/* ------------------------------------------------------ */

/* allocate a test array and fill it with data
 * so as to force Linux to _really_ allocate it */
long *make_array(long *sum)
{
    unsigned long long t;
    unsigned int long_size=sizeof(long);
    long *a;

#ifdef HAVE_AVX512
    a=aligned_alloc(64, arr_size * long_size);
#else
    a=calloc(arr_size, long_size);
#endif

    if(NULL==a) {
        perror("Error allocating memory");
        exit(1);
    }

    /* make sure both arrays are allocated, fill with pattern */
    for(t=0; t<arr_size; t++) {
        a[t]=0xaa;
    }
    if (sum != NULL) {
        *sum = 0;
        for(t=0; t<arr_size; t++) {
            *sum += 0xaa;
        }
    }
    return a;
}

#ifdef MULTITHREADED
void *thread_worker(void *arg)
{
    unsigned long thread_id = (unsigned long)arg;
    unsigned int long_size=sizeof(long);
    unsigned long long array_bytes=arr_size*long_size;
    unsigned long long t;
    unsigned long long const plain_start = thread_id * (arr_size / num_threads);
    unsigned long long const plain_stop = (thread_id + 1) * (arr_size / num_threads);

    while (!done) {
        if (sem_wait(&start_sem) != 0) {
            err(1, "sem_wait(start_sem)");
        }
        if (done) {
            return NULL;
        }
        if(test_type==TEST_MEMCPY) { /* memcpy test */
            memcpy(arr_b + (thread_id * (arr_size / num_threads)), arr_a + (thread_id * (arr_size / num_threads)), array_bytes / num_threads);
        } else if(test_type==TEST_MCBLOCK) { /* memcpy block test */
            char* src = (char*)(arr_a + (thread_id * (arr_size / num_threads)));
            char* dst = (char*)(arr_b + (thread_id * (arr_size / num_threads)));
            for (t=array_bytes / num_threads; t >= block_size; t-=block_size, src+=block_size){
                dst=(char *) memcpy(dst, src, block_size) + block_size;
            }
            if(t) {
                dst=(char *) memcpy(dst, src, t) + t;
            }
        } else if(test_type==TEST_PLAIN) { /* plain test */
            for(t=plain_start; t<plain_stop; t++) {
                arr_b[t]=arr_a[t];
            }
#ifdef HAVE_AVX512
        } else if(test_type==TEST_AVX512) {
            rte_memcpy(arr_b, arr_a, array_bytes);
#endif // HAVE_AVX512
        } else if(test_type==TEST_READ_PLAIN) {
            long tmp = 0;
            for(t=plain_start; t<plain_stop; t++) {
                tmp += arr_a[t];
            }
            if (sanity_check) {
                partial_sum[thread_id] = tmp;
            }
        } else if(test_type==TEST_WRITE_PLAIN) {
            long tmp = 1374181804651713298;
            for(t=plain_start; t<plain_stop; t++) {
                arr_b[t] = tmp;
            }
#ifdef HAVE_AVX512
        } else if(test_type==TEST_READ_AVX512) {
            __m512i zmm0 = _mm512_setzero_epi32();
            __m512i zmm1;
            uint8_t *src = (uint8_t*)(arr_a + plain_start);
            const uint8_t *end = (uint8_t*)(arr_a + plain_stop);
            long tmp = 0;
            while (src < end) {
                zmm1 = _mm512_load_si512((const void *)src);
                zmm0 = _mm512_add_epi64(zmm0, zmm1);
                src += 64;
            }
            tmp = (long)_mm512_reduce_add_epi64(zmm0);
            if (sanity_check) {
                assert((plain_start & 0x0000000000000007) == 0);
                assert((plain_stop & 0x0000000000000007) == 0);
                partial_sum[thread_id] = tmp;
            }
        } else if(test_type==TEST_WRITE_AVX512) {
            uint8_t *src = (uint8_t*)(arr_a + plain_start);
            uint8_t *dst = (uint8_t*)(arr_b + plain_start);
            const uint8_t *end = (uint8_t*)(arr_b + plain_stop);
            __m512i zmm0 = _mm512_load_si512(src);
            while (dst < end) {
                _mm512_store_si512((void*)(dst), zmm0);
                dst += 64;
            }
#endif // HAVE_AVX512
        }
        if (sem_post(&stop_sem) != 0) {
            err(1, "sem_post(stop_sem)");
        }
        if (sem_wait(&sync_sem) != 0) {
            err(1, "sem_wait(sync_sem)");
        }
    }
    return NULL;
}

void start_threads()
{
    for (unsigned int i = 0 ; i < num_threads; i++) {
        sem_post(&start_sem);
    }
}

void await_threads()
{
    for (unsigned int i = 0 ; i < num_threads; i++) {
        sem_wait(&stop_sem);
    }
}

void sync_threads()
{
    for (unsigned int i = 0 ; i < num_threads; i++) {
        sem_post(&sync_sem);
    }
}
#endif

/* actual benchmark */
/* arr_size: number of type 'long' elements in test arrays
 * long_size: sizeof(long) cached
 * test_type: 0=use memcpy, 1=use plain copy loop (whatever GCC thinks best)
 *
 * return value: elapsed time in seconds
 */
double worker()
{
    struct timespec starttime, endtime;
    double te;
    /* array size in bytes */

#ifdef MULTITHREADED
    clock_gettime(CLOCK_MONOTONIC, &starttime);
    start_threads();
    await_threads();
    clock_gettime(CLOCK_MONOTONIC, &endtime);
    sync_threads();
#else

    unsigned int long_size=sizeof(long);
    unsigned long long array_bytes=arr_size*long_size;
    unsigned long long t;
    if(test_type==TEST_MEMCPY) { /* memcpy test */
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        memcpy(arr_b, arr_a, array_bytes);
        clock_gettime(CLOCK_MONOTONIC, &endtime);
    } else if(test_type==TEST_MCBLOCK) { /* memcpy block test */
        char* src = (char*)arr_a;
        char* dst = (char*)arr_b;
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        for (t=array_bytes; t >= block_size; t-=block_size, src+=block_size){
            dst=(char *) memcpy(dst, src, block_size) + block_size;
        }
        if(t) {
            dst=(char *) memcpy(dst, src, t) + t;
        }
        clock_gettime(CLOCK_MONOTONIC, &endtime);
    } else if(test_type==TEST_PLAIN) { /* plain test */
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        for(t=0; t<arr_size; t++) {
            arr_b[t]=arr_a[t];
        }
        clock_gettime(CLOCK_MONOTONIC, &endtime);
#ifdef HAVE_AVX512
    } else if(test_type==TEST_AVX512) {
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        rte_memcpy(arr_b, arr_a, array_bytes);
        clock_gettime(CLOCK_MONOTONIC, &endtime);
#endif // HAVE_AVX512
    } else if(test_type==TEST_READ_PLAIN) {
        long tmp = 0;
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        for(t=0; t<arr_size; t++) {
            tmp += arr_a[t];
        }
        clock_gettime(CLOCK_MONOTONIC, &endtime);
        if (sanity_check) {
            assert(tmp == arr_a_sum);
        }
    } else if(test_type==TEST_WRITE_PLAIN) {
        long tmp = 0;
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        for(t=0; t<arr_size; t++) {
            arr_b[t] = ++tmp;
        }
        clock_gettime(CLOCK_MONOTONIC, &endtime);
#ifdef HAVE_AVX512
    } else if(test_type==TEST_READ_AVX512) {
        __m512i zmm0 = _mm512_setzero_epi32();
        __m512i zmm1;
        long tmp = 0;
        uint8_t *src = (uint8_t*)arr_a;
        const uint8_t *end = src + arr_size * sizeof(long);
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        while (src < end) {
            zmm1 = _mm512_load_si512((const void *)src);
            zmm0 = _mm512_add_epi64(zmm0, zmm1);
            src += 64;
        }
        clock_gettime(CLOCK_MONOTONIC, &endtime);
        tmp = (long)_mm512_reduce_add_epi64(zmm0);
        if (sanity_check) {
            if (tmp != arr_a_sum) {
                printf("expected: arr_a_sum == %12ld (%016lx)\n", arr_a_sum, arr_a_sum);
                printf("output:  reduce_add == %12ld (%016lx)\n", tmp, tmp);
            }
            assert(tmp == arr_a_sum);
        }
    } else if(test_type==TEST_WRITE_AVX512) {
        const uint8_t *src = (uint8_t*)arr_b;
        uint8_t *dst = (uint8_t*)arr_b;
        const uint8_t *end = dst + arr_size * sizeof(long);
        __m512i zmm0 = _mm512_load_si512(src);
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        while (dst < end) {
            _mm512_store_si512((void*)(dst), zmm0);
            dst += 64;
        }
        clock_gettime(CLOCK_MONOTONIC, &endtime);
#endif // HAVE_AVX512
    }
#endif // !MULTITHREADED

    te=((double)(endtime.tv_sec*1000000000-starttime.tv_sec*1000000000+endtime.tv_nsec-starttime.tv_nsec))/1000000000;

    return te;
}

/* ------------------------------------------------------ */

/* pretty print worker's output in human-readable terms */
/* te: elapsed time in seconds
 * mt: amount of transferred data in MiB
 * test_type: see 'worker' above
 *
 * return value: -
 */
void printout(double te, double mt)
{
    switch(test_type) {
        case TEST_MEMCPY:
            printf("e_method=MEMCPY ");
            break;
        case TEST_PLAIN:
            printf("e_method=PLAIN ");
            break;
        case TEST_MCBLOCK:
            printf("e_method=MCBLOCK ");
            break;
    }
    printf("| data_MiB=%f time_s=%f throughput_MiBps=%f\n", mt, te, mt/te);
    return;
}

/* ------------------------------------------------------ */

int main(int argc, char **argv)
{
    unsigned int long_size=0;
    double te, te_sum; /* time elapsed */
    unsigned int i;
    int o; /* getopt options */
    unsigned long testno;

    /* options */

    /* how many runs to average? */
    unsigned int nr_loops=DEFAULT_NR_LOOPS;
    /* what tests to run (-t x) */
    int tests[MAX_TESTS];
    double mt=0; /* MiBytes transferred == array size in MiB */
    int quiet=0; /* suppress extra messages */

    tests[0]=0;
    tests[1]=0;
    tests[2]=0;
    tests[3]=0;
    tests[4]=0;
    tests[5]=0;
    tests[6]=0;
    tests[7]=0;

    while((o=getopt(argc, argv, "ha:b:c:qn:N:t:B:C")) != EOF) {
        switch(o) {
            case 'h':
                usage();
                exit(1);
                break;
#ifdef NUMA
            case 'a': /* NUMA node */
                bitmask_a = numa_parse_nodestring(optarg);
                break;
            case 'b': /* NUMA node */
                bitmask_b = numa_parse_nodestring(optarg);
                break;
            case 'c': /* NUMA node */
                numa_node_cpu = strtoul(optarg, (char **)NULL, 10);
                break;
#endif
            case 'n': /* no. loops */
                nr_loops=strtoul(optarg, (char **)NULL, 10);
                break;
#ifdef MULTITHREADED
            case 'N': /* no. threads */
                num_threads=strtoul(optarg, (char **)NULL, 10);
                break;
#endif
            case 't': /* test to run */
                testno=strtoul(optarg, (char **)NULL, 10);
                if(testno>MAX_TESTS-1) {
                    printf("Error: test number must be between 0 and %d\n", MAX_TESTS-1);
                    exit(1);
                }
                tests[testno]=1;
                break;
            case 'B': /* block size in bytes*/
                block_size=strtoull(optarg, (char **)NULL, 10);
                if(0>=block_size) {
                    printf("Error: what block size do you mean?\n");
                    exit(1);
                }
                break;
            case 'C':
                sanity_check = 1;
                break;
            case 'q': /* quiet */
                quiet=1;
                break;
            default:
                break;
        }
    }

#ifndef HAVE_AVX512
    if (tests[TEST_AVX512]) {
        printf("Error: AVX512 memcpy requested, but this mbw build has been compiled without AVX512 support\n");
        exit(1);
    }
    if (tests[TEST_READ_AVX512]) {
        printf("Error: AVX512 read requested, but this mbw build has been compiled without AVX512 support\n");
        exit(1);
    }
    if (tests[TEST_WRITE_AVX512]) {
        printf("Error: AVX512 write requested, but this mbw build has been compiled without AVX512 support\n");
        exit(1);
    }
#endif

    /* default is to run most tests if no specific tests were requested */
    if( (tests[0]+tests[1]+tests[2]+tests[3]+tests[4]+tests[5]+tests[6]+tests[7]) == 0) {
        tests[0]=1;
        tests[1]=1;
        tests[2]=1;
        tests[4]=1;
        tests[5]=1;
        tests[6]=1;
        tests[7]=1;
    }

    if( nr_loops==0 && ((tests[0]+tests[1]+tests[2]+tests[3]+tests[4]+tests[5]+tests[6]+tests[7]) != 1) ) {
        printf("Error: nr_loops can be zero if only one test selected!\n");
        exit(1);
    }

    if(optind<argc) {
        mt=strtoul(argv[optind++], (char **)NULL, 10);
    } else {
        printf("Error: no array size given!\n");
        exit(1);
    }

    if(0>=mt) {
        printf("Error: array size wrong!\n");
        exit(1);
    }

    /* ------------------------------------------------------ */

    long_size=sizeof(long); /* the size of long on this platform */
    arr_size=1024*1024/long_size*mt; /* how many longs then in one array? */

    if(arr_size*long_size < block_size) {
        printf("Error: array size larger than block size (%llu bytes)!\n", block_size);
        exit(1);
    }

    if(!quiet) {
        printf("Long uses %d bytes. ", long_size);
        if(tests[2]) {
            printf("Using %lld bytes as blocks for memcpy block copy test.\n", block_size);
        }
    }

#ifdef NUMA
    struct bitmask *bitmask_all = numa_allocate_nodemask();
    numa_bitmask_setall(bitmask_all);
    if (bitmask_a) {
        numa_set_membind(bitmask_a);
        numa_free_nodemask(bitmask_a);
    }
#endif
    if (tests[TEST_MEMCPY]+tests[TEST_PLAIN]+tests[TEST_MCBLOCK]+tests[TEST_AVX512]+tests[TEST_READ_PLAIN]+tests[TEST_READ_AVX512]) {
        if (!quiet) {
            printf("Allocating %lld elements = %lld MiB of input memory.\n", arr_size, arr_size*long_size / 1024 / 1024);
        }
        arr_a=make_array(&arr_a_sum);
    }

#ifdef NUMA
    if (bitmask_b) {
        numa_set_membind(bitmask_b);
        numa_free_nodemask(bitmask_b);
    }
#endif
    if (tests[TEST_MEMCPY]+tests[TEST_PLAIN]+tests[TEST_MCBLOCK]+tests[TEST_AVX512]+tests[TEST_WRITE_PLAIN]+tests[TEST_WRITE_AVX512]) {
        if (!quiet) {
            printf("Allocating %lld elements = %lld MiB of output memory.\n", arr_size, arr_size*long_size / 1024 / 1024);
        }
        arr_b=make_array(NULL);
    }

#ifdef NUMA
    numa_set_membind(bitmask_all);
    numa_free_nodemask(bitmask_all);
#endif

#ifdef NUMA
    if (arr_a != NULL) {
        mp_pages[0] = arr_a;
        if (move_pages(0, 1, mp_pages, NULL, mp_status, 0) == -1) {
            perror("move_pages(arr_a)");
        }
        else if (mp_status[0] < 0) {
            printf("move_pages(arr_a) error: %d\n", mp_status[0]);
        }
        else {
            numa_node_a = mp_status[0];
        }
    }

    if (arr_b != NULL) {
        mp_pages[0] = arr_b;
        if (move_pages(0, 1, mp_pages, NULL, mp_status, 0) == -1) {
            perror("move_pages(arr_b)");
        }
        else if (mp_status[0] < 0) {
            printf("move_pages(arr_b) error: %d\n", mp_status[0]);
        }
        else {
            numa_node_b = mp_status[0];
        }
    }

    if (numa_node_cpu != -1) {
        if (numa_run_on_node(numa_node_cpu) == -1) {
            perror("numa_run_on_node");
            numa_node_cpu = -1;
        }
    }
#endif

    /* ------------------------------------------------------ */
    if(!quiet) {
        printf("Getting down to business... Doing %d runs per test.\n", nr_loops);
    }

#ifdef MULTITHREADED
    if (sem_init(&start_sem, 0, 0) != 0) {
        err(1, "sem_init");
    }
    if (sem_init(&stop_sem, 0, 0) != 0) {
        err(1, "sem_init");
    }
    if (sem_init(&sync_sem, 0, 0) != 0) {
        err(1, "sem_init");
    }
    threads = calloc(num_threads, sizeof(pthread_t));
    if (sanity_check) {
        partial_sum = calloc(num_threads, sizeof(long));
    }
    for (i=0; i < num_threads; i++) {
        if (sanity_check) {
            partial_sum[i] = 0;
        }
        if (pthread_create(&threads[i], NULL, thread_worker, (void*)(unsigned long)i) != 0) {
            err(1, "pthread_create");
        }
    }
#endif

    /* run all tests requested, the proper number of times */
    for(test_type=0; test_type<MAX_TESTS; test_type++) {
        te_sum=0;
        if(tests[test_type]) {
            for (i=0; nr_loops==0 || i<nr_loops; i++) {
                te=worker();
                te_sum+=te;
                if (test_type == TEST_MEMCPY) {
                    printf("[::] memcpy");
                } else if (test_type == TEST_PLAIN) {
                    printf("[::] copy");
                } else if (test_type == TEST_MCBLOCK) {
                    printf("[::] mcblock");
                } else if (test_type == TEST_AVX512) {
                    printf("[::] copy-avx512");
                } else if (test_type == TEST_READ_PLAIN) {
                    printf("[::] read");
                } else if (test_type == TEST_WRITE_PLAIN) {
                    printf("[::] write");
                } else if (test_type == TEST_READ_AVX512) {
                    printf("[::] read-avx512");
                } else if (test_type == TEST_WRITE_AVX512) {
                    printf("[::] write-avx512");
                }
                printf(" | block_size_B=%llu array_size_B=%llu ", block_size, arr_size*long_size);
#ifdef MULTITHREADED
                printf("n_threads=%ld ", num_threads);
#else
                printf("n_threads=1 ");
#endif
#ifdef NUMA
                printf("from_numa_node=%d to_numa_node=%d cpu_numa_node=%d numa_distance_ram_ram=%d numa_distance_ram_cpu=%d numa_distance_cpu_ram=%d ", numa_node_a, numa_node_b, numa_node_cpu, numa_distance(numa_node_a, numa_node_b), numa_distance(numa_node_a, numa_node_cpu), numa_distance(numa_node_cpu, numa_node_b));
#else
                printf("from_numa_node=X to_numa_node=X cpu_numa_node=X numa_distance_ram_ram=X numa_distance_ram_cpu=X numa_distance_cpu_ram=X ");
#endif
                printout(te, mt);
            }
        }
    }

#ifdef MULTITHREADED
    done = 1;
    start_threads();
    for (i=0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            err(1, "pthread_join");
        }
    }
    if (sanity_check && (tests[TEST_READ_PLAIN] || tests[TEST_READ_AVX512])) {
        long tmp = 0;
        for (i=0; i < num_threads; i++) {
            tmp += partial_sum[i];
        }
        if (tmp != arr_a_sum) {
            printf("expected:  arr_a_sum == %12ld (%016lx)\n", arr_a_sum, arr_a_sum);
            printf("output: sum(partial) == %12ld (%016lx)\n", tmp, tmp);
        }
        assert(tmp == arr_a_sum);
    }
#endif

    free(arr_a);
    free(arr_b);
    return 0;
}

