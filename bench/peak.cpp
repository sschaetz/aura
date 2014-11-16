// run peak benchmarks:
//
// * single performance with multiply-add kernel
// * double performance with multiply-add kernel
// * ondevice throughput (copy)
// * ondevice throughput (scale)
// * ondevice throughput (sum) 
// * ondevice throughput (triad) 
// * bus throughput (host to device) 
// * bus throughput (device to host) 


#include <iostream>
#include <bitset>
#include <algorithm>
#include <vector>
#include <boost/aura/backend.hpp>
#include <boost/aura/misc/sequence.hpp>
#include <boost/aura/misc/benchmark.hpp>

const char * ops_tbl[] = { "sflop", "dflop", "devcopy", "devscale", 
  "devadd", "devtriad", "tphtd", "tpdth" };

using namespace boost::aura;
using namespace boost::aura::backend;

const char * kernel_file = "bench/peak.cc"; 

inline void run_host_to_device(feed & f, device_ptr<float> dst, 
  std::vector<float> & src) 
{
	copy(dst, &src[0], src.size()*sizeof(float), f); 
	wait_for(f);
}

inline void run_device_to_host(feed & f, std::vector<float> & dst, 
  device_ptr<float> src) 
{
	copy(&dst[0], src, dst.size()*sizeof(float), f); 
	wait_for(f);
}

template <typename T>
inline void run_kernel(feed & f, kernel & k, 
  const mesh& m, const bundle& b, device_ptr<T> m1) 
{
	invoke(k, m, b, args(m1.get()), f);
	wait_for(f); 
}

inline void run_kernel(feed & f, kernel & k, 
  const mesh& m, const bundle& b, device_ptr<float> m1, device_ptr<float> m2) 
{
	invoke(k, m, b, args(m1.get(), m2.get()), f);
	wait_for(f); 
}

inline void run_kernel(feed & f, kernel & k, 
  const mesh & m, const bundle & b, device_ptr<float> m1, 
  device_ptr<float> m2, float s) 
{
	invoke(k, m, b, args(m1.get(), m2.get(), s), f);
	wait_for(f); 
}

inline void run_kernel(feed & f, kernel & k, 
  const mesh & m, const bundle & b, device_ptr<float> m1, 
  device_ptr<float> m2, 
  device_ptr<float> m3) 
{
	invoke(k, m, b, args(m1.get(), m2.get(), m3.get()), f);
	wait_for(f); 
}

inline void run_kernel(feed & f, kernel & k, 
  const mesh & m, const bundle & b, device_ptr<float> m1, 
  device_ptr<float> m2, 
  device_ptr<float> m3, float s) 
{
	invoke(k, m, b, args(m1.get(), m2.get(), m3.get(), s), f);
	wait_for(f); 
}

inline void print_results(const char* name, std::size_t vsize, 
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& mesh,
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& bundle,
	double min, double max, double mean, double stdev,
	int runs, double runtime
	)
{
	std::cout << name << " (" << vsize << ") mesh (" << 
		mesh << ") bundle (" << bundle << ") min " << min << 
		" max " << max << " mean " << mean << " stdev " << stdev << 
		" runs " << runs << " runtime " << runtime << std::endl;
}

/// validate flop test
#define PEAK_FLOP_MADD		\
	r0 = r1*r8+r0;		\
	r1 = r15*r9+r2;		\
	r2 = r14*r10+r4;	\
	r3 = r13*r11+r6;	\
	r4 = r12*r12+r8;	\
	r5 = r11*r13+r10;       \
	r6 = r10*r14+r12;	\
	r7 = r9*r15+r14;	\
	r8 = r7*r0+r1;		\
	r9 = r8*r1+r3;		\
	r10 = r6*r2+r5;		\
	r11 = r5*r3+r7;		\
	r12 = r4*r4+r9;		\
	r13 = r3*r5+r11;	\
	r14 = r2*r6+r13;	\
	r15 = r0*r7+r15;	\
	/**/

template <typename T>
T get_flop_test_result(int id) {
	T r0 = 0.0000001 * id;
	T r1 = 0.0000001 * id;
	T r2 = 0.0000002 * id;
	T r3 = 0.0000003 * id;
	T r4 = 0.0000004 * id;
	T r5 = 0.0000005 * id;
	T r6 = 0.0000006 * id;
	T r7 = 0.0000007 * id;
	T r8 = 0.0000008 * id;
	T r9 = 0.0000009 * id;
	T r10 = 0.0000010 * id;
	T r11 = 0.0000011 * id;
	T r12 = 0.0000012 * id;
	T r13 = 0.0000013 * id;
	T r14 = 0.0000014 * id;
	T r15 = 0.0000015 * id; 
	for (int i=0; i<64; i++) {
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
		PEAK_FLOP_MADD;
	}
	r0 += r1 + r2 + r3 + r4 + r5 + r6 + r7 + 
	r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
	return r0;
}

template <typename T>
struct float_comparator{
	float_comparator(std::size_t precission) : 
		tolerance_(std::numeric_limits<T>::epsilon() * precission) 
	{}
	
	bool operator()(const T& val1, const T& val2) {
		return std::abs(val1 - val2) < tolerance_;
	}
	
	const std::size_t tolerance_;
};

/// run flop test
template <typename T>
void run_test_flop(feed& f, kernel& kern, 
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& mesh,
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& bundle,
	std::size_t vsize, device_ptr<T> mem, 
	std::size_t runtime)
{
	double min, max, mean, stdev;
	std::size_t runs;
	run_kernel(f, kern, mesh, bundle, mem);
	std::vector<T> result_device(vsize);
	std::vector<T> result_host(vsize);
	copy(&result_device[0], mem, vsize, f);
	wait_for(f);
	// quite imprecise
	float_comparator<T> fc(10000000);
	for (unsigned int i=0; i<vsize; i++) {
		result_host[i] = get_flop_test_result<T>(i);
	}
	if (!std::equal(result_device.begin(), result_device.end(), 
				result_host.begin(), fc)) {
		printf("float test failed!\n");
		return;
	}
	AURA_BENCHMARK(run_kernel(f, kern, 
				mesh, bundle, mem), 
			runtime, min, max, mean, stdev, runs);
	print_results(ops_tbl[0], vsize, mesh, bundle, min,
			  max, mean, stdev, runs, runtime);
}

void run_test_copy(feed& f, kernel& kern, 
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& mesh,
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& bundle,
	std::size_t vsize, 
	device_ptr<float> mem1, device_ptr<float> mem2,
	std::size_t runtime) 
{
	double min, max, mean, stdev;
	std::size_t runs;
	std::vector<float> result_device(vsize, 0);
	copy(mem1, &result_device[0], vsize, f);
	run_kernel(f, kern, mesh, bundle, mem1, mem2);
	std::vector<float> result_host(vsize, 17);
	copy(&result_device[0], mem1, vsize, f);
	wait_for(f);
	float_comparator<float> fc(10000000);
	if (!std::equal(result_device.begin(), result_device.end(), 
				result_host.begin(), fc)) {
		printf("copy test failed!\n");
		return;
	}
	
	AURA_BENCHMARK(run_kernel(f, kern, mesh, bundle, 
		mem1, mem2), runtime, min, max, mean, stdev, runs);
	print_results(ops_tbl[2], vsize, mesh, bundle, min,
		max, mean, stdev, runs, runtime);

}

void run_test_scale(feed& f, kernel& kern, 
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& mesh,
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& bundle,
	std::size_t vsize, 
	device_ptr<float> mem1, device_ptr<float> mem2, 
	float s,
	std::size_t runtime) 
{
	double min, max, mean, stdev;
	std::size_t runs;
	std::vector<float> result_device(vsize, 0);
	copy(mem1, &result_device[0], vsize, f);
	run_kernel(f, kern, mesh, bundle, mem1, mem2, s);
	std::vector<float> result_host(vsize, 17*s);
	copy(&result_device[0], mem1, vsize, f);
	wait_for(f);
	float_comparator<float> fc(10000000);
	if (!std::equal(result_device.begin(), result_device.end(), 
				result_host.begin(), fc)) {
		printf("scale test failed!\n");
		return;
	}
	
	AURA_BENCHMARK(run_kernel(f, kern, mesh, bundle, 
		mem1, mem2, s), runtime, min, max, mean, stdev, runs);
	print_results(ops_tbl[3], vsize, mesh, bundle, min,
		max, mean, stdev, runs, runtime);

}

void run_test_add(feed& f, kernel& kern, 
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& mesh,
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& bundle,
	std::size_t vsize, 
	device_ptr<float> mem1, device_ptr<float> mem2,
	device_ptr<float> mem3,
	std::size_t runtime) 
{
	double min, max, mean, stdev;
	std::size_t runs;
	std::vector<float> result_device(vsize, 0);
	copy(mem1, &result_device[0], vsize, f);
	run_kernel(f, kern, mesh, bundle, mem1, mem2, mem3);
	std::vector<float> result_host(vsize, 17.+17.);
	copy(&result_device[0], mem1, vsize, f);
	wait_for(f);
	float_comparator<float> fc(10000000);
	if (!std::equal(result_device.begin(), result_device.end(), 
				result_host.begin(), fc)) {
		printf("add test failed!\n");
		return;
	}
	AURA_BENCHMARK(run_kernel(f, kern, mesh, bundle, 
		mem1, mem2, mem3), runtime, min, max, mean, stdev, runs);
	print_results(ops_tbl[4], vsize, mesh, bundle, min,
		max, mean, stdev, runs, runtime);

}

void run_test_triad(feed& f, kernel& kern, 
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& mesh,
	const svec<std::size_t, AURA_MAX_MESH_DIMS>& bundle,
	std::size_t vsize, 
	device_ptr<float> mem1, device_ptr<float> mem2,
	device_ptr<float> mem3, float s,
	std::size_t runtime) 
{
	double min, max, mean, stdev;
	std::size_t runs;
	std::vector<float> result_device(vsize, 0);
	copy(mem1, &result_device[0], vsize, f);
	run_kernel(f, kern, mesh, bundle, mem1, mem2, mem3, s);
	std::vector<float> result_host(vsize, 17.+s*17.);
	copy(&result_device[0], mem1, vsize, f);
	wait_for(f);
	float_comparator<float> fc(10000000);
	if (!std::equal(result_device.begin(), result_device.end(), 
				result_host.begin(), fc)) {
		printf("triad test failed!\n");
		return;
	}
	
	AURA_BENCHMARK(run_kernel(f, kern, mesh, bundle, 
		mem1, mem2, mem3, s), runtime, min, max, mean, stdev, runs);
	print_results(ops_tbl[5], vsize, mesh, bundle, min,
		max, mean, stdev, runs, runtime);

}

inline void run_tests(
		std::vector<svec<std::size_t, AURA_MAX_MESH_DIMS> > & meshes,
		std::vector<svec<std::size_t, AURA_MAX_BUNDLE_DIMS> > & bundles,
		std::vector<svec<std::size_t, 1> > & sizes,
		std::vector<svec<std::size_t, 1> > & dev_ordinals, 
		std::size_t runtime,
		std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > & ops) 
{

	// benchmark result variables
	double min, max, mean, stdev;
	std::size_t runs;

	device d(dev_ordinals[0][0]);
	feed f(d);

	module m = create_module_from_file(kernel_file, d, 
	AURA_BACKEND_COMPILE_FLAGS);

	if (ops[0] || ops[1]) {
		kernel ksflop = create_kernel(m, "peak_flop_single"); 
		kernel kdflop = create_kernel(m, "peak_flop_double"); 
		for(std::size_t m=0; m<meshes.size(); m++) {
			std::size_t vsize = product(meshes[m]);
			
			device_ptr<float> mems = 
				device_malloc<float>(vsize, d);
			device_ptr<double> memd = 
				device_malloc<double>(vsize, d);
			
			for (std::size_t b=0; b<bundles.size(); b++) {
				if(ops[0]) { // sflop
					run_test_flop<float>(f, ksflop, 
							meshes[m], bundles[b], 
							vsize, mems, runtime);
				}
				if(ops[1]) { // dflop
					run_test_flop<double>(f, kdflop, 
							meshes[m], bundles[b], 
							vsize, memd, runtime);
				}
			}
			device_free(mems);
			device_free(memd);
		}
	}

	if (ops[2] || ops[3] || ops[4] || ops[5]) {
		kernel kcopy = create_kernel(m, "peak_copy"); 
		kernel kscale = create_kernel(m, "peak_scale"); 
		kernel kadd = create_kernel(m, "peak_add"); 
		kernel ktriad = create_kernel(m, "peak_triad"); 
		for (std::size_t m=0; m<meshes.size(); m++) {
			std::size_t vsize = product(meshes[m])*64;
			device_ptr<float> mem1 = device_malloc<float>(vsize, d);
			device_ptr<float> mem2 = device_malloc<float>(vsize, d);
			device_ptr<float> mem3 = device_malloc<float>(vsize, d);
			std::vector<float> hostmem(vsize, 17);
			copy(mem1, &hostmem[0], vsize, f);
			copy(mem2, &hostmem[0], vsize, f);
			copy(mem3, &hostmem[0], vsize, f);
			wait_for(f);
			for(std::size_t b=0; b<bundles.size(); b++) {
				if (ops[2]) { // copy 
					run_test_copy(f, kcopy, meshes[m], 
							bundles[b], vsize, 
							mem1, mem2, runtime);
				}
				if(ops[3]) { // scale 
					run_test_scale(f, kscale, meshes[m], 
							bundles[b], vsize, 
							mem1, mem2, 42., 
							runtime);
				}
				if(ops[4]) { // add 
					run_test_add(f, kadd, meshes[m], 
							bundles[b], vsize, 
							mem1, mem2, mem3, 
							runtime);
				}
				if(ops[5]) { // triad 
					run_test_triad(f, ktriad, meshes[m], 
							bundles[b], vsize, 
							mem1, mem2, mem3, 42., 
							runtime);
				}
			}
			device_free(mem1);
			device_free(mem2);
			device_free(mem3);
		}
	}


	if(ops[6] || ops[7]) {
		for(std::size_t s=0; s<sizes.size(); s++) {
			std::vector<float> a1(sizes[s][0], 42.);
			std::vector<float> a2(sizes[s][0]);
			device_ptr<float> m = 
				device_malloc<float>(sizes[s][0], d);

			if (ops[6]) { // tphtd
				run_host_to_device(f, m, a1);
				copy(&a2[0], m, a2.size(), f);
				wait_for(f);

				if (!std::equal(a1.begin(), a1.end(), 
							a2.begin())) {
					printf("%s failed!\n", ops_tbl[6]);
					return;
				} 

				AURA_BENCHMARK(run_host_to_device(f, m, a1), 
						runtime, min, max, 
						mean, stdev, runs);
				std::cout << ops_tbl[6] << " (" << sizes[s] << 
					") min " << min << " max " << max << 
					" mean " << mean << " stdev " << 
					stdev << " runs " << runs << 
					" runtime " << runtime << std::endl;
			}

			if (ops[7]) { // tpdth
				std::fill(a2.begin(), a2.end(), 0.0);
				copy(m, &a1[0], a1.size(), f);
				run_device_to_host(f, a2, m);

				if (!std::equal(a1.begin(), a1.end(), 
							a2.begin())) {
					printf("%s failed!\n", ops_tbl[7]);
					return;
				} 

				AURA_BENCHMARK(run_device_to_host(f, a2, m), 
						runtime, min, max, mean, stdev, 
						runs);
				std::cout << ops_tbl[6] << " (" << sizes[s] << 
					") min " << min << 
					" max " << max << " mean " << mean << 
					" stdev " << stdev << 
					" runs " << runs << " runtime " << 
					runtime << std::endl;
			}
			device_free(m);
		}
	}
}


int main(int argc, char *argv[]) 
{

	initialize();

	// parse command line arguments:
	// -m mesh sizes (sequence, max rank 3)
	// -b bundle sizes (sequence, max rank 3)
	// -s size of vector used for test (sequence, max rank 1)
	// -d device (single value or pair for device to device)
	// -t time (time per benchmark in ms)

	// config params
	std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > ops;
	std::vector<svec<std::size_t, AURA_MAX_MESH_DIMS> > meshes;
	std::vector<svec<std::size_t, AURA_MAX_BUNDLE_DIMS> > bundles;
	std::vector<svec<std::size_t, 1> > sizes;
	std::vector<svec<std::size_t, 1> > dev_ordinals;
	std::size_t runtime = 0;

	// parse config
	int opt;
	while ((opt = getopt(argc, argv, "m:b:s:d:t:")) != -1) {
		switch (opt) {
			case 'm': {
				printf("mesh: %s ", optarg);
				meshes = boost::aura::generate_sequence<std::size_t,
				       AURA_MAX_MESH_DIMS>(optarg);
				break;
			}
			case 'b': {
				printf("bundle: %s ", optarg);
				bundles = boost::aura::generate_sequence<std::size_t, 
					AURA_MAX_BUNDLE_DIMS>(optarg);
				break;
			}
			case 's': {
				printf("size: %s ", optarg);
				sizes = boost::aura::generate_sequence<std::size_t, 1>(
						optarg);
				break;
			}
			case 'd': {
				printf("device %s ", optarg);
				dev_ordinals = boost::aura::generate_sequence<
					std::size_t, 1> (optarg);
				break;
			}
			case 't': {
				runtime = atoi(optarg);
				printf("time: %lu ms ", runtime);
				// benchmark script expects us
				runtime *= 1000; 
				break;
			}
			default: {
				fprintf(stderr, "Usage: %s -m <meshsize>"
						" -b <bundlesize>"
						" -s <vectorsize>"
						" -d <device ordinal (1 or 2)>"
						" -t <runtime (ms)> "
						"<operations>\n", argv[0]);
				fprintf(stderr, "Operations are: ");
				for (unsigned int i=0; i<sizeof(ops_tbl)/
						sizeof(ops_tbl[0]); i++) {
					fprintf(stderr, "%s ", ops_tbl[i]);
				}
				fprintf(stderr, "\n");
				exit(-1);
			}
	}
	}
	printf("operations: ");
	for (unsigned int i=0; i<sizeof(ops_tbl)/sizeof(ops_tbl[0]); i++) {
		ops[i] = false;
		for (int j=optind; j<argc; j++) {
			if (NULL != strstr(argv[j], ops_tbl[i])) {
				printf("%s ", ops_tbl[i]);
				ops[i] = true;
			}
		}
	}
	printf("\n");

	// output info about selected device  
	{
		printf("selected device(s): ");
		for (std::size_t i=0; i<dev_ordinals.size(); i++) {
			device d(dev_ordinals[i][0]);
			device_info di = device_get_info(d);
			print_device_info(di); 
		}
	}

	run_tests(meshes, bundles, sizes, dev_ordinals, runtime, ops);
}
