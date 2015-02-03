
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <bitset>
#include <tuple>
#include <complex>

#include <boost/aura/ext/fftw.hpp>
#include <boost/aura/misc/sequence.hpp>
#include <boost/aura/misc/benchmark.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/fft.hpp>

using namespace boost::aura;

typedef std::complex<float> cfloat;

// FIXME missing type (double, float) and r2c c2r

// configuration
std::vector<bounds> sizes;
std::vector<boost::aura::svec<std::size_t, 1> > batches;
std::size_t runtime;
std::size_t threads;
int devordinal;
bool bench_fftw;
const char * ops_tbl[] = { "fwdip", "invip", "fwdop", "invop" };
std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > ops;

#if 0

// we have multiple wait_for free functions, std::for_each
// can not decide which one should be used
void wait_for_feed(feed& f)
{
	wait_for(f);
}

// benchmark functions -----

void run_fwdip(std::vector<device_ptr<cfloat> > & mem1,
               std::vector<fft> & ffth, std::vector<feed> & feeds)
{
	for(std::size_t n = 0; n<feeds.size(); n++) {
		fft_forward(mem1[n], mem1[n], ffth[n], feeds[n]);
	}
	std::for_each(feeds.begin(), feeds.end(), &wait_for_feed);
}

void run_invip(std::vector<device_ptr<cfloat> > & mem1,
               std::vector<fft> & ffth, std::vector<feed> & feeds) {
	for(std::size_t n = 0; n<feeds.size(); n++) {
		fft_inverse(mem1[n], mem1[n], ffth[n], feeds[n]);
	}
	std::for_each(feeds.begin(), feeds.end(), &wait_for_feed);
}

void run_fwdop(std::vector<device_ptr<cfloat> > & mem1,
               std::vector<device_ptr<cfloat> > & mem2,
               std::vector<fft> & ffth,
               std::vector<feed> & feeds)
{
	for(std::size_t n = 0; n<feeds.size(); n++) {
		fft_forward(mem2[n], mem1[n], ffth[n], feeds[n]);
	}
	std::for_each(feeds.begin(), feeds.end(), &wait_for_feed);
}

void run_invop(std::vector<device_ptr<cfloat> > & mem1,
               std::vector<device_ptr<cfloat> > & mem2,
               std::vector<fft> & ffth,
               std::vector<feed> & feeds)
{
	for(std::size_t n = 0; n<feeds.size(); n++) {
		fft_inverse(mem2[n], mem1[n], ffth[n], feeds[n]);
	}
	std::for_each(feeds.begin(), feeds.end(), &wait_for_feed);
}

// -----

void print_results(const char * name, double min, double max,
                   double mean, double stdev, std::size_t runs,
                   const bounds & s,
                   const boost::aura::svec<std::size_t, 1> & batch)
{
	printf("%s %lux ", name, batch[0]);
	for(std::size_t i=0; i<s.size(); i++) {
		printf("%lu ", s[i]);
	}
	printf("min %f max %f mean %f stdev %f runs %lu\n",
	       min, max, mean, stdev, runs);
}

void run_tests()
{
	boost::aura::backend::initialize();
	boost::aura::backend::fft_initialize();

	// create devices, feeds
	std::vector<boost::aura::backend::device> devices;
	std::vector<boost::aura::backend::feed> feeds;
	// reserve to make sure the device objects are not moved
	devices.reserve(devordinals.size());
	feeds.reserve(devordinals.size());
	for(std::size_t i=0; i<devordinals.size(); i++) {
		devices.push_back(boost::aura::backend::device(devordinals[i]));
		feeds.push_back(boost::aura::backend::feed(devices[i]));
	}

	for(std::size_t b=0; b<batch.size(); b++) {
		for(std::size_t s=0; s<size.size(); s++) {
			// allocate device_ptr<cfloat> , make fft plan
			std::vector<boost::aura::backend::device_ptr<cfloat> > mem1;
			std::vector<boost::aura::backend::device_ptr<cfloat> > mem2;
			std::vector<boost::aura::backend::fft> ffth;
			for(std::size_t i=0; i<devices.size(); i++) {
				std::size_t msize = boost::aura::product(size[s]) * batch[b][0];
				mem1.push_back(boost::aura::backend::device_malloc<cfloat>(msize, devices[i]));
				mem2.push_back(boost::aura::backend::device_malloc<cfloat>(msize, devices[i]));
				ffth.push_back(boost::aura::backend::fft(devices[i], feeds[i], size[s],
				                boost::aura::backend::fft::type::c2c, batch[b][0]));
			}

			// benchmark result variables
			double min, max, mean, stdev;
			std::size_t runs;

			if(ops[0]) {
				run_fwdip(mem1, ffth, feeds);
				AURA_BENCHMARK(run_fwdip(mem1, ffth, feeds),
				               runtime, min, max, mean, stdev, runs);
				print_results(ops_tbl[0], min, max, mean, stdev, runs,
				              size[s], batch[b]);
			}
			if(ops[1]) {
				run_invip(mem1, ffth, feeds);
				AURA_BENCHMARK(run_invip(mem1, ffth, feeds),
				               runtime, min, max, mean, stdev, runs);
				print_results(ops_tbl[1], min, max, mean, stdev, runs,
				              size[s], batch[b]);
			}
			if(ops[2]) {
				run_fwdop(mem1, mem2, ffth, feeds);
				AURA_BENCHMARK(run_fwdop(mem1, mem2, ffth, feeds),
				               runtime, min, max, mean, stdev, runs);
				print_results(ops_tbl[2], min, max, mean, stdev, runs,
				              size[s], batch[b]);
			}
			if(ops[3]) {
				run_invop(mem1, mem2, ffth, feeds);
				AURA_BENCHMARK(run_invop(mem1, mem2, ffth, feeds),
				               runtime, min, max, mean, stdev, runs);
				print_results(ops_tbl[3], min, max, mean, stdev, runs,
				              size[s], batch[b]);
			}

			// free device_ptr<cfloat>
			for(std::size_t i=0; i<devices.size(); i++) {
				boost::aura::backend::device_free(mem1[i]);
				boost::aura::backend::device_free(mem2[i]);
			}
		}
	}
	boost::aura::backend::fft_terminate();
}

#endif


void run_acc_fwdip(device_array<cfloat>& v1, fft& fh, feed& f)
{
	fft_forward(v1, v1, fh, f);
	wait_for(f);
}

void run_acc_invip(device_array<cfloat>& v1, fft& fh, feed& f)
{
	fft_inverse(v1, v1, fh, f);
	wait_for(f);
}

void run_acc_fwdop(device_array<cfloat>& v1,
		device_array<cfloat>& v2, fft& fh, feed& f)
{
	fft_forward(v1, v2, fh, f);
	wait_for(f);
}

void run_acc_invop(device_array<cfloat>& v1,
		device_array<cfloat>& v2, fft& fh, feed& f)
{
	fft_inverse(v1, v2, fh, f);
	wait_for(f);
}


void run_bench_accelerator()
{
	initialize();
	fft_initialize();

	device d(devordinal);
	feed f(d);
	for(auto batch : batches) {
		for(auto size : sizes) {
			auto size2 = size;
			size2.push_back(batch);
			device_array<cfloat> v1(size2, d);
			device_array<cfloat> v2(size2, d);
			fft fh(d, f, size, fft::type::c2c, batch);
			if (!fh.valid()) {
				std::cout << "INVALID FFT PLAN batch " << 
					batch << " size " << 
					size << std::endl;
				continue;
			}
			benchmark_result bs;
			if(ops[0]) {
				run_acc_fwdip(v1, fh, f);
				AURA_BENCH(run_acc_fwdip(v1, fh, f),
				               runtime, bs);
				std::cout << ops_tbl[0] << " batch " << 
					batch << " size " << 
					size << " " << bs << std::endl;
			}
			if(ops[1]) {
				run_acc_invip(v1, fh, f);
				AURA_BENCH(run_acc_invip(v1, fh, f),
				               runtime, bs);
				std::cout << ops_tbl[1] << " batch " << 
					batch << " size " << 
					size << " " << bs << std::endl;
			}
			if(ops[2]) {
				run_acc_fwdop(v1, v2, fh, f);
				AURA_BENCH(run_acc_fwdop(v1, v2, fh, f),
				               runtime, bs);
				std::cout << ops_tbl[2] << " batch " << 
					batch << " size " << 
					size << " " << bs << std::endl;
			}
			if(ops[3]) {
				run_acc_invop(v1, v2, fh, f);
				AURA_BENCH(run_acc_invop(v1, v2, fh, f),
				               runtime, bs);
				std::cout << ops_tbl[3] << " batch " << 
					batch << " size " << 
					size << " " << bs << std::endl;
			}
		}
	}
	fft_terminate();

}

void run_bench_fftw()
{
	fftw::fft_initialize();
	fftwf_plan_with_nthreads(threads);
	for(auto batch : batches) {
		for(auto size : sizes) {
			std::complex<float>* v1 = (cfloat*)fftwf_malloc(sizeof(cfloat)*product(size));
			std::complex<float>* v2 = (cfloat*)fftwf_malloc(sizeof(cfloat)*product(size));
			fftw::fft fh(size, fftw::fft::type::c2c, 
					v1, v2, batch);
			benchmark_result bs;
			if(ops[0]) {
				fftw::fft_forward(v1, v1, fh);
				AURA_BENCH(fftw::fft_forward(
							v1, 
							v1, fh),
				               runtime, bs);
				std::cout << ops_tbl[0] << " batch " << 
					batch << " size " << 
					size << " " << bs << std::endl;
			}
			if(ops[1]) {
				fftw::fft_inverse(v1, v1, fh);
				AURA_BENCH(fftw::fft_forward(
							v1, 
							v1, fh),
				               runtime, bs);
				std::cout << ops_tbl[1] << " batch " << 
					batch << " size " << 
					size << " " << bs << std::endl;
			}
			if(ops[2]) {
				fftw::fft_forward(v1, v2, fh);
				AURA_BENCH(fftw::fft_forward(
							v1, 
							v2, fh),
				               runtime, bs);
				std::cout << ops_tbl[2] << " batch " << 
					batch << " size " << 
					size << " " << bs << std::endl;
			}
			if(ops[3]) {
				fftw::fft_inverse(v1, v2, fh);
				AURA_BENCH(fftw::fft_forward(
							v1, 
							v2, fh),
				               runtime, bs);
				std::cout << ops_tbl[3] << " batch " << 
					batch << " size " << 
					size << " " << bs << std::endl;
			}		
			fftwf_free(v1);
			fftwf_free(v2);
		}
	}
	fft_terminate();
}

void run_tests()
{
	if (bench_fftw) {
		run_bench_fftw();
	} else if (devordinal != -1) {
		run_bench_accelerator();
	}
}

// nifty code from 
// http://rosettacode.org/wiki/Determine_if_a_string_is_numeric#C.2B.2B
bool is_numeric(const std::string& input) {
    return std::all_of(input.begin(), input.end(), ::isdigit);
}

int main(int argc, char *argv[])
{

	// parse command line arguments:
	// the vector size -s, the batch size -b (both sequences)
	// the runtime per test -t in ms
	// and a list of device ordinals
	// and the options: fwdip, invip, fwdop, invop

	int opt;
	while ((opt = getopt(argc, argv, "s:b:t:d:p:")) != -1) {
		switch (opt) {
			case 's': 
			{
				printf("size: %s ", optarg);
				sizes = boost::aura::generate_sequence<int, 
				     AURA_SVEC_MAX_SIZE>(optarg);
				printf("(%lu) ", sizes.size());
				break;
			}
			case 't': 
			{
				runtime = atoi(optarg);
				printf("time: %lu ms ", runtime);
				// benchmark script expects us
				runtime *= 1000;
				break;
			}
			case 'p': 
			{
				threads = atoi(optarg);
				printf("threads: %lu ", threads);
				// benchmark script expects us
				break;
			}

			case 'b': 
			{
				printf("batch: %s ", optarg);
				batches = boost::aura::generate_sequence<
					std::size_t, 1>(optarg);
				printf("(%lu) ", batches.size());
				break;
			}
			case 'd': 
			{
				devordinal = -1;
				bench_fftw = false;
				
				if (is_numeric(std::string(optarg))) {
					devordinal = std::stoi(
							std::string(optarg));
				} else if (*optarg == 'f') {
					bench_fftw = true;
				}
				break;
			}
			default: {
				fprintf(stderr, "Usage: %s -s <vectorsize> -b "
						"<batchsize> -t <runtime> "
						"<operations>\n", argv[0]);
				exit(-1);
			}
		}
	}
	printf("options: ");
	for(unsigned int i=0; i<sizeof(ops_tbl)/sizeof(ops_tbl[0]); i++) {
		ops[i] = false;
		for(int j=optind; j<argc; j++) {
			if(NULL != strstr(argv[j], ops_tbl[i])) {
				printf("%s ", ops_tbl[i]);
				ops[i] = true;
			}
		}
	}
	std::cout << std::endl << "device: " << devordinal;
	std::cout << std::endl << "bench_fftw: " << bench_fftw;
	std::cout << std::endl << "epxected runtime: " << 
		batches.size()*sizes.size()*runtime*ops.count()/1000./1000.;
	std::cout << std::endl;

	run_tests();
}


