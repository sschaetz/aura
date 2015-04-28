#define BOOST_TEST_MODULE math.norm2 

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/blas/norm2.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/math/complex.hpp>
#include <fpu_control.h>

using namespace boost::aura;
using namespace boost::aura::math;

// support functions to calculate the norm2 on the CPU in different ways... ------------------

template <typename T>
void cpu_recu_sum(T *vec, std::size_t N){
    // Calculates the vector sum recursively on vector sub-blocks of size N to imitate gpu fiber bundles
    if (N == 1){
        return;
    }

    std::div_t divresult = div(N,2);

    if (divresult.rem)
        throw "Vector size has to be a power of 2";

    std::size_t N2 = (std::size_t) divresult.quot;

    for(std::size_t i = 0; i< N2; i++){
        vec[i] += vec[i+N2];
    }
    cpu_recu_sum(vec, N2);

}
template <typename T>
float cpu_recu_norm2(std::vector<T>& vec){
    // Calculates the norm2 recursively on vector sub-blocks to imitate gpu fiber bundles

    // setup bundle size
    const int PSEUDO_BUNDLE_SIZE = 128;
    //const int PSEUDO_BUNDLE_SIZE = (int) vec.size();

    // get the vector size
    std::size_t N = vec.size();

    // check if the vector size is a multiple of the bundle size
    std::div_t divresult = div(N,PSEUDO_BUNDLE_SIZE);
    if (divresult.rem)
       throw "Vector size has to be a multiple of the bundle size";

    // calculate the squared magnitude of each vector element
    for(auto &it : vec){
        it = pow( fabs( it ), 2);
    }

    // calculate the vector sum
    float res = 0;
    for(int i = 0; i< (int)N; i+= PSEUDO_BUNDLE_SIZE)
    {
        cpu_recu_sum(&vec[i],PSEUDO_BUNDLE_SIZE);
        res += (float)abs(vec[i]);
    }

    // calculate the squareroot
    res = sqrt(res);

    return(res);
}

// simple l2 norm that accumulates all elements and thus potentially creates high rounding errors
template <typename T>
float cpu_norm2(std::vector<T> vec){
    float res = 0.;
    for (auto it : vec){
        res += pow(abs(it),2.);
    }
    res = std::sqrt(res);
    return(res);
}


std::default_random_engine generator(1);
std::uniform_real_distribution<float> distribution(-1e3,1e3);
auto random_float = [&](){ return distribution(generator);};
auto random_cfloat = [&](){ return cfloat(random_float(),random_float());};




// norm2_float
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(norm2_float)
{
    // size and value of the testvector
    int numel = 16*1024;

    // initialize
    initialize();
    int num = device_get_count();
    BOOST_REQUIRE(num > 0);
    device d(0);
    feed f(d);

    // allocate testvectors on cpu and gpu
    std::vector<float> input(numel);
    std::generate(input.begin(), input.end(), random_float);

    std::vector<float> output(1, 0.0);
    device_array<float> device_input(numel, d);
    device_array<float> device_output(1, d);
    copy(input, device_input, f);

    // call norm2 -----
    math::norm2(device_input, device_output, f);
    // ------

    // transfer result
    copy(device_output, output, f);
    wait_for(f);

    // calculate the norm on the cpu
    float cpu_res = cpu_recu_norm2(input);  // calculate with bundle simulation

    // show result
    std::cout << "gpu_res = " << output[0] << std::endl;
    std::cout << "cpu_res = " << cpu_res << std::endl;

    BOOST_CHECK(fabs( cpu_res - output[0]) < std::numeric_limits<float>::epsilon() * fabs( cpu_res + output[0]));
    // BOOST_CHECK(fabs( cpu_res - output[0]) < std::numeric_limits<float>::min());


}



// norm2_cfloat
// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(norm2_cfloat)
{
    int numel = 16*1024;


    // initialize
    initialize();
    int num = device_get_count();
    BOOST_REQUIRE(num > 0);
    device d(0);
    feed f(d);

    // allocate testvectors on cpu and gpu
    std::vector<cfloat> input(numel);
    std::generate(input.begin(), input.end(), random_cfloat);

    std::vector<float> output(1, 0.0);
    device_array<cfloat> device_input(numel, d);
    device_array<float> device_output(1, d);
    copy(input, device_input, f);

    // call norm2 -----
    math::norm2(device_input, device_output, f);
    // ------

    // transfer result
    copy(device_output, output, f);
    wait_for(f);

    // try to enforce single precision operations
//    fpu_control_t fpu_oldcw, fpu_cw;
//    _FPU_GETCW(fpu_oldcw); //get old cw
//    fpu_cw = (fpu_oldcw & ~_FPU_EXTENDED & ~_FPU_DOUBLE & ~_FPU_SINGLE) | _FPU_SINGLE;
//    _FPU_SETCW(fpu_cw);
    // it might also be a good idea to set the compiler option -ffloat-store (if available)

    // calculate the norm on the cpu
    float cpu_res = cpu_recu_norm2(input);  // calculate with bundle simulation

    // show result
    std::cout << "gpu_res = " << output[0] << std::endl;
    std::cout << "cpu_res = " << cpu_res << std::endl;
    // std::cout << "val = " << val << std::endl;

    BOOST_CHECK(fabs( cpu_res - output[0]) < std::numeric_limits<float>::epsilon() * fabs( cpu_res + output[0]));
	// BOOST_CHECK(fabs( cpu_res - output[0]) < std::numeric_limits<float>::min());
}
