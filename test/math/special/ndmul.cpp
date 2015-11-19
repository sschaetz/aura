#define BOOST_TEST_MODULE math.ndmul

#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/special/ndmul.hpp>
#include <boost/aura/device_array.hpp>

#include <iomanip>

using namespace boost::aura;

// ndmul_float
// _____________________________________________________________________________

typedef std::complex<float> cfloat;

std::default_random_engine generator(1);
std::uniform_real_distribution<float> distribution(-1e5,1e5);
auto random_float = [](){ return distribution(generator);};
auto random_cfloat = [](){ return cfloat(random_float(),random_float());};

std::vector<int> sizes = {1,2,3,4,5,128, 512};//,256,512,1024, 1024*1024,1024*1024*16};
int maxdims = 16;


BOOST_AUTO_TEST_CASE(ndmul_float)
{
    initialize();
    int num = device_get_count();
    BOOST_REQUIRE(num > 0);
    device d(0);


    for (auto x : sizes) {
        for (int j = 1; j < maxdims; j++) {
            int y = x*x*j;
            std::vector<float> input1(x);
            std::vector<float> input2(y);
            std::vector<float> output(y, 0.0);

            std::generate(input1.begin(), input1.end(), random_float);
            std::generate(input2.begin(), input2.end(), random_float);

            device_array<float> device_input1(x, d);
            device_array<float> device_input2(y, d);
            device_array<float> device_output(y, d);

            feed f(d);
            copy(input1, device_input1, f);
            copy(input2, device_input2, f);
            copy(output, device_output, f);

            math::ndmul(device_input1, device_input2, device_output, f);

            copy(device_output, output, f);
            wait_for(f);

            // simulate result on host
            std::vector<float> result(y);
            for (int i =0; i < y; i++) {
                result[i] 	= input1[i%input1.size()]*input2[i];
            }
            /* std::transform(input1.begin(), input1.end(),
                        input2.begin(), input2.begin(),
                        [](const float& a, const float& b) {
                            return a*b;
                        }
                    ); */
            BOOST_CHECK(
                std::equal(output.begin(), output.end(),
                    result.begin())
                );
        }
    }
}


BOOST_AUTO_TEST_CASE(ndmul_cfloat)
{
    initialize();
    int num = device_get_count();
    BOOST_REQUIRE(num > 0);
    device d(0);

    for (auto x : sizes) {
        for (int j = 1; j < maxdims; j++) {
            int y = x*x*j;
            std::vector<cfloat> input1(x);
            std::vector<cfloat> input2(y);
            std::vector<cfloat> output(y, cfloat(0.0,0.0));

            std::vector<bool> check(y,true);
            std::vector<bool> check1(y,true);

            std::generate(input1.begin(), input1.end(), random_cfloat);
            std::generate(input2.begin(), input2.end(), random_cfloat);

            device_array<cfloat> device_input1(x, d);
            device_array<cfloat> device_input2(y, d);
            device_array<cfloat> device_output(y, d);

            feed f(d);
            copy(input1, device_input1, f);
            copy(input2, device_input2, f);
            copy(output, device_output, f);

            math::ndmul(device_input1, device_input2, device_output, f);

            copy(device_output, output, f);
            wait_for(f);

            // simulate result on host
            std::vector<cfloat> result(y);
            for (int i =0; i < y; i++) {
                result[i] 	= input1[i%input1.size()]*input2[i];
            }
            /* std::transform(input1.begin(), input1.end(),
                        input2.begin(), input2.begin(),
                        [](const cfloat& a, const cfloat& b) {
                            return a*b;
                        }
                    ); */

            std::transform(output.begin(), output.end(),
                        result.begin(), check.begin(),
                        [](const cfloat& a, const cfloat& b) {
                            return abs(a-b) <= std::numeric_limits<float>::epsilon() * abs(a+b);   //see c++ documentation about epsilon (numeric limits)
                        }
                    );

            BOOST_CHECK(
                std::equal(check.begin(), check.end(),
                    check1.begin())
                );
        }

    }

}

// float and cfloat
BOOST_AUTO_TEST_CASE(ndmul_mixedfloat)
{
    initialize();
    int num = device_get_count();
    BOOST_REQUIRE(num > 0);
    device d(0);

    for (auto x : sizes) {
        for (int j = 1; j < maxdims; j++) {

            int y = x*x*j;
            std::vector<float> input1(x);
            std::vector<cfloat> input2(y);
            std::vector<cfloat> output(y, cfloat(0.0,0.0));

            std::vector<bool> check(y,true);
            std::vector<bool> check1(y,true);

            std::generate(input1.begin(), input1.end(), random_float);
            std::generate(input2.begin(), input2.end(), random_cfloat);

            device_array<float> device_input1(x, d);
            device_array<cfloat> device_input2(y, d);
            device_array<cfloat> device_output(y, d);

            feed f(d);
            copy(input1, device_input1, f);
            copy(input2, device_input2, f);
            copy(output, device_output, f);

            math::ndmul(device_input1, device_input2, device_output, f);

            copy(device_output, output, f);
            wait_for(f);

            // simulate result on host
            std::vector<cfloat> result(y);
            for (int i =0; i < y; i++) {
                result[i] 	= input1[i%input1.size()]*input2[i];
            }
            /* std::transform(input1.begin(), input1.end(),
                        input2.begin(), input2.begin(),
                        [](const cfloat& a, const cfloat& b) {
                            return a*b;
                        }
                    ); */

            std::transform(output.begin(), output.end(),
                        result.begin(), check.begin(),
                        [](const cfloat& a, const cfloat& b) {
                            return abs(a-b) <= std::numeric_limits<float>::epsilon() * abs(a+b);   //see c++ documentation about epsilon (numeric limits)
                        }
                    );

            BOOST_CHECK(
                std::equal(check.begin(), check.end(),
                    check1.begin())
                );
        }

    }

}
