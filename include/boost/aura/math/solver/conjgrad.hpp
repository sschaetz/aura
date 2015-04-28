#ifndef CONJGRAD_HPP
#define CONJGRAD_HPP

//#define CG_DEBUG

#include <vector>
//#include <boost/aura/device_array.hpp>
#include <boost/aura/math/math.hpp>

namespace boost
{
namespace aura
{

/* ======================================================
 (linear) conjugate gradient algorithm as in
 en.wikipedia.org/wiki/Conjugate_gradient_method

 It solves the linear matrix equation
 x = arg_min || A x - b ||_2^2

 If cg_tol is -1 (default) it will be automatically
 set to: 1e-2 * || A(x) -b ||_2^2
 ======================================================*/
template <typename T, typename funOp_3args>
std::tuple<int, float> conjgrad (         // return values are the number of final iterations and residuum
                funOp_3args  A,           // System matrix function A(x, result, feed)
                const device_array<T> &b, // Vector of measurements
                device_array<T> &x,       // Parameter vector to be optimized (should contain the initial guess at the beginning)
                feed &f, device &d,

                // (optional) CG stopping criteria
                const int cg_maxit = 500, double cg_tol = -1)
{
    // iteration counter and residuum
    int its = 0;
    float residuum;

    // get data dimension
    bounds bB = b.get_bounds();

    // create device arrays for temporary results
    device_array<T> r(bB, d);
    device_array<T> p(bB, d);
    device_array<T> temp(bB, d);
    device_array<T> Ap(bB, d);
    device_array<T> alpha(1, d);

    // scalar variables for the previos and the updated residuum
    device_array<T> resOld(1, d);
    device_array<T> resNew(1, d);

    // r = b - A*x
    A(x,r,f);
    math::sub(b,r,r,f); // r = b - r

    // p = r
    copy(r,p,f);

    // rsold = r' * r
    math::dot(r,r,resOld,f);

    // check if the residuum is already nearly zero
    residuum = fabs(resOld.get_value(f));
    if ( residuum < std::numeric_limits<float>::epsilon()){
        std::cout << "Residuum < epsilon!!\n";
        return(std::make_pair(its,residuum));
    }

    // if cg_tol is set to -1, calculate it from the initial residuum
    if (cg_tol == -1)
    {
        //cg_tol = 1e-4 * sqrt(residuum);
        cg_tol = 1e-2 * residuum;
    }

    // used the squared cg_tolerance to avoid squareroots in the inner loop
    cg_tol = pow(cg_tol,2);


    for (its =0; its < cg_maxit; its++){
        // Ap = A * p
        A(p,Ap,f);

        // alpha = resOld / (p' * Ap);
        math::dot(p, Ap, alpha, f);
        math::div(resOld,alpha,alpha,f);

        // x = x + alpha* p
        math::mul(p,alpha,temp,f);
        math::add(temp,x,x,f);

        // r = r - alpha * Ap;
        math::mul(Ap,alpha,temp,f);
        math::sub(r,temp,r,f);

        // resNew = r'*r
        math::dot(r,r,resNew,f);

        #ifdef CG_DEBUG
        std::cout << "cg_res " << resNew.get_value(f) << std::endl;
        #endif

        residuum = abs(resNew.get_value(0,f));
        if (residuum < cg_tol)
        {
            #ifdef CG_DEBUG
            std::cout << "cg tolerance reached after " << its << " iterations.\n";
            #endif
            return(std::make_pair(its+1,residuum));
        }

        // p = r + rsnew / rsold * p
        math::div(resNew,resOld,resOld,f);
        math::mul(p,resOld,p,f);
        math::add(p,r,p,f);

        // resOld = resNew;
        copy(resNew,resOld,f);
    }
    std::cout << "Maximum number of cg iterations exceeded\n";
    return(std::make_pair(its+1,residuum));
}



// ======================================================
// another version of congrad accepting an actual (squared) system matrix A
// ======================================================
template <typename T>
std::tuple<int, float> conjgrad (         // return values are the number of final iterations and residuum
                const device_array<T> &A, // Squared system matrix A
                const device_array<T> &b, // Vector of measurements
                device_array<T> &x,       // Parameter vector to be optimized (should contain the initial guess at the beginning)
                feed &f, device &d,

                // (optional) CG stopping criteria
                const int cg_maxit = 500, const double cg_tol = -1)
{

    // get dimensions of A
    std::size_t N = A.size();

    // test if A is squared (...or at least test if its definitely not squared :)
    assert(N%2 == 0);

    // define a lambda expression for A*x
    auto A_fun = [&A](device_array<T> &x, device_array<T> &result, feed &f)
    {
        // result = A_fun(x) = A * x
        cpuMatVecMul(A,x,result,f);  // PLEASE IMPLEMENT ME IN AURA!!
    };

    // call the conjgrad function
    return(conjgrad(A_fun, b, x, f, d, cg_maxit, cg_tol));
}

} // namespace aura
} // namespace boost
#endif // CONJGRAD_HPP
