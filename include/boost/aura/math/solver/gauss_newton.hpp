#ifndef GAUSS_NEWTON_HPP
#define GAUSS_NEWTON_HPP

#define IRGN_DEBUG

#include <boost/aura/math/math.hpp>


typedef std::complex<float> cfloat;
using namespace boost::aura;




/*======================================================
 Gauss-Newton Method for unsymmetric objective functions.

 It solves nonlinear optimization problem:
 x = arg_min { || Fh(F(x)) - Fh(y) ||_2^2 }

 For alpha_start > 0, the algorithm turns into the
 "Iteratively Regularized Gauss Newton" (IRGN) Method, solving
 the above problem + alpha || x - x0 ||_2^2.
 Alpha is divided by alpha_divi after each Newton step
 =======================================================*/
template <typename T, typename funOp_3args, typename funOp_4args, typename funOp_4args_2> //FIXME: two typenames look ugly
void gauss_newton(
       funOp_3args F,            // F  (x, result, feed)
       funOp_4args dF,           // dF (x, dx, result, feed)
       funOp_4args_2 dFh,        // dFh(x, y, result, feed)
       const device_array<T> &y, // Vector of measurements
       device_array<T> &x,       // Parameter vector to be optimized (should contain the initial guess at the beginning)
       device_array<T> &xr,      // Regularizer
       feed &f, device &d,       // Aura scope

       //
       const std::vector<float> &alpha)
{
    // x is the iteratively altered solution vector and assumed to initially contain the initial_guess

    // get number of newton steps from the length of alpha
    int irgn_steps = alpha.size();

    // get data dimensions
    bounds bX = x.get_bounds();
    bounds bY = y.get_bounds();

    // lambda function and variables to pass to the linear cg
    //funOp_3args A;              // pseudo system matrix
    device_array<T> s(bX, d);   // search vector
    device_array<T> b(bX, d);   // transformed data

    // allocate temporary vectors with Nx and Ny elements, needed in A
    device_array<T> tmpVec_bX(bX, d);
    device_array<T> tmpRes(bY, d);

    #ifdef IRGN_DEBUG
    device_array<float> tmpResScalar(1,d);
    #endif

    // allocate a scalar for the current regularization factor on the GPU
    device_array<float> curr_alpha(1,d);

    // declare variables for the cg iterations and the cg residuum
    int   cg_its;
    float cg_res;

    // loop over newton steps
    for (int i = 0; i < irgn_steps; i++) {

        #ifdef IRGN_DEBUG
        std::cout << "It: " << i+1 <<" ";
        #endif

        // copy current regularization factor to the GPU
        curr_alpha.set_value(alpha[i], f);

        // formulate the components "A" and "b" for the linear inner problem
        auto A = [&](device_array<T>& s, device_array<T>& result, feed &f)
        {
                //---------
                // result = dFh( x, dF(x,s)) + alpha * s
                //---------

                //dF(x,s)
                dF(x, s, tmpRes, f);

                //dFh(x, dF(x,s))
                dFh(x, tmpRes, result, f);

                //regularization
                if (alpha[i]){
                    // ... + alpha * s
                    math::mul(s, curr_alpha, tmpVec_bX, f);
                    math::add(result, tmpVec_bX, result, f);
                }
        };

        // ------------
        // b = dFh(x, y-F(x)) - alpha * (x - xr)
        // ------------

        // y - F(x)
        F(x, tmpRes, f);
        math::sub(y, tmpRes, tmpRes, f);

        #ifdef IRGN_DEBUG
        math::norm2(tmpRes, tmpResScalar,f);
        std::cout <<  tmpResScalar.get_value(f);
        #endif

        // dFh
        dFh(x, tmpRes, b, f);

        //regularization
        if (alpha[i]){
            // ... - alpha * (x - xr)
            math::sub(x, xr, tmpVec_bX, f);
            math::mul(tmpVec_bX, curr_alpha, tmpVec_bX,f);
            math::sub(b, tmpVec_bX, b, f);
        }

        // ------------
        // ------------


        // initial guess for s
        math::memset_zero(s, f);


        // CG ----------------

        // call the cg to solve || A s - b ||_2^2
        std::tie(cg_its, cg_res) = conjgrad(A, b, s, f, d);

        // --------------------

        #ifdef IRGN_DEBUG
        //std::cout << "(cg its: " << cg_its << ") res: " << cg_res << std::endl;
        std::cout << "(" << cg_its << ")"  << std::endl;
        #endif

        // update x = x + s
        math::add(x, s, x, f);
    }
}



/*======================================================
 Gauss-Newton Method for unsymmetric objective functions.
 (Use x0 as regularizer)

 It solves nonlinear optimization problem:
 x = arg_min { || Fh(F(x)) - Fh(y) ||_2^2 }

 For alpha_start > 0, the algorithm turns into the
 "Iteratively Regularized Gauss Newton" (IRGN) Method, solving
 the above problem + alpha || x - x0 ||_2^2.
 Alpha is divided by alpha_divi after each Newton step
 =======================================================*/
template <typename T, typename funOp_3args, typename funOp_4args, typename funOp_4args_2> //FIXME: two typenames look ugly
void gauss_newton(
       funOp_3args F,            // F  (x, result, feed)
       funOp_4args dF,           // dF (x, dx, result, feed)
       funOp_4args_2 dFh,        // dFh(x, y, result, feed)
       const device_array<T> &y, // Vector of measurements
       device_array<T> &x,       // Parameter vector to be optimized (should contain the initial guess at the beginning)
       feed &f, device &d,       // Aura scope

       //
       const std::vector<float> &alpha)
{
    // get data dimensions
    bounds bX = x.get_bounds();

    // store x0 (for regularization)
    device_array<T> xr(bX, d);
    copy(x,xr,f);
    
    gauss_newton(F, dF, dFh, y, x, xr, f, d, alpha);

}


/*======================================================
 Gauss-Newton Method for unsymmetric objective functions.

 It solves nonlinear optimization problem:
 x = arg_min { || Fh(F(x)) - Fh(y) ||_2^2 }

 For alpha_start > 0, the algorithm turns into the
 "Iteratively Regularized Gauss Newton" (IRGN) Method, solving
 the above problem + alpha || x - x0 ||_2^2.
 Alpha is divided by alpha_divi after each Newton step
 =======================================================*/
template <typename T, typename funOp_3args, typename funOp_4args, typename funOp_4args_2> //FIXME: two typenames look ugly
void gauss_newton(
       funOp_3args F,            // F  (x, result, feed)
       funOp_4args dF,           // dF (x, dx, result, feed)
       funOp_4args_2 dFh,        // dFh(x, y, result, feed)
       const device_array<T> &y, // Vector of measurements
       device_array<T> &x,       // Parameter vector to be optimized (should contain the initial guess at the beginning)
       feed &f, device &d,       // Aura scope

       // (optional) parameters for the IRGN
       const unsigned int irgn_steps = 10, const float alpha_start = 0, const float alpha_divi = 3)
{
    // x is the iteratively altered solution vector and assumed to initially contain the initial_guess

    // create a vector alpha containing the regularization factor for each newton step
    std::vector<float> alpha(irgn_steps);

    // fill the vector
    alpha[0] = alpha_start;
    if (alpha_start != 0 && irgn_steps > 1){
        for (std::size_t i = 1; i < irgn_steps; i++){
            alpha[i] = alpha[i-1] / alpha_divi;
        }
    }

    // call the IRGN
    gauss_newton(F, dF, dFh, y, x, f, d, alpha);

}


/* =====================================================
 Gauss-Newton Method for symmetric objective functions.

 It solves nonlinear optimization problem:
 x = arg_min { || F(x) - y ||_2^2 }

 For alpha_start > 0, the algorithm turns into the
 "Iteratively Regularized Gauss Newton" (IRGN) Method, solving
 the above problem + alpha || x - x0 ||_2^2.
 Alpha is divided by alpha_divi after each Newton step
 =======================================================*/       
template <typename T, typename funOp_3args, typename funOp_4args>
void gauss_newton(
           funOp_3args F,      // F(x, result, feed)
           funOp_4args dF,     // dF(x, dx, result, feed)
           device_array<T> &y, // Vector of measurements
           device_array<T> &x, // Parameter vector to be optimized (should contain the initial guess at the beginning)
           feed &f, device &d, // Aura scope

           // (optional) parameters for the IRGN
           const int irgn_steps = 10, const float alpha_start = 0, const float alpha_divi = 3)
{
    // x is the iteratively altered solution vector and assumed to initially contain the initial_guess

    // the (iteratively adapted) regularization factor
    device_array<float> g_alpha(1,d);
    device_array<float> g_alpha_divi(1,d);
    g_alpha.set_value(0, alpha_start, f);
    g_alpha_divi.set_value(0, alpha_divi, f);

    // get data dimensions
    size_t Ny = y.size();
    assert(Ny == x.size());

    // lambda function and variables to pass to the linear cg
    funOp_3args A;              // pseudo system matrix
    device_array<T> s(Ny, d);   // search vector
    device_array<T> b(Ny, d);   // data


    // store x0 (for regularization)
    device_array<T> x0(Ny, d);
    copy(x,x0,f);

    // allocate a temporary vector with Ny elements, needed in A
    device_array<T> tmpVec_Ny(Ny, d);

    // loop over newton steps
    for (int i = 0; i < irgn_steps; i++) {


        // ----------------

        // formulate the components "A" and "b" for linear inner problem
        A = [&](device_array<T>& s, device_array<T>& result, feed &f)
        {
                // result = dF(x) * s ...
                dF(x, s, result, f);

                // ... + alpha * s  (regularization)
                math::mul(s,g_alpha,tmpVec_Ny,f);
                math::add(result,tmpVec_Ny,result,f);

        };

        // b = y - F(x) - alpha * (x-x0)
        F(x, b, f);
        math::sub(y,b,b,f);

        // regularization
        math::sub(x,x0,tmpVec_Ny,f);
        math::mul(tmpVec_Ny, g_alpha, tmpVec_Ny,f);
        math::sub(b,tmpVec_Ny,b,f);

        // ----------------


        // initial guess for s
        math::memset_zero(s,f);

        // call the cg to solve || A s - b ||_2^2
        conjgrad(A, b, s, f, d);

        // update x = x + s
        math::add(x,s,x,f);

        // decrease regularization (alpha /= alpha_divi);
        math::div(g_alpha, g_alpha_divi, g_alpha,f);
    }
}



#endif // GAUSS_NEWTON_HPP
