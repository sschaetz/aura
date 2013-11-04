#ifndef AURA_BACKEND_OPENCL_FFT_HPP
#define AURA_BACKEND_OPENCL_FFT_HPP

#include <tuple>
#include <boost/move/move.hpp>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/memory.hpp>
#include <aura/detail/svec.hpp>
#include <aura/backend/opencl/device.hpp>

#include <clFFT.h>

namespace aura {
namespace backend_detail {
namespace opencl {

typedef std::size_t fft_size;
typedef svec<fft_size, 3> fft_dim;
typedef svec<fft_size, 3> fft_embed;

/**
 * fft class
 */
class fft {

private:
  BOOST_MOVABLE_BUT_NOT_COPYABLE(fft)
  typedef std::tuple<clfftPrecision, clfftLayout, clfftLayout> clfft_type;

public:
  enum type {
    r2c,  // real to complex  
    c2r,  // complex to real 
    c2c,  // complex to complex  
    d2z,  // double to double-complex 
    z2d,  // double-complex to double 
    z2z   // double-complex to double-complex
  };

  enum direction {
    fwd = CLFFT_FORWARD,
    inv = CLFFT_BACKWARD 
  };

  /**
   * create empty fft object without device and stream
   */
  inline explicit fft() : context_(nullptr) { 
  }

  /**
   * create fft 
   *
   * @param d device to create fft for
   */
  inline explicit fft(device & d, const fft_dim & dim, const fft::type & type,
    std::size_t batch = 1, 
    const fft_embed & iembed = fft_embed(),
    std::size_t istride = 1, std::size_t idist = 0,
    const fft_embed & oembed = fft_embed(),
    std::size_t ostride = 1, std::size_t odist = 0) : context_(d.get_context()), 
    type_(type) {

    // FIXME handle strides and embed etc.
    // we need to create a default plan
    AURA_CLFFT_SAFE_CALL(clfftCreateDefaultPlan(&inplace_handle_, 
      context_->get_backend_context(), static_cast<clfftDim>(dim.size()), &dim[0]));

    AURA_CLFFT_SAFE_CALL(clfftSetPlanBatchSize(inplace_handle_, batch));

    clfft_type temptype = map_type(type);

    AURA_CLFFT_SAFE_CALL(clfftSetPlanPrecision(inplace_handle_, 
      std::get<0>(temptype)));
    AURA_CLFFT_SAFE_CALL(clfftSetLayout(inplace_handle_, 
      std::get<1>(temptype), std::get<2>(temptype)));
   
    // different result location, rest is the same
    AURA_CLFFT_SAFE_CALL(clfftCopyPlan(&outofplace_handle_, 
      context_->get_backend_context(), inplace_handle_));
    
    AURA_CLFFT_SAFE_CALL(clfftSetResultLocation(inplace_handle_, 
      CLFFT_INPLACE)); 
    AURA_CLFFT_SAFE_CALL(clfftSetResultLocation(outofplace_handle_, 
      CLFFT_OUTOFPLACE));

    // FIXME
    // since we have no feed here we can not bake the plans, we need an
    // extra method for that maybe
  }

  /**
   * move constructor, move fft information here, invalidate other
   *
   * @param f fft to move here
   */
  fft(BOOST_RV_REF(fft) f) :
    context_(f.context_), inplace_handle_(f.inplace_handle_), 
    outofplace_handle_(f.outofplace_handle_), empty_(false)
  {  
    f.empty_ = true; 
  }

  /**
   * move assignment, move fft information here, invalidate other
   *
   * @param f fft to move here
   */
  fft& operator=(BOOST_RV_REF(fft) f)
  {
    finalize();
    context_= f.context_;
    inplace_handle_ = f.inplace_handle_;
    outofplace_handle_ = f.outofplace_handle_;
    type_ = f.type_;
    f.context_= nullptr; 
    return *this;
  }

  /**
   * destroy fft
   */
  inline ~fft() {
    finalize();
  }

  /**
   * set feed
   */
  void set_feed(const feed & f) { }

  /**
   * return fft type
   */
  const type & get_type() const {
    return type_;
  }

  /// map fft type to clfft_type 
  clfft_type map_type(fft::type type) {
    switch(type) {
      case r2c: 
        return clfft_type(CLFFT_SINGLE, 
          CLFFT_REAL, 
          CLFFT_COMPLEX_INTERLEAVED);
      case c2r: 
        return clfft_type(CLFFT_SINGLE, 
          CLFFT_COMPLEX_INTERLEAVED, 
          CLFFT_REAL);
      case c2c: 
        return clfft_type(CLFFT_SINGLE, 
          CLFFT_COMPLEX_INTERLEAVED, 
          CLFFT_COMPLEX_INTERLEAVED);
      case d2z: 
        return clfft_type(CLFFT_DOUBLE, 
          CLFFT_REAL, 
          CLFFT_COMPLEX_INTERLEAVED);
      case z2d: 
        return clfft_type(CLFFT_DOUBLE, 
            CLFFT_COMPLEX_INTERLEAVED, 
            CLFFT_REAL);
      case z2z: 
        return clfft_type(CLFFT_DOUBLE, 
          CLFFT_COMPLEX_INTERLEAVED, 
          CLFFT_COMPLEX_INTERLEAVED);
      default:
        return clfft_type(ENDPRECISION, ENDLAYOUT, ENDLAYOUT);
    }
  }

protected:
  /// context handle
  detail::context * context_;
  
private:
  /// finalize object (called from dtor and move assign)
  void finalize() {
    if(nullptr != context_) {
      AURA_CLFFT_SAFE_CALL(clfftDestroyPlan(&inplace_handle_));
      AURA_CLFFT_SAFE_CALL(clfftDestroyPlan(&outofplace_handle_));
    }
  }

  /// in-place plan 
  clfftPlanHandle inplace_handle_; 

  /// out-of-place plan  
  clfftPlanHandle outofplace_handle_; 

  /// fft type
  type type_;

  /// empty marker
  bool empty_;

  // give free functions access to context 
  friend void fft_forward(memory & dst, memory & src, 
    fft & plan, const feed & f);
  friend void fft_inverse(memory & dst, memory & src, 
    fft & plan, const feed & f);

};

/// initialize fft library
inline void fft_initialize() {
  clfftSetupData setupdata;
  AURA_CLFFT_SAFE_CALL(clfftInitSetupData(&setupdata));
  AURA_CLFFT_SAFE_CALL(clfftSetup(&setupdata));
}
/// finish using fft library and release all associated resources
inline void fft_terminate() {
  clfftTeardown();
}

/**
 * @brief calculate forward fourier transform
 * 
 * @param dst pointer to result of fourier transform
 * @param src pointer to input of fourier transform
 * @param plan that is used to calculate the fourier transform
 * @param f feed the fourier transform should be calculated in
 */
inline void fft_forward(memory & dst, memory & src, 
  fft & plan, const feed & f) {
  if(dst == src) {
    AURA_CLFFT_SAFE_CALL(clfftEnqueueTransform(plan.inplace_handle_, 
      CLFFT_FORWARD, 1, const_cast<cl_command_queue*>(&f.get_backend_stream()), 
      0, NULL, NULL, &src, NULL, NULL));
  } else {
    AURA_CLFFT_SAFE_CALL(clfftEnqueueTransform(plan.outofplace_handle_, 
      CLFFT_FORWARD, 1, const_cast<cl_command_queue*>(&f.get_backend_stream()), 
      0, NULL, NULL, &src, &dst, NULL));
  }
}


/**
 * @brief calculate forward fourier transform
 * 
 * @param dst pointer to result of fourier transform
 * @param src pointer to input of fourier transform
 * @param plan that is used to calculate the fourier transform
 * @param f feed the fourier transform should be calculated in
 */
inline void fft_inverse(memory & dst, memory & src, 
  fft & plan, const feed & f) {
  if(dst == src) {
    AURA_CLFFT_SAFE_CALL(clfftEnqueueTransform(plan.inplace_handle_, 
      CLFFT_BACKWARD, 1, const_cast<cl_command_queue*>(&f.get_backend_stream()), 
      0, NULL, NULL, &src, NULL, NULL));
  } else {
    AURA_CLFFT_SAFE_CALL(clfftEnqueueTransform(plan.outofplace_handle_, 
      CLFFT_BACKWARD, 1, const_cast<cl_command_queue*>(&f.get_backend_stream()), 
      0, NULL, NULL, &src, &dst, NULL));
  }
}

} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_FFT_HPP

