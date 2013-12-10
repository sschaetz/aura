#ifndef AURA_BACKEND_OPENCL_FFT_HPP
#define AURA_BACKEND_OPENCL_FFT_HPP

#include <tuple>
#include <boost/move/move.hpp>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device_ptr.hpp>
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
  inline explicit fft(device & d, feed & f,
    const fft_dim & dim, const fft::type & type,
    std::size_t batch = 1, 
    const fft_embed & iembed = fft_embed(),
    std::size_t istride = 1, std::size_t idist = 0,
    const fft_embed & oembed = fft_embed(),
    std::size_t ostride = 1, std::size_t odist = 0) : 
    context_(d.get_context()), buffer_(), type_(type) {

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

    // bake plan
    AURA_CLFFT_SAFE_CALL(clfftBakePlan(inplace_handle_, 1, 
      const_cast<cl_command_queue*>(&f.get_backend_stream()),
      nullptr, nullptr));
    AURA_CLFFT_SAFE_CALL(clfftBakePlan(outofplace_handle_, 1, 
      const_cast<cl_command_queue*>(&f.get_backend_stream()),
      nullptr, nullptr));
    wait_for(f);    
    std::size_t buffer_size1, buffer_size2;
    AURA_CLFFT_SAFE_CALL(clfftGetTmpBufSize(inplace_handle_, &buffer_size1));
    AURA_CLFFT_SAFE_CALL(clfftGetTmpBufSize(outofplace_handle_, &buffer_size2));
    if(0 < buffer_size1 || 0 < buffer_size2) {
      buffer_ = device_malloc<char>((buffer_size1 > buffer_size2) ? 
        buffer_size1 : buffer_size2, d);
    }
  }

  /**
   * move constructor, move fft information here, invalidate other
   *
   * @param f fft to move here
   */
  fft(BOOST_RV_REF(fft) f) :
    context_(f.context_), inplace_handle_(f.inplace_handle_), 
    outofplace_handle_(f.outofplace_handle_),
    buffer_(f.buffer_) {  
    f.context_ = nullptr; 
  }

  /**
   * move assignment, move fft information here, invalidate other
   *
   * @param f fft to move here
   */
  fft& operator=(BOOST_RV_REF(fft) f) {
    finalize();
    context_= f.context_;
    inplace_handle_ = f.inplace_handle_;
    outofplace_handle_ = f.outofplace_handle_;
    type_ = f.type_;
    buffer_ = f.buffer_; 
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
      if(nullptr != buffer_) {
        device_free(buffer_);
      }
    }
  }

  /// in-place plan 
  clfftPlanHandle inplace_handle_; 

  /// out-of-place plan  
  clfftPlanHandle outofplace_handle_; 
  
  /// temporary buffer for transforms 
  device_ptr<char> buffer_;

  /// fft type
  type type_;

  // give free functions access to context 
  template <typename T1, typename T2>
  friend void fft_forward(device_ptr<T1> & dst, device_ptr<T2> & src, 
    fft & plan, const feed & f);
  template <typename T1, typename T2>
  friend void fft_inverse(device_ptr<T1> & dst, device_ptr<T2> & src, 
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
template <typename T1, typename T2>
void fft_forward(device_ptr<T1> & dst, device_ptr<T2> & src, 
  fft & plan, const feed & f) {
  typename device_ptr<T1>::backend_type dm = dst.get();
  typename device_ptr<T1>::backend_type sm = src.get();
  if(dst == src) {
    AURA_CLFFT_SAFE_CALL(clfftEnqueueTransform(plan.inplace_handle_, 
      CLFFT_FORWARD, 1, const_cast<cl_command_queue*>(&f.get_backend_stream()), 
      0, NULL, NULL, &sm, NULL, plan.buffer_.get()));
  } else {
    AURA_CLFFT_SAFE_CALL(clfftEnqueueTransform(plan.outofplace_handle_, 
      CLFFT_FORWARD, 1, const_cast<cl_command_queue*>(&f.get_backend_stream()), 
      0, NULL, NULL, &sm, &dm, plan.buffer_.get()));
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
template <typename T1, typename T2>
void fft_inverse(device_ptr<T1> & dst, device_ptr<T2> & src, 
  fft & plan, const feed & f) {
  typename device_ptr<T1>::backend_type dm = dst.get();
  typename device_ptr<T1>::backend_type sm = src.get();
  if(dst == src) {
    AURA_CLFFT_SAFE_CALL(clfftEnqueueTransform(plan.inplace_handle_, 
      CLFFT_BACKWARD, 1, const_cast<cl_command_queue*>(&f.get_backend_stream()), 
      0, NULL, NULL, &sm, NULL, plan.buffer_.get()));
  } else {
    AURA_CLFFT_SAFE_CALL(clfftEnqueueTransform(plan.outofplace_handle_, 
      CLFFT_BACKWARD, 1, const_cast<cl_command_queue*>(&f.get_backend_stream()), 
      0, NULL, NULL, &sm, &dm, plan.buffer_.get()));
  }
}

} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_FFT_HPP

