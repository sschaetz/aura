#ifndef AURA_BACKEND_CUDA_DEVICE_PTR_HPP
#define AURA_BACKEND_CUDA_DEVICE_PTR_HPP

#include <cstddef>
#include <cuda.h>
#include <boost/aura/backend/cuda/device.hpp>
#include <boost/aura/backend/shared/memory_tag.hpp>


namespace boost
{
namespace aura {
namespace backend_detail {
namespace cuda {

template <typename T>
struct device_ptr {

  /// backend handle type
  /// can not be assigned to nullptr since 
  /// it is defined as long long unsigned int
  typedef CUdeviceptr backend_type;
  
  /// backend handle type
  typedef const CUdeviceptr const_backend_type;

public:

 /**
  * @brief create pointer that points nowhere
  */
  device_ptr() : memory_(0), offset_(0), device_(nullptr), 
	tag_(memory_tag::rw) {}


 /**
  * @brief create pointer that points nowhere
  */
  device_ptr(std::nullptr_t) : memory_(0), offset_(0), device_(nullptr), 
	tag_(memory_tag::rw) {}

 /**
  * @brief create device pointer that points to memory
  *
  * @param m memory that identifies device memory
  * @param d device the memory is allocated on 
  */
  device_ptr(backend_type & m, device & d, memory_tag tag = memory_tag::rw) : 
	  memory_(m), offset_(0), device_(&d), tag_(tag) {}

 /**
  * @brief create device pointer that points to memory
  *
  * @param m memory that identifies device memory
  * @param o offset of memory object 
  * @param d device the memory is allocated on 
  */
  device_ptr(const_backend_type & m, const std::size_t & o, device & d, 
		  memory_tag tag = memory_tag::rw) : 
	  memory_(m), offset_(o), device_(&d), tag_(tag) {}
  
  void invalidate() {
    memory_ = 0;
    device_ = nullptr;
    offset_ = 0;
    tag_ = memory_tag::rw;
  }
  
  /// returns a pointer to the device memory 
  backend_type get() { return memory_; }
  
  /// returns a pointer to the device memory 
  const_backend_type get() const { return memory_; }
  
  /// returns a pointer to the device memory 
  std::size_t get_offset() const { return offset_; }
  
  /// returns a pointer to the device memory 
  device & get_device() { return *device_; }

  /// returns the memory tag
  memory_tag get_memory_tag() const { return tag_; }

  /// assign operator
  device_ptr<T>& operator =(device_ptr<T> const & b) {
    memory_ = b.memory_;
    offset_ = b.offset_;
    device_ = b.device_;
    tag_ = b.tag_;
    return *this;
  }

  /// assign nullptr operator
  device_ptr<T>& operator =(std::nullptr_t) {
	invalidate();
	return *this;
  }
 
  /// addition operator
  device_ptr<T> operator +(const std::size_t & b) const {
    return device_ptr<T>(memory_, offset_+b, *device_, tag_);
  }

  /// addition assignment operator
  device_ptr<T>& operator +=(const std::size_t & b) {
    offset_ += b;
    return *this;
  }

 	/// prefix addition operator
  device_ptr<T> & operator ++() {
    ++offset_;
    return *this;
  }

  /// postfix addition operator
  device_ptr<T> operator ++(int) {
    return device_ptr<T>(memory_, offset_+1, *device_);
  }
 
  /// subtraction operator
  device_ptr<T> operator -(const std::size_t & b) const {
    return device_ptr<T>(memory_, offset_-b, *device_, tag_);
  }

  /// subtraction assignment operator
  device_ptr<T>& operator -=(const std::size_t & b) {
    offset_-=b;
    return *this;
  }

 	/// prefix subtraction operator
  device_ptr<T> & operator --() {
    --offset_;
    return *this;
  }

  /// postfix subtraction operator
  device_ptr<T> operator --(int) {
    return device_ptr<T>(memory_, offset_-1, *device_, tag_);
  }

  /// equal to operator
  bool operator ==(const device_ptr<T> & b) const  {
    if(nullptr == device_ || nullptr == b.device_) {
      return (nullptr == device_ && nullptr == b.device_ && 
        offset_ == b.offset_ && memory_ == b.memory_&&
	tag_ == b.tag_);
    }
    else {
      return (device_->get_ordinal() == b.device_->get_ordinal() && 
        offset_ == b.offset_ && memory_ == b.memory_ &&
	tag_ == b.tag_);
    }
  }
  
  bool operator ==(std::nullptr_t) const {
    return (nullptr == device_ && 0 == offset_ && 0 == memory_);
  }

 

  /// not equal to operator
  bool operator !=(const device_ptr<T> & b) const {
    return !(*this == b);
  }
  
  bool operator !=(std::nullptr_t) const {
    return !(*this == nullptr);
  }

private:

  /// actual pointer that identifies device memory
  backend_type memory_;
  
  /// the offset (OpenCL does not support arithmetic on the pointer)
  std::size_t offset_;

  /// reference to device the feed was created for
  device * device_;

  /// read+write readonly writeonly?
  memory_tag tag_;
};

/// equal to operator (reverse order)
template<typename T>
bool operator ==(std::nullptr_t, const device_ptr<T> & ptr) {
  return (ptr == nullptr);
}

/// not equal to operator (reverse order)
template<typename T>
bool operator !=(std::nullptr_t, const device_ptr<T> & ptr) {
  return (ptr != nullptr);
}

} // cuda 
} // backend_detail
} // aura
} // boost

#endif // AURA_BACKEND_CUDA_DEVICE_PTR_HPP

