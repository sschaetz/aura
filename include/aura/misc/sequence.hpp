#ifndef AURA_MISC_SEQUENCE_HPP
#define AURA_MISC_SEQUENCE_HPP

#include <stdio.h>
#include <cctype>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <aura/error.hpp>
#include <aura/detail/svec.hpp>

namespace aura {

/// generate a sequence of tuples, defined by a sequence string
/// T must be a numeric type
template <typename T, std::size_t max_size_>
struct sequence {

  /// default ctor
  inline sequence() {}
  
  /**
   * parse sequence strings like:
   *
   * 4:+1:10
   * 4:-1:3,2:*2:32
   * start-dim0:op arg:stop-dim0,start-dim1:op arg:stop-dim1
   *
   * and generate parser that give the correct sequence
   */
  inline sequence(const char * definition) {
    const char * c = definition;
    char numstore[32];
    char * n = numstore;
    int state = 0; // 0=start, 1=op/arg 2=end
    while(true) {
      if(isdigit(*c)) {
        *n++ = *c;
      } else {
        *n = '\0';
        n = numstore;
      }
      
      if(*c == ':' || *c == ',' || *c == '\0') {
        if(state == 0) {
          start_.push_back(atoi(numstore));          
          cur_.push_back(atoi(numstore));
        }
        else if(state == 1) {
          arg_.push_back(atoi(numstore));          
        }
        else if(state == 2) {
          end_.push_back(atoi(numstore));          
        }
        state = (state+1) % 3;
      }
     
      if(*c == '+' || *c == '-' || *c == '^' || *c == '*' || *c == '/') {
        op_.push_back(*c);
      }

      if(*c == '\0') {
        break;
      }
      c++;
    }
  }

  /// rewind to starting position
  inline void rewind() {
    std::copy(&start_[0], &start_[start_.size()], &cur_[0]);
  }

  /**
   * get the next values of the sequence, return the sequence and true
   * if the end was not reached yet
   */
  inline std::pair<svec<T, max_size_>, bool> next() {

    std::pair<svec<T, max_size_>, bool> ret(cur_, false); 
    // check if end was reached
    for(std::size_t i=0; i<cur_.size(); i++) {
      if(cur_[i] >= end_[i]) {
        return ret;
      } 
    }

    ret.second = true; 
    
    // calculate new values 
    for(std::size_t i=0; i<cur_.size(); i++) {
      switch (op_[i]) {
        case '+':
          cur_[i] += arg_[i];
          break;
        case '-':
          cur_[i] -= arg_[i];
          break;
        case '*':
          cur_[i] *= arg_[i];
          break;
        case '/':
          cur_[i] /= arg_[i];
          break;
        case '^':
          cur_[i] = pow(cur_[i], op_[i]);
          break;
      }
    }
    return ret; 
  }

  // get the number of elements in the sequence
  std::size_t size() {
    std::size_t s = 0;
    bool good = false;
    std::tie(std::ignore, good) = next();
    while(good) {
      s++;
      std::tie(std::ignore, good) = next();
    }
    rewind();
    return s;
  }

  /// current values for each dimension
  svec<T, max_size_> cur_; 
  /// current values for each dimension
  svec<T, max_size_> start_; 
  /// end values for each dimension
  svec<T, max_size_> end_;
  /// operations to calculate next values
  svec<char, max_size_> op_;
  /// arguments to operations to calculate next values
  svec<T, max_size_> arg_;  

};


} // namespace aura

#endif // AURA_MISC_SEQUENCE_HPP

