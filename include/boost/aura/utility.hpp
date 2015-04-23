#ifndef AURA_UTILITY_HPP
#define AURA_UTILITY_HPP
namespace boost
{
  namespace aura
  {
    // make_unique appears in gcc 4.9 XXX This should be obsolete then
    template<typename T, typename... Args>
      std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
      }
  }
}

#endif // AURA_UTILITY_HPP
