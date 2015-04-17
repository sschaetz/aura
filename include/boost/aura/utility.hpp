#ifndef AURA_UTILITY_HPP
#define AURA_UTILITY_HPP
namespace boost
{
  namespace aura
  {
    // Calculates the total sizeof of all aguments
    template<typename... Targs>
      struct tsizeof;
    template<typename T0>
      struct tsizeof<T0>
      {
        enum ts : size_t {sz = sizeof(T0)};
      };

    template<typename T0, typename... Targs>
      struct tsizeof<T0,Targs...>
      {
        enum ts : size_t {sz = sizeof(T0)+tsizeof<Targs...>::sz};
      };
    //----------------------------------------------
    // make_unique appears in gcc 4.9 XXX This should be obsolete then
    template<typename T, typename... Args>
      std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
      }
  }
}

#endif // AURA_UTILITY_HPP
