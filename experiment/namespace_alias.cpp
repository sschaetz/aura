// test how namespace aliases work
//
// can we have a namespace alias and a namespace with the same name?
// the answer is no, removing the #if 0 #endif lines, this code does not
// compile 

namespace level1 {
namespace level2 {
namespace level3 {
  struct foo{};
} // level3
} // level2
} // level1


namespace level1 {
namespace shortcut = level2::level3;
}

#if 0
namespace level1 {
namespace shortcut { // this does not work
  struct bar{};
} // level1
} // shortcut
#endif

int main(void) {
  level1::shortcut::foo f;
  (void)f;
#if 0
  level1::shortcut::bar b;
#endif
}

