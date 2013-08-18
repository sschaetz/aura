// test how namespace aliases work

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

#ifdef 0
namespace level1 {
namespace shortcut { // this does not work
  struct bar{};
} // level1
} // shortcut
#endif

int main(void) {
  level1::shortcut::foo f;
#ifdef 0
  level1::shortcut::bar b;
#endif
}

