#include <aura/misc/cprofile.h>

namespace aura {

namespace profile {

cprofile_sink cprofile_create_sink(unsigned long int initial_size) {
  return (void*) (new memory_sink(initial_size));
}

void cprofile_dump_sink(cprofile_sink sink, const char * filename) {
  ((memory_sink*)sink)->dump(filename);
}

void cprofile_delete_sink(cprofile_sink sink) {
  delete (memory_sink*)sink;
}

void cprofile_start(cprofile_sink sink, const char * name) {
  start(*((memory_sink*)sink), name); 
}

void cprofile_stop(cprofile_sink sink, const char * name) {
  stop(*((memory_sink*)sink), name); 
}

} // namespace profile

} // namespace aura

