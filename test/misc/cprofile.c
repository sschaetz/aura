
#include <aura/misc/cprofile.h>

int main(void) {
  cprofile_sink s = cprofile_create_sink(10000);
  cprofile_start(s, "main");
  cprofile_stop(s, "main");
  cprofile_dump_sink(s, "/tmp/profile.log");
  cprofile_delete_sink(s);
}

