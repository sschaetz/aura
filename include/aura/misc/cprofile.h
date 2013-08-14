#ifndef AURA_MISC_CPROFILE_H
#define AURA_MISC_CPROFILE_H

#ifdef __cplusplus


namespace aura {

namespace profile {

typedef void * cprofile_sink;

extern "C" cprofile_sink cprofile_create_sink(unsigned long int initial_size);
extern "C" void cprofile_delete_sink(cprofile_sink sink);
extern "C" void cprofile_start(cprofile_sink sink, const char * name);
extern "C" void cprofile_stop(cprofile_sink sink, const char * name);
extern "C" void cprofile_dump_sink(cprofile_sink sink, const char * filename);
extern "C" void cprofile_dump_sink_svg(cprofile_sink sink, const char * filename);

#else

typedef void * cprofile_sink;

cprofile_sink cprofile_create_sink(unsigned long int initial_size);
void cprofile_delete_sink(cprofile_sink sink);
void cprofile_start(cprofile_sink sink, const char * name);
void cprofile_stop(cprofile_sink sink, const char * name);
void cprofile_dump_sink(cprofile_sink sink, const char * filename);
void cprofile_dump_sink_svg(cprofile_sink sink, const char * filename);

#endif


#ifdef __cplusplus

} // namespace profile

} // namespace aura

#endif

#endif // AURA_MISC_CPROFILE_H

