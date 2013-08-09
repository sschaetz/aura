#ifndef AURA_MISC_PROFILE_SVG_HPP
#define AURA_MISC_PROFILE_SVG_HPP

#include <stdio.h>
#include <map>
#include <utility>

#include <aura/misc/now.hpp>
#include <aura/error.hpp>
#include <aura/misc/profile.hpp>

namespace aura {

namespace profile {

const char * svg_header = "<?xml version=\"1.0\" standalone=\"no\"?>" 
  "<svg viewBox=\"0 0 1002 400\" "
  "xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
  "xmlns:xlink=\"http://www.w3.org/1999/xlink\">";
const char * svg_footer = "</svg>";


/// dump profile data to file
inline void dump_svg(memory_sink & sink, const char * filename) {
  if(sink.data_.size() < 1) {
    return;
  }
 
  int graph_width = 1000; 
  int box_height = 20;
  int level_height = 30;

  std::lock_guard<std::mutex> guard(sink.mtx_);
  
  // find minimum and maximum
  double min = sink.data_[0].timestamp;
  double max = min;
  for(std::size_t i=0; i<sink.data_.size(); i++) {
    if(min > sink.data_[i].timestamp) {
      min = sink.data_[i].timestamp;
    }
    if(max < sink.data_[i].timestamp) {
      max = sink.data_[i].timestamp;
    }
  }
  double scale = (double)graph_width / (max-min);
  
  //printf("max %f min %f range %f scale %f\n", max, min, max-min, scale);

  // map thread_id to [0 ... N]
  // store for each thread_id the number of levels in a vector
  // store for each (thread_id, function) the current level in the thread
 
  typedef std::map<std::size_t, std::size_t> t_threadmap;
  typedef std::pair<std::size_t, const char *> t_funckey;
  typedef std::map<t_funckey, unsigned int> t_funcmap;
 
  t_threadmap threadmap;
  std::vector<std::size_t> threadlevels;
  t_funcmap funcmap;
  
  std::size_t level_counter = 0;
  for(std::size_t i=0; i<sink.data_.size(); i++) {
    // only print if there is a start value
    if(!sink.data_[i].start) {
      continue;
    }
    t_threadmap::const_iterator tmgot = 
      threadmap.find(sink.data_[i].thread_id);
    if(tmgot == threadmap.end()) {
      threadmap.insert(std::pair<std::size_t, std::size_t>(
        sink.data_[i].thread_id, threadmap.size()));
      threadlevels.push_back(0);
    }

    t_funcmap::const_iterator fmgot = 
      funcmap.find(t_funckey(sink.data_[i].thread_id, sink.data_[i].name));
    if(fmgot == funcmap.end()) {
      funcmap.insert(
        std::pair<t_funckey, std::size_t>(
          t_funckey(sink.data_[i].thread_id, sink.data_[i].name), level_counter));
      level_counter++;
    }
  }

  

  FILE * f = fopen(filename, "w");
  if(f == NULL) {
    AURA_ERROR("Unable to open file.");
  }
  fprintf(f, "%s\n", svg_header);

  for(std::size_t i=0; i<sink.data_.size(); i++) {
    // only print if there is a start value
    if(!sink.data_[i].start) {
      continue;
    }

    // find the current level
    std::size_t tid = threadmap[sink.data_[i].thread_id];
    
    t_funcmap::const_iterator fmgot = 
      funcmap.find(t_funckey(sink.data_[i].thread_id, sink.data_[i].name));
    std::size_t level = fmgot->second + tid;

    // find the stop value
    for(std::size_t j = i+1; j<sink.data_.size(); j++) {
      if(sink.data_[j].thread_id == sink.data_[i].thread_id && 
        0 == strcmp(sink.data_[j].name, sink.data_[i].name) &&
        false == sink.data_[j].start)
      {
        fprintf(f, "<rect x=\"%f\" y=\"%ld\" width=\"%f\" height=\"%d\" " 
          "fill=\"yellow\" stroke=\"black\" stroke-width=\"1\" " 
          "ap_name=\"%s\"/>\n",
          (sink.data_[i].timestamp - min)*scale, level*level_height,
          (sink.data_[j].timestamp - sink.data_[i].timestamp) * scale,
          box_height, sink.data_[i].name);
        break;
      }
    }
  }

  fprintf(f, "%s\n", svg_footer);
  if(0 != fclose(f)) {
    AURA_ERROR("Unable to close file.");
  }
}
  

} // namespace profile

} // namespace aura

#endif // AURA_MISC_PROFILE_SVG_HPP

