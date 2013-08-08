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

  // map function name to level in thread
  typedef std::map<const char *, std::size_t> l0_map;
  // for each thread hold a map of function names 
  typedef std::map<std::size_t, l0_map> l1_map;
  typedef std::map<std::size_t, std::size_t> l2_map;

  l1_map levels;
  for(std::size_t i=0; i<sink.data_.size(); i++) {
    // check if thread exists
    if(0 == levels.count(sink.data_[i].thread_id)) {
      levels.insert(l0_map(), levels.size());
    }
    // check if function exists
    if(0 == levels[sink.data_[i].thread_id].count(sink.data_[i].name)) {
       levels[sink.data_[i].thread_id].insert(
         sink.data_[i].name, levels[sink.data_[i].thread_id].size());
    }
  }

  // now calculate the offset
  typedef l1_map::iterator l1_map_it;
  std::size_t level_offset = 0;
  l2_map level_offsets;
  for (l1_map_it iter = levels.begin(); iter != levels.end(); ++iter) {
    level_offsets.insert(iter->first, level_offset); 
    level_offset += iter->second.size(); 
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
    unsigned int level = levels[sink.data_[i].thread_id][sink.data_[i].name] +
     level_offsets[sink.data_[i].thread_id]; 
    

    // find the stop value
    for(std::size_t j = i+1; j<sink.data_.size(); j++) {
      if(sink.data_[j].thread_id == sink.data_[i].thread_id && 
        0 == strcmp(sink.data_[j].name, sink.data_[i].name) &&
        false == sink.data_[j].start)
      {
        fprintf(f, "<rect x=\"%f\" y=\"%d\" width=\"%f\" height=\"%d\" " 
          "fill=\"yellow\" stroke=\"black\" stroke-width=\"1\" />\n",
          (sink.data_[i].timestamp - min)*scale, level*level_height,
          (sink.data_[j].timestamp - sink.data_[i].timestamp) * scale,
          box_height);
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

