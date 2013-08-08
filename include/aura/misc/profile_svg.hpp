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
  
  // hashmap for levels
  typedef std::pair<std::size_t, const char *> key;
  typedef std::map<key, unsigned int> level_map;
  level_map levels;
  
  FILE * f = fopen(filename, "w");
  if(f == NULL) {
    AURA_ERROR("Unable to open file.");
  }
  fprintf(f, "%s\n", svg_header);

  unsigned int level_counter = 0;
  for(std::size_t i=0; i<sink.data_.size(); i++) {
    // only print if there is a start value
    if(!sink.data_[i].start) {
      continue;
    }

    // find the current level
    unsigned int level = level_counter;
    level_map::const_iterator got = 
      levels.find(key(sink.data_[i].thread_id, sink.data_[i].name));
    if(got == levels.end()) {
      levels.insert(
        std::pair<key, unsigned int>(
          key(sink.data_[i].thread_id, sink.data_[i].name), 
          level_counter));
      level_counter++;
    } else {
      level = got->second;
    }

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

