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

namespace constants {

const int graph_width = 1000; 
const int box_height = 20;
const int padding = 5;

const char * svg_header = "<?xml version=\"1.0\" standalone=\"no\"?>" 
  "<svg viewBox=\"0 0 1000 200\" "
  "xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
  "xmlns:xlink=\"http://www.w3.org/1999/xlink\">";
const char * svg_footer = "</svg>";

/* colors from http://ethanschoonover.com/solarized */

const char * svg_func_colors[] =
  { "#b58900",
    "#cb4b16",
    "#dc322f",
    "#d33682",
    "#6c71c4",
    "#268bd2",
    "#2aa198",
    "#859900",
  };

const int svg_num_func_colors = 8;

const char * svg_thread_colors[] =
  { "#93a1a1",
    "#586e75",
  };

const int svg_num_thread_colors = 2;

} // namespace constants

/// dump profile data to file
inline void dump_svg(memory_sink & sink, const char * filename) {
  if(sink.data_.size() < 1) {
    return;
  }
 

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
  double scale = (double)constants::graph_width / (max-min);
  
  //printf("max %f min %f range %f scale %f\n", max, min, max-min, scale);

  // map thread_id to [0 ... N]
  // store for each thread_id the number of levels in a vector
  // store for each (thread_id, function) the current level in the thread
 
  typedef std::map<std::size_t, std::size_t> t_threadmap;
  typedef std::pair<std::size_t, const char *> t_funckey;
  typedef std::pair<std::size_t, std::size_t> t_funcval;
  typedef std::map<t_funckey, t_funcval> t_funcmap;
 
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

    std::size_t tid = threadmap[sink.data_[i].thread_id];

    t_funcmap::const_iterator fmgot = 
      funcmap.find(t_funckey(sink.data_[i].thread_id, sink.data_[i].name));
    if(fmgot == funcmap.end()) {
      funcmap.insert(
        std::pair<t_funckey, t_funcval>(
          t_funckey(sink.data_[i].thread_id, sink.data_[i].name), 
          t_funcval(level_counter, threadlevels[tid])));
      level_counter++;
      threadlevels[tid] = threadlevels[tid] +1;
    }
  }

  

  FILE * f = fopen(filename, "w");
  if(f == NULL) {
    AURA_ERROR("Unable to open file.");
  }
  fprintf(f, "%s\n", constants::svg_header);

  for(std::size_t i=0; i<sink.data_.size(); i++) {
    // only print if there is a start value
    if(!sink.data_[i].start) {
      continue;
    }

    // find the current level
    std::size_t tid = threadmap[sink.data_[i].thread_id];
    
    t_funcmap::const_iterator fmgot = 
      funcmap.find(t_funckey(sink.data_[i].thread_id, sink.data_[i].name));
    std::size_t level = fmgot->second.first;
    
    // draw a backdrop 
    if(threadlevels[tid] < 10000) {
      fprintf(f, "<rect x=\"0\" y=\"%ld\" width=\"%d\" height=\"%ld\" " 
        "fill=\"%s\" stroke-width=\"0\"/>\n",
        /* y:      */ (tid==0) ? 0 : level*constants::box_height + 
                          level*constants::padding + 
                          tid*constants::padding,
        /* width:  */ constants::graph_width,
        /* height: */ threadlevels[tid] * 
                        (constants::box_height+constants::padding) + 
                        constants::padding,
        /* fill:   */ constants::svg_thread_colors[
                        tid % constants::svg_num_thread_colors]);

      threadlevels[tid] = threadlevels[tid]+10000;
    }
    
    // find the stop value
    for(std::size_t j = i+1; j<sink.data_.size(); j++) {
      if(sink.data_[j].thread_id == sink.data_[i].thread_id && 
        0 == strcmp(sink.data_[j].name, sink.data_[i].name) &&
        false == sink.data_[j].start)
      {
        fprintf(f, "<rect class=\"m\" x=\"%f\" y=\"%ld\" width=\"%f\" " 
          "height=\"%d\" fill=\"%s\" stroke-width=\"0\" " 
          "ap_name=\"%s\" ap_tid=\"%ld\" "
          "ap_start=\"%0.9f\" ap_stop=\"%0.9f\" ap_dur=\"%0.9f\"/>\n",
          /* x:       */ (sink.data_[i].timestamp - min)*scale, 
          /* y:       */ level*constants::box_height + 
                           (level+1)*constants::padding + 
                           tid*constants::padding,
          /* width:   */ (sink.data_[j].timestamp - sink.data_[i].timestamp) * 
                           scale,
          /* height:  */ constants::box_height, 
          /* colors:  */ constants::svg_func_colors[
                           fmgot->second.second % 
                             constants::svg_num_func_colors], 
          /*ap_name:  */ sink.data_[i].name,
          /*ap_tid:   */ tid,
          /*ap_start: */ (sink.data_[i].timestamp-min)*1000,
          /*ap_stop:  */ (sink.data_[j].timestamp-min)*1000,
          /*ap_dur:   */ ((sink.data_[j].timestamp) -
                           (sink.data_[i].timestamp))*1000);
        break;
      }
    }
  }

  fprintf(f, "%s\n", constants::svg_footer);
  if(0 != fclose(f)) {
    AURA_ERROR("Unable to close file.");
  }
}
  

} // namespace profile

} // namespace aura

#endif // AURA_MISC_PROFILE_SVG_HPP

