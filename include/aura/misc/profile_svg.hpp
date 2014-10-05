#ifndef AURA_MISC_PROFILE_SVG_HPP
#define AURA_MISC_PROFILE_SVG_HPP

#include <stdio.h>
#include <map>
#include <utility>
#include <cstring>
#include <tuple>

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

struct func_entry {
  func_entry(const char * name, std::size_t thread_id, double timestamp,
    double duration, std::size_t depth) : name(name), thread_id(thread_id), 
      timestamp(timestamp), duration(duration), depth(depth) {}
  const char * name;
  std::size_t thread_id;
  double timestamp;
  double duration;
  std::size_t depth;
};

inline void dump_svg(memory_sink & sink, const char * filename) {
std::lock_guard<std::mutex> guard(sink.mtx_);
  if(sink.data_.size() < 1 || sink.data_.size()%2 != 0) {
    return;
  }

  // maximum and minimum
  double min = sink.data_[0].timestamp;
  double max = min;

  // map to normalize thread_id
  std::map<std::size_t, std::size_t> threadmap;
  
  // depth fields
  std::vector<std::size_t> func_depths;
  std::vector<std::size_t> max_depths;

  // unique functions handling 
  typedef std::tuple<std::size_t, std::size_t, const char *> t_unique_funcs_key;
  typedef std::map<t_unique_funcs_key, std::size_t> t_unique_funcs;
  t_unique_funcs unique_funcs;
  std::vector<std::size_t> num_unique_funcs;

  // field of functions
  

  std::vector<func_entry> funcs;
  funcs.reserve(sink.data_.size()/2);

  // overall statistics: map function -> (calls, duration)
  std::map<const char *, std::pair<std::size_t, double> > overall_stats;

  for(std::size_t i=0; i<sink.data_.size(); i++) {

    // min max
    if(min > sink.data_[i].timestamp) {
      min = sink.data_[i].timestamp;
    }
    if(max < sink.data_[i].timestamp) {
      max = sink.data_[i].timestamp;
    }

    // normalize thread_id and initialize func and depth maps
    std::map<std::size_t, std::size_t>::const_iterator tmgot = 
      threadmap.find(sink.data_[i].thread_id);
    if(tmgot == threadmap.end()) {
      threadmap.insert(std::pair<std::size_t, std::size_t>(
        sink.data_[i].thread_id, threadmap.size()));
      func_depths.push_back(0);
      max_depths.push_back(0);
      num_unique_funcs.push_back(0);
    }
    std::size_t tid = threadmap[sink.data_[i].thread_id];

    // handle depth
    if(sink.data_[i].start) {
      func_depths[tid]++;
      if(max_depths[tid] < func_depths[tid]) {
        max_depths[tid] = func_depths[tid];
      }
    } else {
      func_depths[tid]--;  
    }

    // find the stop value and build funcs
    if(sink.data_[i].start) {
      std::size_t j = i+1;
      for(; j<sink.data_.size(); j++) {
        if(sink.data_[j].thread_id == sink.data_[i].thread_id && 
          0 == strcmp(sink.data_[j].name, sink.data_[i].name) &&
          false == sink.data_[j].start) {
          funcs.push_back(func_entry(sink.data_[i].name, tid, 
            sink.data_[i].timestamp, 
            sink.data_[j].timestamp-sink.data_[i].timestamp,
            func_depths[tid]));
          break;
        }
      }
      // add unique function
      t_unique_funcs::const_iterator ugot = 
        unique_funcs.find(t_unique_funcs_key(tid, 
          func_depths[tid], sink.data_[i].name));
      if(ugot == unique_funcs.end()) {
        num_unique_funcs[tid]++;
        unique_funcs.insert(std::pair<t_unique_funcs_key, std::size_t>(
          t_unique_funcs_key(tid, func_depths[tid], 
            sink.data_[i].name), num_unique_funcs[tid]));
      }
      // add overall statistics information
      std::map<const char *, std::pair<std::size_t, double> >::iterator 
        sgot = overall_stats.find(sink.data_[i].name);
      if(sgot == overall_stats.end()) {
        overall_stats.insert(
          std::pair<const char *, std::pair<std::size_t, double> >(
            sink.data_[i].name, std::pair<std::size_t, double>(1, 
              sink.data_[j].timestamp-sink.data_[i].timestamp)));
      } else {
        sgot->second.first++;
        sgot->second.second+=sink.data_[j].timestamp-sink.data_[i].timestamp;
      }

    }
  }

  // print stats
  for(std::map<const char *, std::pair<std::size_t, double> >::iterator 
    smit = overall_stats.begin(); 
    smit != overall_stats.end(); ++smit) {
    printf("%s: %0.9f (%lu)\n", 
      smit->first, smit->second.second, smit->second.first);
  }

  // postprocess max_depth field to calculate base offset for entry
  std::vector<size_t> max_depths_bk = max_depths;
  max_depths[0] = 0; 
  for(std::size_t i=1; i<max_depths.size(); i++) {
    max_depths[i] = max_depths[i-1] + max_depths_bk[i-1];
  }

  // output to file
  double scale = (double)constants::graph_width / (max-min);
  
  FILE * f = fopen(filename, "w");
  if(f == NULL) {
    AURA_ERROR("Unable to open file.");
  }
  
  fprintf(f, "%s\n", constants::svg_header);
 
  std::size_t backdrop_counter = 0;
  for(std::size_t i=0; i<funcs.size(); i++) {
    std::size_t tid = funcs[i].thread_id;
    std::size_t level = max_depths[tid] + funcs[i].depth -1;
    std::size_t color = unique_funcs[t_unique_funcs_key(
      tid, funcs[i].depth, funcs[i].name)];
   
    if(backdrop_counter == tid) {
      fprintf(f, "<rect x=\"0\" y=\"%ld\" width=\"%d\" height=\"%ld\" " 
        "fill=\"%s\" stroke-width=\"0\"/>\n",
        /* y:      */ max_depths[tid]*constants::box_height + 
                        max_depths[tid]*constants::padding + 
                        tid*constants::padding,
        /* width:  */ constants::graph_width,
        /* height: */ max_depths_bk[tid] * 
                        (constants::box_height+constants::padding) + 
                        constants::padding,
        /* fill:   */ constants::svg_thread_colors[
                        tid % constants::svg_num_thread_colors]);
      backdrop_counter++;
    }

    
    fprintf(f, "<rect class=\"m\" x=\"%f\" y=\"%ld\" width=\"%f\" " 
      "height=\"%d\" fill=\"%s\" stroke-width=\"0\" " 
      "ap_name=\"%s\" ap_tid=\"%ld\" ap_dur=\"%0.9f\" ap_sta=\"%0.9f\"/>\n",
      /* x:       */ (funcs[i].timestamp - min)*scale, 
      /* y:       */ level*constants::box_height + 
                       (level+1)*constants::padding + 
                       tid*constants::padding,
      /* width:   */ (funcs[i].duration) * scale,
      /* height:  */ constants::box_height, 
      /* colors:  */ constants::svg_func_colors[
                       color % constants::svg_num_func_colors], 
      /*ap_name:  */ funcs[i].name,
      /*ap_tid:   */ tid,
      /*ap_dur:   */ funcs[i].duration*1000,
      /*ap_sta:   */ funcs[i].timestamp*1000
      );
  }
  
  fprintf(f, "%s\n", constants::svg_footer);
  if(0 != fclose(f)) {
    AURA_ERROR("Unable to close file.");
  }
}
  

} // namespace profile

} // namespace aura

#endif // AURA_MISC_PROFILE_SVG_HPP

