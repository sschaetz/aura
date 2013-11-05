#ifndef AURA_BACKEND_SHARED_DEVICE_INFO_HPP
#define AURA_BACKEND_SHARED_DEVICE_INFO_HPP

struct device_info {
  char name[300];
  char vendor[300];
  svec<std::size_t, AURA_MAX_GRID_DIMS> max_grid; 
  svec<std::size_t, AURA_MAX_BLOCK_DIMS> max_block; 
  std::size_t max_threads;
};

#endif // AURA_BACKEND_SHARED_DEVICE_INFO_HPP

