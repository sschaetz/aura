#ifndef AURA_BACKEND_SHARED_DEVICE_INFO_HPP
#define AURA_BACKEND_SHARED_DEVICE_INFO_HPP

struct device_info {
  char name[300];
  char vendor[300];
  svec<std::size_t, AURA_MAX_MESH_DIMS> max_mesh; 
  svec<std::size_t, AURA_MAX_BUNDLE_DIMS> max_bundle; 
  std::size_t max_fibers;
};

#endif // AURA_BACKEND_SHARED_DEVICE_INFO_HPP

