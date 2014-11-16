#ifndef AURA_BACKEND_SHARED_DEVICE_INFO_HPP
#define AURA_BACKEND_SHARED_DEVICE_INFO_HPP

struct device_info {
  char name[300];
  char vendor[300];
  svec<std::size_t, AURA_MAX_MESH_DIMS> max_mesh; 
  svec<std::size_t, AURA_MAX_BUNDLE_DIMS> max_bundle; 
  // max fibers per bundle
  std::size_t max_fibers_per_bundle;
};

/// print device info to stdout
inline void print_device_info(const device_info & di) {
  printf("%s (%s) max mesh size: ", di.name, di.vendor);
  for(std::size_t i=0; i<di.max_mesh.size(); i++) {
    printf("%lu ", di.max_mesh[i]);
  }
  printf("max bundle size: ");
  for(std::size_t i=0; i<di.max_bundle.size(); i++) {
    printf("%lu ", di.max_bundle[i]);
  }
  printf("max_fibers_per_bundle: %lu\n", di.max_fibers_per_bundle);
}

#endif // AURA_BACKEND_SHARED_DEVICE_INFO_HPP

