# Aura Refactoring

## Basic Types

device
feed
library
kernel
device_ptr
device_array
mesh
bundle
fiber


## General methods
.get(): get backend handle
.get_{type}(): access aura object that the object relies on
    # feed.get_device() returns a device
    # to access the backend handle for device from feed: 
      f.get_device().get()
      f.get_device

