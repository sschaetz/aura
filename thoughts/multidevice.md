# Multiple Devices

How should multiple devices be handles?

Through new types?
multi_feed?
multi_device_vector?
multi_invoke?

or multi_feed and multi_device and all 
functions/types undstand them?

device_list, feed_list?

device_array v(size, device_list);

template <typename Segmenter>
device_array v(size s, Segmenter s, device_list dl);

for (feed& f : feed_list) {
	invoke(k, mesh(), bundle(), args(v.get_ptr(f)), f);
}

wait_for(feed_list);


