
int main(void)
{
        boost::aura::initialize();
	boost::aura::device d(0);
	boost::aura::feed f(d);
	
	boost::aura::module mod = boost::aura::create_module_from_file(
			kernel_file, d, AURA_BACKEND_COMPILE_FLAGS);
	boost::aura::kernel k = boost::aura::create_kernel(mod, "scale");
	
	int x = 128; int y = 128;
	std::vector<float> hvec(product(bounds(x, y)), 42.);
	boost::aura::device_array<float> dvec(bounds(x, y), d);
	
	boost::aura::copy(hvec, dvec, f);
	boost::aura::invoke(k, mesh(y, x), bundle(x), 
			args(dvec.begin_ptr, .1), f);
	boost::aura::copy(dvec, hvec, f);
	boost::aura::wait_for(f);
	// hvec contains 4.2 in each element
}

