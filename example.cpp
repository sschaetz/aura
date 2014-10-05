
int main(void)
{
        aura::initialize();
	aura::device d(0);
	aura::feed f(d);
	
	aura::module mod = aura::create_module_from_file(
			kernel_file, d, AURA_BACKEND_COMPILE_FLAGS);
	aura::kernel k = aura::create_kernel(mod, "scale"); 
	
	int x = 128; int y = 128;
	std::vector<float> hvec(product(bounds(x, y)), 42.);
	aura::device_array<float> dvec(bounds(x, y), d);
	
	aura::copy(dvec, hvec, f);
	aura::invoke(k, mesh(y, x), bundle(x), 
			args(dvec.begin_ptr, .1), f);
	aura::copy(hvec, dvec, f);
	aura::wait_for(f);
	// hvec contains 4.2 in each element
}

