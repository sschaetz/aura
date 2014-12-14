#include <future>
#include <tuple>
#include <iostream>

/*
 * Question: 
 * can we nest futures?
 * can we mix nesting and sharing?
 *
 * Answers:
 * apparently yes, both seems to work
 */

int main(void) {
	auto ftr0 = std::async(std::launch::async, []() {
		std::cout << "outer ftr" << std::endl;
		int i = 0;
		auto ftr1 = std::async(std::launch::async, []() {
			std::cout << "inner ftr" << std::endl;
		});
		return std::make_tuple(std::move(ftr1), i);
	});
	auto sftr0 = ftr0.share();
	
	auto ftr2 = std::async(std::launch::async, [&]() {
		std::cout << std::get<1>(sftr0.get()) << std::endl;
	});

	std::get<0>(sftr0.get()).wait();
}

