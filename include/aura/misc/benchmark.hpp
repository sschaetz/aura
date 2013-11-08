/*
 * Copyright (C) 2011-2013 Biomedizinische NMR Forschungs GmbH
 *         Author: Sebastian Schaetz <sschaet@gwdg.de>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *      (See accompanying file LICENSE_1_0.txt or copy at
 *            http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef AURA_MISC_BENCHMARK_HPP
#define AURA_MISC_BENCHMARK_HPP

#include <math.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <aura/misc/now.hpp>

namespace aura {

namespace details {


/**
 * @brief helper to print the histogram characters
 *
 * @param num number in current bucket
 * @param sum overall elements in buckets
 * @param chars number of characters to be drawn overall
 * @param c character to be drawn
 */
inline void print_histogram_helper(int num, int sum, int chars, char c) {
  int n = ceil((double)num/(double)sum * (double)chars);
  printf("\t");
  for(int i=0; i<n; i++) {
    printf("%c", c);
  }
  printf("\n");
}


/**
 * @brief function that returns the number of zeroes after the decimal point
 *
 * @param num number to analyze
 * @return number of zeroes after the decimal point
 */
inline int get_leading_zeroes(double num) {
  int zeroes = 0;
  num -= floor(num);
  for(double z = 0.1; z > 0.00000000001; z/=10) {
    if(num < z) {
      zeroes++;
    }
    else {
      break;
    }
  }
  return zeroes;
}


/**
 * @brief function that prints statistics and histogram of a double-vector
 *
 * @param vec the vector containing the data
 * @param bins the number of bins used in the histogram
 */
inline void print_histogram(std::vector<double> & vec, int bins) {
  double min = *std::min_element(vec.begin(), vec.end());
  double max = *std::max_element(vec.begin(), vec.end());
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  double mean = sum / vec.size();
  std::nth_element(vec.begin(), vec.begin()+vec.size()/2, vec.end());
  double median = vec[vec.size()/2];
  
  std::vector<double> diff(vec.size());
  std::transform(vec.begin(), vec.end(), diff.begin(),
    std::bind2nd(std::minus<double>(), mean));
  double sq_sum = std::inner_product(diff.begin(), 
    diff.end(), diff.begin(), 0.0);
  double sd = std::sqrt(sq_sum / vec.size());

  printf("\nSamples: %lu Min %f, Max %f\n", vec.size(), min, max);
  printf("Mean: %f, Median %f, SD %f\n", mean, median, sd); 
  
  std::vector<int> histogram(bins, 0);
  double binsize = (max - min) / (double) bins;
  for(std::vector<double>::iterator it = vec.begin(); it != vec.end(); it++) {
    int pos = floor((*it-min) / binsize);
    // handle the max value
    pos = (pos == bins ? pos-1 : pos);
    histogram[pos]++;
  }
  int width1 = 1 + ceil(log10(max));
  int width2 = 1 + ceil(log10(vec.size()));
  int signum = get_leading_zeroes(binsize);
  // force number to be at least 1
  signum = (signum == 0) ? 1 : signum;
  width1 += (signum+1);
  int b = 0;
  for(b=0; b<bins-1; b++) {
    printf("[%*.*f %*.*f [ %*d", 
      width1, signum, min+b*binsize, 
      width1, signum, min+(b+1)*binsize, 
      width2, histogram[b]);
    print_histogram_helper(histogram[b], vec.size(), 50, '|');
  }
  printf("[%*.*f %*.*f ] %*d", 
    width1, signum, min+b*binsize, 
    width1, signum, min+(b+1)*binsize, 
    width2, histogram[b]);
  print_histogram_helper(histogram[b], vec.size(), 50, '|');
  printf("\n");
}

} // namespace details 
} // namespace aura 

/**
 * @brief macro to benchmark an expression
 *
 * @param expression the expression that should be benchmarked
 * @param duration the amount of time the expression should execute
 * @param min min of test 
 * @param max max of test 
 * @param mean mean of test 
 * @param stdev standard deviation of test 
 * @param num number of test runs
 */
#define AURA_BENCHMARK(expression, duration, min, max, mean, stdev, num) {     \
  std::vector<double> measurements;                                            \
  double elapsed_time(0.), d2(0.);                                             \
  /* until time is not elapsed */                                              \
  num = 0;                                                                     \
  while(elapsed_time < duration) {                                             \
    double d1 = aura::now();                                                   \
    {                                                                          \
      d2 = aura::now();                                                        \
      expression;                                                              \
      d2 = aura::now() - d2;                                                   \
      num++;                                                                   \
    }                                                                          \
    d1 = aura::now() - d1;                                                     \
    if(d2 > 0 && d1 > 0) {                                                     \
      measurements.push_back(d2);                                              \
      elapsed_time += d1;                                                      \
    }                                                                          \
  }                                                                            \
  double sum = std::accumulate(measurements.begin(), measurements.end(), 0.0); \
  mean = sum / measurements.size();                                            \
  double sq_sum = std::inner_product(measurements.begin(), measurements.end(), \
    measurements.begin(), 0.0);                                                \
  stdev = std::sqrt(sq_sum / measurements.size() - mean * mean);               \
  std::sort(measurements.begin(), measurements.end());                         \
  min = measurements[0];                                                       \
  max = measurements[measurements.size()-1];                                   \
}                                                                              \
/**/

/**
 * @brief macro to benchmark an asynchronous expression
 *
 * @param expression the expression that should be benchmarked
 * @param sync epxression that should be used to synchronize 
 *   asynchronous expression
 * @param duration the amount of time the expression should execute
 * @param min min of test 
 * @param max max of test 
 * @param mean mean of test 
 * @param stdev standard deviation of test 
 * @param num number of test runs
 */
#define AURA_BENCHMARK_ASYNC(expression, sync, duration,                       \
    min, max, mean, stdev, num) {                                              \
  std::vector<double> measurements;                                            \
  double elapsed_time(0.), d2(0.);                                             \
  /* until time is not elapsed */                                              \
  num = 0;                                                                     \
  while(elapsed_time < duration) {                                             \
    double d1 = aura::now();                                                   \
    {                                                                          \
      d2 = aura::now();                                                        \
      expression;                                                              \
      sync;                                                                    \
      d2 = aura::now() - d2;                                                   \
      num++;                                                                   \
    }                                                                          \
    d1 = aura::now() - d1;                                                     \
    if(d2 > 0 && d1 > 0) {                                                     \
      measurements.push_back(d2);                                              \
      elapsed_time += d1;                                                      \
    }                                                                          \
  }                                                                            \
  double sum = std::accumulate(measurements.begin(), measurements.end(), 0.0); \
  mean = sum / measurements.size();                                            \
  double sq_sum = std::inner_product(measurements.begin(), measurements.end(), \
    measurements.begin(), 0.0);                                                \
  stdev = std::sqrt(sq_sum / measurements.size() - mean * mean);               \
  std::sort(measurements.begin(), measurements.end());                         \
  min = measurements[0];                                                       \
  max = measurements[measurements.size()-1];                                   \
}                                                                              \
/**/



/**
 * @brief macro to benchmark an expression with histogram output
 *
 * @param expression the expression that should be benchmarked
 * @param duration the amount of time the expression should execute
 * @param result variable the median runtime of expression is stored in
 */
#define AURA_BENCHMARK_HISTOGRAM(expression, duration, result) {               \
  std::vector<double> measurements;                                            \
  double elapsed_time(0.), d2(0.);                                             \
  /* until time is not elapsed and we have not an unequal number of  */        \
  /* measurements, the latter is important for the median */                   \
  while(elapsed_time < duration || measurements.size() % 2 == 0) {             \
    double d1 = aura::now();                                                   \
    {                                                                          \
      d2 = aura::now();                                                        \
      expression;                                                              \
      d2 = aura::now() - d2;                                                   \
    }                                                                          \
    d1 = aura::now() - d1;                                                     \
    if(d2 > 0 && d1 > 0) {                                                     \
      measurements.push_back(d2);                                              \
      elapsed_time += d1;                                                      \
    }                                                                          \
  }                                                                            \
  size_t n = measurements.size() / 2;                                          \
  std::nth_element(measurements.begin(), measurements.begin()+n,               \
    measurements.end());                                                       \
  result = measurements[n];                                                    \
  aura::details::print_histogram(measurements, 10);                            \
}                                                                              \
/**/


#endif // AURA_MISC_BENCHMARK_HPP
