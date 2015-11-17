aulang
======

Thoughts about a kernel language that can be compiled to OpenCL, CUDA, Metal.
Called gold. File ending .au

Syntax
------

float32
float64
cfloat32
cfloat65

    host_callable 
    function1(arg1 [float32, readonly, global],
              arg2 [cfloat32, writeonly, global],
              meshsize [num_fibers_in_meshsize],
              meshid [fiber_in_mesh])
              => void
    {
        
    }


    host_callable 
    function1(const int* arg1 [float32, readonly, global],
              arg2 [cfloat32, writeonly, global],
              meshsize [num_fibers_in_meshsize],
              meshid [fiber_in_mesh])
              => void
    {
        
    }

