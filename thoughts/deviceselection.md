# Deviceselection

currently we only select devices by ordinal number
this has a number of problems, in particular we can never be sure
we select a GPU or a CPU.

Ideal would be to have a way of saying

* give me a GPU/accelerator in the system
* give me a device that supports double-precision
* give me the device with the biggest memory

and a combination of the above

In addition a device locking mechanism will be necessary to 
use a device exclusively

