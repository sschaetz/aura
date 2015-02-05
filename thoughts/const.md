# Const

Adding const-correctness would be very nice to help catch bugs from
unintentional modifications.


For this, we need const and non-const versions of a lot of member functions.
Fortunately, there seem to be no functions with more than one line of code
that need to be duplicated. 

There are, however, some non-obvious things:

## `device`

The `device` type represents an accelerator and things related to it. That is
why special care needs to be taken with instances of this type. 

The `device` also holds a list of all `module`s that were compiled for it, and
we can create new modules by calling the member functions
`load_from_{string,file}`. Here is were the non-obviousness arises:
These functions obviously change the `device`, since they change its list of
`module`s. So we need to take care where we get our references to a device. 
If we have, for example, a `const device_array<t>` and we call get_device on
it, we cannot call `load_from_string` on that device!


But calling `load_from_string` on a device optained from a `device_array` is
an idiom used in a lot of the math functions. So even though all of the
`device_array` arguments to these math functions are (and need to be) on the
same device, we can only load new modules from the NON-const `device_array`s
passed to the math function. Fortunately, since every kernel needs to modify
SOMETHING on the device (at least we think so for now), there will always be
at least one non-const argument from which a non-const reference to the
underlying device can be extracted, which can then be used to get modules by
`load_from_string`.
