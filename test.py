from coremltools.converters.mil.mil import Builder as mb, types
import coremltools as ct

import numpy as np


x = np.array([1, 3, 5, 5, 3], dtype=np.int32)


@mb.program(input_specs=[mb.TensorSpec(shape=x.shape, dtype=types.int32)])
def prog(x):

    # Sort input
    indices = mb.argsort(x=x, ascending=True)
    x_sorted = mb.gather_along_axis(x=x, indices=indices)

    # Subtract nth+1 element from nth element
    x_sorted = mb.cast(x=x_sorted, dtype="fp16")
    x_sorted_shifted  = mb.pad(x=x_sorted, pad=[1, 0], constant_val=np.float16(-np.inf))
    x_sorted_padded = mb.pad(x=x_sorted, pad=[0, 1], mode="replicate")
    diff = mb.sub(x=x_sorted_padded, y=x_sorted_shifted)

    non_zero_indices = mb.non_zero(x=diff)
    unique_values = mb.gather(x=x_sorted, indices=non_zero_indices)
    unique_values = mb.squeeze(x = unique_values)


    # Loop variable initilization:
    #     (i, length, counts) = (0, len(unique_values), [])
    i = mb.const(val=[0])
    length = mb.shape(x=unique_values)
    counts = mb.cast(x=mb.const(val=[]), dtype="int32")

    def cond(i, length, _):
        # return i < length
        return mb.less(x=i, y=length)

    def body(i, length, counts):
        # i++
        i = mb.add(x=i, y=mb.const(val=[1]))

        counts = mb.concat(values=(counts, mb.const(val=[99])), axis=-1)

        return (i, length, counts)

    return mb.while_loop(_cond=cond, _body=body, loop_vars=(i, length, counts))


m = ct.convert(prog, source="milinternal")
print(m.predict({'x': x}))
