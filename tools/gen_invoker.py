grid = range(1, 4)
block = range(1, 4)
args = range(1, 11)

for g,b,a in [[g,b,a] for g in grid for b in block for a in args if g==b]:
  print("void invoke(kernel & k, typename grid_t<"+str(g)+">::type g,\n" 
    "  typename block_t<"+str(b)+">::type b,\n"
    "  typename args_t<"+str(a)+">::type a, feed & f) {\n"
    "  detail::invoke_impl<"+str(g)+", "+str(b)+", " 
    +str(a)+">(k, g, b, a, f);\n}\n")

       
