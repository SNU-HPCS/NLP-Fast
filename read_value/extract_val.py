from __future__ import print_function

f=open("./predtrain.log", 'r')
while True:
  a=f.read(1)
  if a=='[':
    print(a,end='')
    while a!=']':
      a=f.read(1)
      print(a,end='')
    print('')
f.close()

