TF_CFLAGS=-I/usr/local/lib/python2.7/dist-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0
TF_LFLAGS=-L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework

all:
	g++ -std=c++11 -shared zero_counter.cc -o zero_counter.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2

clean:
	rm -rf *.so
