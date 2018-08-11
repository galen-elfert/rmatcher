all:
	g++ --std=c++11 rmatcher.cc `pkg-config --libs --cflags opencv` -o rmatcher
