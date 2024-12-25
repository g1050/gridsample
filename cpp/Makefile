# 定义编译器
CXX = g++
# 定义编译选项
CXXFLAGS = -Wall -std=c++11
# 定义目标可执行文件
TARGET = main
# 定义源文件
SRCS = main.cpp gridsample.cpp
# 定义头文件
HEADERS = gridsample.h
# 定义对象文件
OBJS = $(SRCS:.cpp=.o)

# 默认目标，编译和链接
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# 编译 .cpp 文件为 .o 文件
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理目标，删除生成的中间文件和可执行文件
clean:
	rm -f $(OBJS) $(TARGET)

# 提供一个 phony target 以防止与文件名冲突
.PHONY: all clean
