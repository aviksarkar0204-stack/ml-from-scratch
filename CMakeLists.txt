cmake_minimum_required(VERSION 4.1)
project(ML_from_Scratch C)

set(CMAKE_C_STANDARD 11)

add_executable(Gaussian_Naive_Bayes Gaussian_Naive_Bayes/Gaussian_Naive_Bayes.c)
target_link_libraries(Gaussian_Naive_Bayes m)


