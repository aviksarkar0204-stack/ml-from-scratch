cmake_minimum_required(VERSION 4.1)
project(ML_from_Scratch C)

set(CMAKE_C_STANDARD 11)

add_executable(Gaussian_Naive_Bayes Gaussian_Naive_Bayes/Gaussian_Naive_Bayes.c
        Multinomial_Naive_Bayes/Multinomail_Naive_Bayes.c
        Bernoulli_Naive_Bayes/Bernoulli_Naive_Bayes.cpp
        K-Mean/K-Mean.c)
target_link_libraries(Gaussian_Naive_Bayes m)


