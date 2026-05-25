cmake_minimum_required(VERSION 4.1)
project(ML_from_Scratch C)

set(CMAKE_C_STANDARD 11)

add_executable(Gaussian_Naive_Bayes Gaussian_Naive_Bayes/Gaussian_Naive_Bayes.c
        Multinomial_Naive_Bayes/Multinomail_Naive_Bayes.c
        Bernoulli_Naive_Bayes/Bernoulli_Naive_Bayes.cpp
        K-Mean/K-Mean.c
        Decision_Tree/Decision_Tree.c
        Random_Forest/Random_Forest.c
        DBSCAN.c
        Support_Vector_Machine/Support_Vector_Machine.cpp
        Support_Vector_Machine/Support_Vector_Machine.cpp)
target_link_libraries(Gaussian_Naive_Bayes m)


