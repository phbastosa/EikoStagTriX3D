# ifndef ADMIN_HPP
# define ADMIN_HPP

# include <cmath>
# include <chrono>
# include <string>
# include <vector>
# include <iomanip>
# include <sstream>
# include <fstream>
# include <iostream>
# include <algorithm>

bool str2bool(std::string s);

void import_binary_float(std::string path, float * array, int n);
void export_binary_float(std::string path, float * array, int n);

void import_text_file(std::string path, std::vector<std::string> &elements);

std::string catch_parameter(std::string target, std::string file);

std::vector<std::string> split(std::string s, char delimiter);

std::vector<std::vector<std::vector<float>>> kaiser_weights(float x, float y, float z, int ix0, int iy0, int iz0, float dx, float dy, float dz); 
std::vector<std::vector<std::vector<float>>> gaussian_weights(float x, float y, float z, int ix0, int iy0, int iz0, float dx, float dy, float dz); 

# endif