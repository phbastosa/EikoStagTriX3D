# include "admin.hpp"

bool str2bool(std::string s)
{
    bool b;

    std::for_each(s.begin(), s.end(), [](char & c){c = ::tolower(c);});
    std::istringstream(s) >> std::boolalpha >> b;

    return b;
}

void import_binary_float(std::string path, float * array, int n)
{
    std::ifstream file(path, std::ios::in);

    if (!file.is_open())
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");
    
    file.read((char *) array, n * sizeof(float));
    
    file.close();    
}

void export_binary_float(std::string path, float * array, int n)
{
    std::ofstream file(path, std::ios::out);
    
    if (!file.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");

    file.write((char *) array, n * sizeof(float));

    std::cout<<"\nBinary file \033[34m" + path + "\033[0;0m was successfully written."<<std::endl;

    file.close();
}

void import_text_file(std::string path, std::vector<std::string> &elements)
{
    std::ifstream file(path, std::ios::in);
    
    if (!file.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");

    std::string line;
    while(getline(file, line))
        if (line[0] != '#') elements.push_back(line);

    file.close();
}

std::string catch_parameter(std::string target, std::string file)
{
    char spaces = ' ';
    char comment = '#';

    std::string line;
    std::string variable;

    std::ifstream parameters(file);

    if (!parameters.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + file + "\033[0;0m could not be opened!");

    while (getline(parameters, line))
    {           
        if ((line.front() != comment) && (line.front() != spaces))        
        {
            if (line.find(target) == 0)
            {
                for (int i = line.find("=")+2; i < line.size(); i++)
                {    
                    if (line[i] == '#') break;
                    variable += line[i];            
                }

                break;
            }
        }                 
    }
    
    parameters.close();

    variable.erase(remove(variable.begin(), variable.end(), ' '), variable.end());

    return variable;
}

std::vector<std::string> split(std::string s, char delimiter)
{
    std::string token;
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);

    while (getline(tokenStream, token, delimiter)) 
        tokens.push_back(token);
   
    return tokens;
}

float sinc(float x) 
{
    if (fabsf(x) < 1e-8f) return 1.0f;
    return sinf(M_PI * x) / (M_PI * x);
}

float bessel_i0(float x) 
{
    float sum = 1.0f, term = 1.0f, k = 1.0f;
    
    while (term > 1e-10f) 
    {
        term *= (x / (2.0f * k)) * (x / (2.0f * k));
        sum += term;
        k += 1.0f;
    }

    return sum;
}

std::vector<std::vector<std::vector<float>>> kaiser_weights(float x, float y, float z, int ix0, int iy0, int iz0, float dx, float dy, float dz, float beta) 
{
    const int N = 5;
    float sum = 0.0f;

    std::vector<std::vector<std::vector<float>>> weights(N, std::vector<std::vector<float>>(N, std::vector<float>(N)));

    float rmax = 2.0f*sqrtf(dx*dx + dy*dy + dz*dz);
    float I0_beta = bessel_i0(beta);

    for (int i = 0; i < N; ++i) 
    {    
        float zi = (iz0 + i - 2) * dz;
        float dzr = (z - zi) / dz;
        
        for (int j = 0; j < N; ++j) 
        {
            float xj = (ix0 + j - 2) * dx;
            float dxr = (x - xj) / dx;

            for (int k = 0; k < N; ++k) 
            {
                float yk = (iy0 + k - 2) * dy;
                float dyr = (y - yk) / dy;

                float rz = z - zi;
                float rx = x - xj;
                float ry = y - yk;

                float r = sqrtf(rx * rx + ry * ry + rz * rz);

                float rnorm = 2.0f * r / rmax;

                float wij = 0.0f;
                if (rnorm <= 1.0f) 
                {
                    float arg = beta * sqrtf(1.0f - rnorm * rnorm);
                    wij = bessel_i0(arg) / I0_beta;
                }

                float sinc_term = sinc(dxr) * sinc(dyr) * sinc(dzr);

                weights[i][j][k] = sinc_term * wij;

                sum += weights[i][j][k];
            }
        }
    }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                weights[i][j][k] /= sum;

    return weights;
}

std::vector<std::vector<std::vector<float>>> gaussian_weights(float x, float y, float z, int ix0, int iy0, int iz0, float dx, float dy, float dz)
{
    const int N = 5;
    float sum = 0.0f;

    std::vector<std::vector<std::vector<float>>> weights(N, std::vector<std::vector<float>>(N, std::vector<float>(N)));

    float rmax = 2.0f*sqrtf(dx*dx + dy*dy + dz*dz);

    for (int i = 0; i < N; ++i) 
    {    
        float zi = (iz0 + i - 2) * dz;
        
        for (int j = 0; j < N; ++j) 
        {
            float xj = (ix0 + j - 2) * dx;

            for (int k = 0; k < N; ++k) 
            {
                float yk = (iy0 + k - 2) * dy;

                float rz = z - zi;
                float rx = x - xj;
                float ry = y - yk;

                float r = sqrtf(rx * rx + ry * ry + rz * rz) / rmax;

                weights[i][j][k] = 1.0f/sqrtf(2.0f*M_PI)*expf(-0.5f*r*r);

                sum += weights[i][j][k];
            }
        }
    }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                weights[i][j][k] /= sum;

    return weights;
}
