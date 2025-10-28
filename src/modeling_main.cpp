# include <sys/resource.h>

# include "modeling/triclinic_ssg.cuh"
# include "modeling/triclinic_rsg.cuh"

int main(int argc, char **argv)
{
    std::vector<Triclinic *> modeling = 
    {
        new Triclinic_SSG(),
        new Triclinic_RSG()
    };

    auto file = std::string(argv[1]);
    auto type = std::stoi(catch_parameter("modeling_type", file));    

    struct rusage usage_init;
    getrusage(RUSAGE_SELF, &usage_init);

    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    size_t used_init = total_byte - free_byte;

    modeling[type]->parameters = file;

    modeling[type]->set_parameters();

    auto ti = std::chrono::system_clock::now();

    modeling[type]->run_wave_propagation();

    auto tf = std::chrono::system_clock::now();

    struct rusage usage_total;
    getrusage(RUSAGE_SELF, &usage_total);
    size_t rss = usage_total.ru_maxrss - usage_init.ru_maxrss;

    cudaMemGetInfo(&free_byte, &total_byte);
    size_t used = (total_byte - free_byte) - used_init;

    std::cout << "\nCurrent RAM usage: " << rss / 1024 << " MB";
    std::cout << "\nCurrent GPU usage: " << used / 1024 / 1024 << " MB\n";

    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;
    
    return 0;
}