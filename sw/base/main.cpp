#include <iostream>
#include <string>
#include <malloc.h>
#include <time.h> 
#include <sys/time.h>  
#include <chrono>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <signal.h> 
#ifdef EN_AVX
#include <x86intrin.h>
#endif
#include <boost/program_options.hpp>

#include "cBench.hpp"
#include "cProc.hpp"

using namespace std;
using namespace fpga;

/* Signal handler */
std::atomic<bool> stalled(false); 
void gotInt(int) {
    stalled.store(true);
}

/* Def params */
constexpr auto const targetRegion = 1;
constexpr auto const defHuge = true;
constexpr auto const nReps = 1;
constexpr auto const defSize = 512;
constexpr auto const nBenchRuns = 1;

/**
 * @brief Loopback example
 * 
 */
int main(int argc, char *argv[])  
{
    // ---------------------------------------------------------------
    // Init 
    // ---------------------------------------------------------------
    
    // Sighandler
    struct sigaction sa;
    memset( &sa, 0, sizeof(sa) );
    sa.sa_handler = gotInt;
    sigfillset(&sa.sa_mask);
    sigaction(SIGINT,&sa,NULL);

    // Read arguments
    boost::program_options::options_description programDescription("Options:");
    programDescription.add_options()
        ("target,t", boost::program_options::value<uint32_t>(), "Target vFPGA")
        ("huge,h", boost::program_options::value<bool>(), "Hugepages")
        ("reps,r", boost::program_options::value<uint32_t>(), "Number of repetitions")
        ("size,s", boost::program_options::value<uint32_t>(), "Transfer size")
        ("nbenchruns,n", boost::program_options::value<uint32_t>(), "Number of bench runs");
    
    boost::program_options::variables_map commandLineArgs;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, programDescription), commandLineArgs);
    boost::program_options::notify(commandLineArgs);

    uint32_t target_region = targetRegion;
    bool huge = defHuge;
    uint32_t n_reps = nReps;
    uint32_t size = defSize;
    uint32_t n_bench_runs = nBenchRuns;

    if(commandLineArgs.count("target") > 0) target_region = commandLineArgs["target"].as<uint32_t>();
    if(commandLineArgs.count("huge") > 0) huge = commandLineArgs["huge"].as<bool>();
    if(commandLineArgs.count("reps") > 0) n_reps = commandLineArgs["reps"].as<uint32_t>();
    if(commandLineArgs.count("size") > 0) size = commandLineArgs["size"].as<uint32_t>();
    if(commandLineArgs.count("nbenchruns") > 0) n_bench_runs = commandLineArgs["nbenchruns"].as<uint32_t>();

    uint32_t n_pages = huge ? ((size + hugePageSize - 1) / hugePageSize) : ((size + pageSize - 1) / pageSize);

    PR_HEADER("PARAMS");
    std::cout << "Target region: " << target_region << std::endl;
    std::cout << "Huge pages: " << huge << std::endl;
    std::cout << "Number of allocated pages: " << n_pages << std::endl;
    std::cout << "Number of repetitions: " << n_reps << std::endl;
    std::cout << "Transfer size: " << size << std::endl;
    std::cout << "Number of benchmark runs: " << n_bench_runs << std::endl;

    // ---------------------------------------------------------------
    // Handles
    // ---------------------------------------------------------------
    cProc cproc(target_region, getpid());
    uint64_t *hMem = (uint64_t*) cproc.getMem({huge ? CoyoteAlloc::HUGE_2M : CoyoteAlloc::REG_4K, n_pages});
    
    // ---------------------------------------------------------------
    // Runs 
    // ---------------------------------------------------------------
    cBench bench(n_bench_runs);
    uint32_t n_runs = 0;
    cproc.clearCompleted();
    
    // Throughput test
    auto benchmark_thr = [&]() {
        bool k = false;
        n_runs++;

        // Transfer the data
        for(int i = 0; i < n_reps; i++)
            cproc.invoke({CoyoteOper::TRANSFER, (void*) hMem, (void*) hMem, size, size, false, false});

        while(cproc.checkCompleted(CoyoteOper::TRANSFER) != n_reps * n_runs) { if( stalled.load() ) throw std::runtime_error("Stalled, SIGINT caught"); }
    };
    bench.runtime(benchmark_thr);
    PR_HEADER("THROUGHPUT");
    std::cout << "Throughput: " << (1000 * size) / (bench.getAvg() / n_reps) << " MB/s" << std::endl;
    
    n_runs = 0;
    cproc.clearCompleted();

    // Latency test
    auto benchmark_lat = [&]() {
        // Transfer the data
        for(int i = 0; i < n_reps; i++) {
                cproc.invoke({CoyoteOper::TRANSFER, (void*) hMem, (void*) hMem, size, size, true, false});
                while(cproc.checkCompleted(CoyoteOper::TRANSFER) != 1) { if( stalled.load() ) throw std::runtime_error("Stalled, SIGINT caught"); }                    
        }
    };
    bench.runtime(benchmark_lat);
    PR_HEADER("LATENCY");
    std::cout << "Latency: " << bench.getAvg() / (n_reps) << " ns" << std::endl;
    
    // ---------------------------------------------------------------
    // Release 
    // ---------------------------------------------------------------
    
    // Print status
    cproc.printDebug();
    
    return EXIT_SUCCESS;
}
