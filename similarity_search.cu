#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDimension = 384;
constexpr int kBlockSize = 256;
constexpr int kRepetitions = 5;

// The CPU and GPU both accumulate float32 values, but the GPU may use fused
// multiply-add instructions. An absolute tolerance of 1e-5 allows for those
// small rounding differences for 384-term dot products.
constexpr float kAbsoluteTolerance = 1.0e-5f;

void check_cuda(cudaError_t result, const char* expression, const char* file,
                int line) {
    if (result != cudaSuccess) {
        std::ostringstream message;
        message << "CUDA error at " << file << ':' << line << " for "
                << expression << ": " << cudaGetErrorString(result);
        throw std::runtime_error(message.str());
    }
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

// Each thread independently processes one document and writes one score. The
// complete dot product is accumulated by that thread without shared memory or
// a parallel reduction.
__global__ void cosine_similarity_kernel(const float* documents,
                                         const float* query, float* scores,
                                         int document_count, int dimension) {
    const int document_index = blockIdx.x * blockDim.x + threadIdx.x;

    // The final block is usually only partly occupied (for example, 1000 is
    // not divisible by the block size 256), so out-of-range threads must stop.
    if (document_index >= document_count) {
        return;
    }

    float sum = 0.0f;
    const std::size_t row_start =
        static_cast<std::size_t>(document_index) * dimension;
    for (int column = 0; column < dimension; ++column) {
        sum += documents[row_start + column] * query[column];
    }
    scores[document_index] = sum;
}

bool normalize_vector(float* values, int dimension) {
    double squared_norm = 0.0;
    for (int column = 0; column < dimension; ++column) {
        const double value = static_cast<double>(values[column]);
        squared_norm += value * value;
    }

    // A zero vector has no cosine direction. Leaving it as all zeros is safe
    // and gives it a dot-product score of zero without division by zero.
    if (squared_norm == 0.0) {
        return false;
    }

    const float inverse_norm =
        static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (int column = 0; column < dimension; ++column) {
        values[column] *= inverse_norm;
    }
    return true;
}

void normalize_rows(std::vector<float>& documents, int document_count,
                    int dimension) {
    for (int document = 0; document < document_count; ++document) {
        normalize_vector(documents.data() +
                             static_cast<std::size_t>(document) * dimension,
                         dimension);
    }
}

void generate_data(int document_count, int dimension,
                   std::vector<float>& documents, std::vector<float>& query) {
    documents.resize(static_cast<std::size_t>(document_count) * dimension);
    query.resize(dimension);

    // A fixed seed makes every run generate the same input for a given size.
    std::mt19937 generator(202603u + static_cast<unsigned>(document_count));
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float& value : documents) {
        value = distribution(generator);
    }
    for (float& value : query) {
        value = distribution(generator);
    }

    // Deliberately include one zero document to exercise safe normalization.
    std::fill(documents.begin(), documents.begin() + dimension, 0.0f);
    normalize_rows(documents, document_count, dimension);
    if (!normalize_vector(query.data(), dimension)) {
        // Keep the input valid if a future generator produces a zero query.
        query[0] = 1.0f;
    }
}

void cpu_similarity_float(const std::vector<float>& documents,
                          const std::vector<float>& query,
                          std::vector<float>& scores, int document_count,
                          int dimension) {
    scores.resize(document_count);
    for (int document = 0; document < document_count; ++document) {
        float sum = 0.0f;
        const std::size_t row_start =
            static_cast<std::size_t>(document) * dimension;
        for (int column = 0; column < dimension; ++column) {
            sum += documents[row_start + column] * query[column];
        }
        scores[document] = sum;
    }
}

void cpu_similarity_double(const std::vector<float>& documents,
                           const std::vector<float>& query,
                           std::vector<double>& scores, int document_count,
                           int dimension) {
    scores.resize(document_count);
    for (int document = 0; document < document_count; ++document) {
        double sum = 0.0;
        const std::size_t row_start =
            static_cast<std::size_t>(document) * dimension;
        for (int column = 0; column < dimension; ++column) {
            sum += static_cast<double>(documents[row_start + column]) *
                   static_cast<double>(query[column]);
        }
        scores[document] = sum;
    }
}

struct ErrorSummary {
    double gpu_vs_cpu_float = 0.0;
    double gpu_vs_cpu_double = 0.0;
    double cpu_float_vs_cpu_double = 0.0;
};

ErrorSummary validate_scores(const std::vector<float>& gpu_scores,
                             const std::vector<float>& cpu_float_scores,
                             const std::vector<double>& cpu_double_scores) {
    if (gpu_scores.size() != cpu_float_scores.size() ||
        gpu_scores.size() != cpu_double_scores.size()) {
        throw std::runtime_error("Score arrays have different sizes.");
    }

    ErrorSummary errors;
    for (std::size_t index = 0; index < gpu_scores.size(); ++index) {
        if (!std::isfinite(gpu_scores[index])) {
            throw std::runtime_error("GPU produced a non-finite score.");
        }
        const double gpu_vs_float =
            std::abs(static_cast<double>(gpu_scores[index]) -
                     static_cast<double>(cpu_float_scores[index]));
        const double gpu_vs_double =
            std::abs(static_cast<double>(gpu_scores[index]) -
                     cpu_double_scores[index]);
        const double float_vs_double =
            std::abs(static_cast<double>(cpu_float_scores[index]) -
                     cpu_double_scores[index]);
        errors.gpu_vs_cpu_float =
            std::max(errors.gpu_vs_cpu_float, gpu_vs_float);
        errors.gpu_vs_cpu_double =
            std::max(errors.gpu_vs_cpu_double, gpu_vs_double);
        errors.cpu_float_vs_cpu_double =
            std::max(errors.cpu_float_vs_cpu_double, float_vs_double);

        if (gpu_vs_float > kAbsoluteTolerance) {
            std::ostringstream message;
            message << "Score mismatch at document " << index
                    << ": GPU=" << gpu_scores[index]
                    << ", CPU float=" << cpu_float_scores[index]
                    << ", absolute error=" << gpu_vs_float
                    << ", tolerance=" << kAbsoluteTolerance;
            throw std::runtime_error(message.str());
        }
    }
    return errors;
}

void launch_kernel(const float* device_documents, const float* device_query,
                   float* device_scores, int document_count, int dimension) {
    const int grid_size =
        (document_count + kBlockSize - 1) / kBlockSize;
    cosine_similarity_kernel<<<grid_size, kBlockSize>>>(
        device_documents, device_query, device_scores, document_count,
        dimension);
    // Launch errors are asynchronous, so check the launch immediately. A later
    // synchronization or copy checks errors that happen while the kernel runs.
    CUDA_CHECK(cudaGetLastError());
}

std::vector<float> gpu_similarity_once(const std::vector<float>& documents,
                                       const std::vector<float>& query,
                                       int document_count, int dimension) {
    const std::size_t document_bytes = documents.size() * sizeof(float);
    const std::size_t query_bytes = query.size() * sizeof(float);
    const std::size_t score_bytes =
        static_cast<std::size_t>(document_count) * sizeof(float);

    float* device_documents = nullptr;
    float* device_query = nullptr;
    float* device_scores = nullptr;

    // Device memory is separate from normal CPU memory and must be allocated
    // explicitly before data can be transferred to the GPU.
    CUDA_CHECK(cudaMalloc(&device_documents, document_bytes));
    CUDA_CHECK(cudaMalloc(&device_query, query_bytes));
    CUDA_CHECK(cudaMalloc(&device_scores, score_bytes));

    // Copy the normalized inputs from host (CPU) memory to device (GPU) memory.
    CUDA_CHECK(cudaMemcpy(device_documents, documents.data(), document_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_query, query.data(), query_bytes,
                          cudaMemcpyHostToDevice));

    launch_kernel(device_documents, device_query, device_scores, document_count,
                  dimension);

    std::vector<float> scores(document_count);
    // This blocking copy waits for the kernel and returns its output to the CPU.
    CUDA_CHECK(cudaMemcpy(scores.data(), device_scores, score_bytes,
                          cudaMemcpyDeviceToHost));

    // Every cudaMalloc has a matching cudaFree once the buffers are no longer
    // needed. CUDA_CHECK also makes cleanup errors visible.
    CUDA_CHECK(cudaFree(device_scores));
    CUDA_CHECK(cudaFree(device_query));
    CUDA_CHECK(cudaFree(device_documents));
    return scores;
}

std::vector<int> top_k_indices(const std::vector<float>& scores, int k) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    k = std::min(k, static_cast<int>(indices.size()));
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&scores](int left, int right) {
                          if (scores[left] == scores[right]) {
                              return left < right;
                          }
                          return scores[left] > scores[right];
                      });
    indices.resize(k);
    return indices;
}

void run_correctness_checks() {
    std::cout << "\nCorrectness checks\n";

    // Tiny hand-checkable case. After normalization, the expected cosine
    // scores for query [1, 0] are [1, 0, 1/sqrt(2), -1, 0].
    constexpr int tiny_count = 5;
    constexpr int tiny_dimension = 2;
    std::vector<float> tiny_documents = {
        1.0f, 0.0f,  // same direction
        0.0f, 1.0f,  // perpendicular
        1.0f, 1.0f,  // 45 degrees
        -1.0f, 0.0f, // opposite direction
        0.0f, 0.0f   // zero vector stays zero
    };
    std::vector<float> tiny_query = {1.0f, 0.0f};
    normalize_rows(tiny_documents, tiny_count, tiny_dimension);
    normalize_vector(tiny_query.data(), tiny_dimension);

    std::vector<float> tiny_cpu;
    std::vector<double> tiny_double;
    cpu_similarity_float(tiny_documents, tiny_query, tiny_cpu, tiny_count,
                         tiny_dimension);
    cpu_similarity_double(tiny_documents, tiny_query, tiny_double, tiny_count,
                          tiny_dimension);
    const std::vector<float> tiny_gpu = gpu_similarity_once(
        tiny_documents, tiny_query, tiny_count, tiny_dimension);
    validate_scores(tiny_gpu, tiny_cpu, tiny_double);

    const std::vector<double> expected = {
        1.0, 0.0, 1.0 / std::sqrt(2.0), -1.0, 0.0};
    for (int index = 0; index < tiny_count; ++index) {
        if (std::abs(static_cast<double>(tiny_cpu[index]) - expected[index]) >
            kAbsoluteTolerance) {
            throw std::runtime_error("Tiny example did not match its expected score.");
        }
        std::cout << "  document " << index << ": expected " << expected[index]
                  << ", GPU " << tiny_gpu[index] << '\n';
    }

    // 1003 is intentionally not divisible by 256, which exercises the kernel's
    // bounds check in the final block on a realistic 384-dimensional input.
    constexpr int odd_document_count = 1003;
    std::vector<float> documents;
    std::vector<float> query;
    generate_data(odd_document_count, kDimension, documents, query);
    std::vector<float> cpu_float_scores;
    std::vector<double> cpu_double_scores;
    cpu_similarity_float(documents, query, cpu_float_scores, odd_document_count,
                         kDimension);
    cpu_similarity_double(documents, query, cpu_double_scores,
                          odd_document_count, kDimension);
    const std::vector<float> gpu_scores = gpu_similarity_once(
        documents, query, odd_document_count, kDimension);
    const ErrorSummary errors =
        validate_scores(gpu_scores, cpu_float_scores, cpu_double_scores);
    std::cout << "  1003-document bounds-check case passed.\n"
              << "  max |GPU - CPU float|  = " << errors.gpu_vs_cpu_float
              << '\n'
              << "  max |GPU - CPU double| = " << errors.gpu_vs_cpu_double
              << "\n";
}

struct BenchmarkRow {
    int document_count;
    int repetition;
    double cpu_ms;
    float gpu_kernel_ms;
    double transfer_plus_kernel_ms;
    ErrorSummary errors;
};

std::vector<BenchmarkRow> benchmark_size(int document_count) {
    std::cout << "\nBenchmarking " << document_count << " documents x "
              << kDimension << " dimensions\n";

    // Generation and CPU normalization happen before any timer starts.
    std::vector<float> documents;
    std::vector<float> query;
    generate_data(document_count, kDimension, documents, query);

    std::vector<float> cpu_scores(document_count);
    std::vector<double> cpu_double_scores;

    // CPU warm-up (unmeasured).
    cpu_similarity_float(documents, query, cpu_scores, document_count,
                         kDimension);
    cpu_similarity_double(documents, query, cpu_double_scores, document_count,
                          kDimension);

    const std::size_t document_bytes = documents.size() * sizeof(float);
    const std::size_t query_bytes = query.size() * sizeof(float);
    const std::size_t score_bytes =
        static_cast<std::size_t>(document_count) * sizeof(float);
    float* device_documents = nullptr;
    float* device_query = nullptr;
    float* device_scores = nullptr;

    // Allocation is intentionally outside both GPU timings.
    CUDA_CHECK(cudaMalloc(&device_documents, document_bytes));
    CUDA_CHECK(cudaMalloc(&device_query, query_bytes));
    CUDA_CHECK(cudaMalloc(&device_scores, score_bytes));

    // Put inputs on the device once for kernel-only measurements.
    CUDA_CHECK(cudaMemcpy(device_documents, documents.data(), document_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_query, query.data(), query_bytes,
                          cudaMemcpyHostToDevice));

    // GPU warm-up (unmeasured) initializes the CUDA execution path.
    launch_kernel(device_documents, device_query, device_scores, document_count,
                  kDimension);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(cpu_scores.data(), device_scores, score_bytes,
                          cudaMemcpyDeviceToHost));

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    std::vector<BenchmarkRow> rows;
    rows.reserve(kRepetitions);
    std::vector<float> gpu_scores(document_count);

    for (int repetition = 1; repetition <= kRepetitions; ++repetition) {
        // CPU time covers only the sequential float32 dot products.
        const auto cpu_start = std::chrono::steady_clock::now();
        cpu_similarity_float(documents, query, cpu_scores, document_count,
                             kDimension);
        const auto cpu_stop = std::chrono::steady_clock::now();
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_stop - cpu_start)
                .count();

        // CUDA events live on the device timeline and measure only the kernel
        // with inputs already resident in GPU memory.
        CUDA_CHECK(cudaEventRecord(kernel_start));
        launch_kernel(device_documents, device_query, device_scores,
                      document_count, kDimension);
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));
        float gpu_kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, kernel_start,
                                        kernel_stop));

        // This separate wall-clock measurement includes H2D input transfers,
        // the kernel launch/execution, and the D2H score transfer. The final
        // blocking cudaMemcpy waits for the kernel to finish.
        const auto total_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(device_documents, documents.data(), document_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_query, query.data(), query_bytes,
                              cudaMemcpyHostToDevice));
        launch_kernel(device_documents, device_query, device_scores,
                      document_count, kDimension);
        CUDA_CHECK(cudaMemcpy(gpu_scores.data(), device_scores, score_bytes,
                              cudaMemcpyDeviceToHost));
        const auto total_stop = std::chrono::steady_clock::now();
        const double transfer_plus_kernel_ms =
            std::chrono::duration<double, std::milli>(total_stop - total_start)
                .count();

        const ErrorSummary errors =
            validate_scores(gpu_scores, cpu_scores, cpu_double_scores);
        rows.push_back({document_count, repetition, cpu_ms, gpu_kernel_ms,
                        transfer_plus_kernel_ms, errors});
    }

    ErrorSummary maximum_errors;
    for (const BenchmarkRow& row : rows) {
        maximum_errors.gpu_vs_cpu_float = std::max(
            maximum_errors.gpu_vs_cpu_float,
            row.errors.gpu_vs_cpu_float);
        maximum_errors.gpu_vs_cpu_double = std::max(
            maximum_errors.gpu_vs_cpu_double,
            row.errors.gpu_vs_cpu_double);
        maximum_errors.cpu_float_vs_cpu_double = std::max(
            maximum_errors.cpu_float_vs_cpu_double,
            row.errors.cpu_float_vs_cpu_double);
    }
    std::cout << "  Maximum absolute errors across repetitions:\n"
              << "    GPU vs CPU float  = "
              << maximum_errors.gpu_vs_cpu_float << '\n'
              << "    GPU vs CPU double = "
              << maximum_errors.gpu_vs_cpu_double << '\n'
              << "    CPU float vs CPU double = "
              << maximum_errors.cpu_float_vs_cpu_double << '\n';

    // Top-k runs on the CPU after all timers have stopped.
    const std::vector<int> top_five = top_k_indices(gpu_scores, 5);
    std::cout << "  Top five document IDs and scores:";
    for (int index : top_five) {
        std::cout << " (" << index << ", " << gpu_scores[index] << ')';
    }
    std::cout << '\n';

    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaFree(device_scores));
    CUDA_CHECK(cudaFree(device_query));
    CUDA_CHECK(cudaFree(device_documents));
    return rows;
}

void write_csv(const std::string& path,
               const std::vector<BenchmarkRow>& rows) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not open CSV output: " + path);
    }
    output << "document_count,dimension,repetition,cpu_ms,gpu_kernel_ms,"
              "transfer_plus_kernel_ms,max_abs_gpu_vs_cpu_float,"
              "max_abs_gpu_vs_cpu_double,max_abs_cpu_float_vs_cpu_double\n";
    output << std::setprecision(9);
    for (const BenchmarkRow& row : rows) {
        output << row.document_count << ',' << kDimension << ','
               << row.repetition << ',' << row.cpu_ms << ','
               << row.gpu_kernel_ms << ',' << row.transfer_plus_kernel_ms
               << ',' << row.errors.gpu_vs_cpu_float << ','
               << row.errors.gpu_vs_cpu_double << ','
               << row.errors.cpu_float_vs_cpu_double << '\n';
    }
}

void require_cuda_device() {
    int device_count = 0;
    const cudaError_t result = cudaGetDeviceCount(&device_count);
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA is unavailable: ") + cudaGetErrorString(result) +
            ". In Colab, choose Runtime > Change runtime type > GPU and rerun.");
    }
    if (device_count == 0) {
        throw std::runtime_error(
            "No CUDA GPU was found. In Colab, choose Runtime > Change runtime "
            "type > GPU and rerun. No CPU-only fallback is used.");
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "Using CUDA device: " << properties.name << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool correctness_only = false;
        std::string output_path = "benchmark_results.csv";
        for (int argument = 1; argument < argc; ++argument) {
            const std::string value = argv[argument];
            if (value == "--correctness-only") {
                correctness_only = true;
            } else if (value == "--output" && argument + 1 < argc) {
                output_path = argv[++argument];
            } else {
                throw std::runtime_error(
                    "Usage: ./similarity_search [--correctness-only] "
                    "[--output PATH]");
            }
        }

        require_cuda_device();
        std::cout << std::fixed << std::setprecision(7);
        run_correctness_checks();
        if (correctness_only) {
            std::cout << "\nCorrectness-only run completed successfully.\n";
            return 0;
        }

        std::vector<BenchmarkRow> all_rows;
        for (int document_count : {1000, 10000, 100000}) {
            std::vector<BenchmarkRow> rows = benchmark_size(document_count);
            all_rows.insert(all_rows.end(), rows.begin(), rows.end());
        }
        write_csv(output_path, all_rows);
        std::cout << "\nAll GPU scores passed the absolute tolerance of "
                  << kAbsoluteTolerance << ".\n"
                  << "Wrote " << all_rows.size() << " measured rows to "
                  << output_path << ".\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
