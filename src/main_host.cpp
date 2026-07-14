/*
 * main_host.cpp — Linux host inference runner for ExecuTorch
 *
 * Loads a .pte model from disk, runs inference, and prints results.
 * Use this to validate the model and ExecuTorch runtime on the host
 * before deploying to the STM32.
 *
 * Usage:
 *   ./build/host/inference_host models/tiny_model.pte
 */

#include <cstdio>
#include <cstring>

#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/executor/method.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/portable_type/tensor_impl.h"
#include "executorch/runtime/core/hierarchical_allocator.h"
#include "executorch/runtime/platform/runtime.h"
#include "executorch/extension/data_loader/file_data_loader.h"

using namespace executorch::runtime;
using namespace executorch::extension;
using namespace executorch::runtime::etensor;

static uint8_t method_pool[64 * 1024];
static uint8_t planned_pool[64 * 1024];

int main(int argc, char** argv) {
    const char* pte_path = (argc > 1) ? argv[1] : "models/tiny_model.pte";

    runtime_init();

    // Host uses FileDataLoader — reads directly from disk
    Result<FileDataLoader> loader_result = FileDataLoader::from(pte_path);
    if (!loader_result.ok()) {
        fprintf(stderr, "ERROR: Could not open model: %s\n", pte_path);
        return 1;
    }
    FileDataLoader loader = std::move(loader_result.get());

    Result<Program> program_result = Program::load(&loader);
    if (!program_result.ok()) {
        fprintf(stderr, "ERROR: Program::load failed (code %d)\n",
                (int)program_result.error());
        return 1;
    }
    Program program = std::move(program_result.get());

    MemoryAllocator method_allocator(sizeof(method_pool), method_pool);
    Span<uint8_t> planned_span(planned_pool, sizeof(planned_pool));
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(&method_allocator, &planned_allocator);

    Result<Method> method_result = program.load_method("forward", &memory_manager);
    if (!method_result.ok()) {
        fprintf(stderr, "ERROR: load_method failed (code %d)\n",
                (int)method_result.error());
        return 1;
    }
    Method method = std::move(method_result.get());

    // Input tensor [1, 4] of float32
    static float input_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    static TensorImpl::SizesType    sizes[2]     = {1, 4};
    static TensorImpl::DimOrderType dim_order[2] = {0, 1};
    static TensorImpl::StridesType  strides[2]   = {4, 1};

    TensorImpl input_impl(
        ScalarType::Float,
        /*dim=*/2,
        sizes,
        input_data,
        dim_order,
        strides
    );
    Tensor input_tensor(&input_impl);

    Error set_err = method.set_input(EValue(input_tensor), 0);
    if (set_err != Error::Ok) {
        fprintf(stderr, "ERROR: set_input failed\n");
        return 1;
    }

    Error exec_err = method.execute();
    if (exec_err != Error::Ok) {
        fprintf(stderr, "ERROR: execute() failed\n");
        return 1;
    }

    EValue output = method.get_output(0);
    Tensor out_tensor = output.toTensor();
    const float* out_data = out_tensor.const_data_ptr<float>();
    int64_t num_elements = out_tensor.numel();

    printf("Input:    ");
    for (int i = 0; i < 4; i++) printf("%.1f ", input_data[i]);
    printf("\n");

    printf("Output:   ");
    for (int64_t i = 0; i < num_elements; i++) printf("%.1f ", out_data[i]);
    printf("\n");

    printf("Expected: 3.0 5.0 7.0 9.0\n");

    bool correct = true;
    float expected[4] = {3.0f, 5.0f, 7.0f, 9.0f};
    for (int i = 0; i < 4; i++) {
        if (out_data[i] != expected[i]) {
            correct = false;
            fprintf(stderr, "MISMATCH at [%d]: got %.1f expected %.1f\n",
                    i, out_data[i], expected[i]);
        }
    }

    printf("%s\n", correct ? "PASS" : "FAIL");
    return correct ? 0 : 1;
}
