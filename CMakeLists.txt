cmake_minimum_required(VERSION 3.20)
project(stm32_executorch CXX C ASM)

# ------------------------------------------------------------------ #
#  Validation                                                          #
# ------------------------------------------------------------------ #
if(NOT DEFINED TARGET_PLATFORM)
    message(FATAL_ERROR
        "TARGET_PLATFORM must be set. Use a CMake preset:\n"
        "  cmake --preset host   (Linux debug)\n"
        "  cmake --preset stm32  (STM32F407 bare-metal)")
endif()

if(NOT DEFINED EXECUTORCH_BUILD_DIR)
    message(FATAL_ERROR "EXECUTORCH_BUILD_DIR must be set (via preset).")
endif()

# ExecuTorch headers come from the source tree, not the build tree.
# Include path is the PARENT of the executorch/ directory so that
# #include "executorch/runtime/..." resolves correctly.
set(EXECUTORCH_SRC_DIR "$ENV{HOME}/executorch")

message(STATUS "TARGET_PLATFORM:      ${TARGET_PLATFORM}")
message(STATUS "EXECUTORCH_BUILD_DIR: ${EXECUTORCH_BUILD_DIR}")
message(STATUS "EXECUTORCH_SRC_DIR:   ${EXECUTORCH_SRC_DIR}")

set(ET_LIB_DIR ${EXECUTORCH_BUILD_DIR})

# ------------------------------------------------------------------ #
#  Imported ExecuTorch libraries                                       #
#                                                                      #
#  These are pre-built in EXECUTORCH_BUILD_DIR.                       #
#  We use STATIC IMPORTED rather than find_package because            #
#  ExecuTorch 1.3.0 does not install a CMake config file.            #
# ------------------------------------------------------------------ #

add_library(executorch_core STATIC IMPORTED)
set_target_properties(executorch_core PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch_core.a)

add_library(executorch STATIC IMPORTED)
set_target_properties(executorch PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch.a)

add_library(executorch_selected_kernels STATIC IMPORTED)
set_target_properties(executorch_selected_kernels PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch_selected_kernels.a)

add_library(portable_kernels STATIC IMPORTED)
set_target_properties(portable_kernels PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/kernels/portable/libportable_kernels.a)

add_library(extension_data_loader STATIC IMPORTED)
set_target_properties(extension_data_loader PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/extension/data_loader/libextension_data_loader.a)

add_library(flatccrt STATIC IMPORTED)
set_target_properties(flatccrt PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/third-party/flatcc_ep/lib/libflatccrt.a)

# Common include directories for both targets
set(ET_INCLUDES
    ${EXECUTORCH_SRC_DIR}
    ${EXECUTORCH_SRC_DIR}/executorch/third-party/flatbuffers/include
    ${ET_LIB_DIR}/third-party/flatcc_ep/include
    # torch/include needed for c10/util/irange.h referenced by exec_aten headers
    $ENV{HOME}/executorch/et-env/lib/python3.12/site-packages/torch/include
)

# ------------------------------------------------------------------ #
#  Host (Linux) build                                                  #
# ------------------------------------------------------------------ #
if(TARGET_PLATFORM STREQUAL "host")

    add_executable(inference_host
        src/main_host.cpp
    )

    # ExecuTorch headers require C++17 (std::variant, is_same_v, constexpr if).
    # Linux GCC defaults to gnu++17, but Apple Clang from the macOS Command
    # Line Tools still defaults to an older standard, so set it explicitly.
    target_compile_options(inference_host PRIVATE
        -std=c++17 -Wall -Wextra -g -Og
    )

    target_include_directories(inference_host PRIVATE
        ${ET_INCLUDES}
    )

    # The whole selected-kernels archive must be pulled in because op
    # registration happens via static initializers (global constructors).
    # Otherwise the linker discards object files with no directly referenced
    # symbols, the registration code is never linked, and inference fails
    # with OperatorMissing at runtime.
    #
    # GNU ld spells this --whole-archive/--no-whole-archive around the lib;
    # Apple's ld64 (macOS) has no such flag and instead takes -force_load
    # with the archive path, which we get from the imported target.
    if(APPLE)
        target_link_libraries(inference_host PRIVATE
            -Wl,-force_load,$<TARGET_FILE:executorch_selected_kernels>
        )
    else()
        target_link_libraries(inference_host PRIVATE
            -Wl,--whole-archive
            executorch_selected_kernels
            -Wl,--no-whole-archive
        )
    endif()

    target_link_libraries(inference_host PRIVATE
        portable_kernels
        executorch
        executorch_core
        extension_data_loader
        flatccrt
    )

# ------------------------------------------------------------------ #
#  STM32F407 (bare-metal Cortex-M4) build                             #
# ------------------------------------------------------------------ #
elseif(TARGET_PLATFORM STREQUAL "stm32")

    set(CPU_FLAGS
        "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard"
    )

    set(CMAKE_C_FLAGS
        "${CPU_FLAGS} -Os -ffunction-sections -fdata-sections -mno-unaligned-access -DET_LOG_ENABLED=0"
        CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS
        "${CPU_FLAGS} -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -std=c++17 -mno-unaligned-access -DET_LOG_ENABLED=0"
        CACHE STRING "" FORCE)

    #
    # Convert .pte file to a C array header at configure time.
    #
    # IMPORTANT: We chdir into models/ before running xxd so that the
    # generated variable name is "tiny_model_pte" (based on filename only)
    # rather than the full mangled path. xxd -i uses the input filename
    # as the C variable name, replacing non-alphanumeric chars with '_'.
    #
    # We also post-process the output to add __attribute__((aligned(16)))
    # to the array declaration. ExecuTorch's FlatBuffer parser checks
    # that the program data pointer is aligned to alignof(std::max_align_t)
    # which is typically 8 or 16 bytes. The default char[] has no alignment
    # guarantee and fails the IsAligned() check in Program::load(),
    # returning Error::InvalidArgument (0x12).
    #
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/models
            xxd -i tiny_model.pte
        OUTPUT_VARIABLE XXD_OUTPUT
        RESULT_VARIABLE XXD_RESULT
    )
    if(NOT XXD_RESULT EQUAL 0)
        message(FATAL_ERROR
            "xxd failed. Is models/tiny_model.pte present?\n"
            "Run: python3 export_model.py")
    endif()

    # Add alignment attribute and const qualifier
    string(REPLACE
        "unsigned char tiny_model_pte[]"
        "__attribute__((aligned(16))) const unsigned char tiny_model_pte[]"
        XXD_OUTPUT "${XXD_OUTPUT}")
    string(REPLACE
        "unsigned int tiny_model_pte_len"
        "const unsigned int tiny_model_pte_len"
        XXD_OUTPUT "${XXD_OUTPUT}")

    file(WRITE ${CMAKE_BINARY_DIR}/tiny_model_pte.h "${XXD_OUTPUT}")
    message(STATUS "Generated tiny_model_pte.h in ${CMAKE_BINARY_DIR}")

    add_executable(firmware.elf
        startup_stm32f407.s
        src/syscalls.c
        src/main.cpp
    )

    target_compile_options(firmware.elf PRIVATE
        -g -Og    # debug info + optimize for debugging
   )


    target_include_directories(firmware.elf PRIVATE
        ${ET_INCLUDES}
        ${CMAKE_BINARY_DIR}      # for generated tiny_model_pte.h
    )

    # See host target comments for --whole-archive rationale.
    # Also note: portable_kernels is listed outside --whole-archive
    # as the actual kernel implementations are referenced by symbol
    # from executorch_selected_kernels and linked normally.
    target_link_libraries(firmware.elf PRIVATE
        -Wl,--whole-archive
        executorch_selected_kernels
        -Wl,--no-whole-archive
        portable_kernels
        executorch
        executorch_core
        extension_data_loader
        flatccrt
    )

    # Use -nostartfiles to exclude crti.o/crtn.o from the link.
    # crti.o provides _init() as an open-ended section that requires
    # crtn.o to close it with bx lr. Without crtn.o, _init() falls
    # through into _fini() and then into unmapped memory, causing a
    # fault during __libc_init_array. We provide our own _init/_fini
    # stubs in startup_stm32f407.s instead.
    target_link_options(firmware.elf PRIVATE
        -nostartfiles
        -specs=nano.specs
        -T${CMAKE_SOURCE_DIR}/stm32f407.ld
        -Wl,--gc-sections
        -lc -lm -lgcc
    )

    # Generate .bin for flashing with OpenOCD
    add_custom_command(TARGET firmware.elf POST_BUILD
        COMMAND arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
        COMMENT "Generating firmware.bin"
    )

    # Print memory usage after build
    add_custom_command(TARGET firmware.elf POST_BUILD
        COMMAND arm-none-eabi-size firmware.elf
        COMMENT "Firmware size:"
    )

else()
    message(FATAL_ERROR
        "Unknown TARGET_PLATFORM: '${TARGET_PLATFORM}'. "
        "Must be 'host' or 'stm32'.")
endif()
