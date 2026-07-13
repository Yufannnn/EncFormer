add_executable(example_context
        1_bfv.cu
        2_bgv.cu
        3_ckks.cu
        4_kernel_fusing.cu
        example.cu
)
target_link_libraries(example_context PRIVATE Phantom)
target_include_directories(example_context PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(example_serialization
        serialization.cu
)
target_link_libraries(example_serialization PRIVATE Phantom)
target_include_directories(example_serialization PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(bench_ckks_micro
        bench_ckks_micro.cu
)
target_link_libraries(bench_ckks_micro PRIVATE Phantom)
target_include_directories(bench_ckks_micro PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(bench_ckks_qkv_small
        bench_ckks_qkv_small.cu
)
target_link_libraries(bench_ckks_qkv_small PRIVATE Phantom)
target_include_directories(bench_ckks_qkv_small PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(bench_ckks_qkv_small PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_s_small
        bench_ckks_s_small.cu
)
target_link_libraries(bench_ckks_s_small PRIVATE Phantom)
target_include_directories(bench_ckks_s_small PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(bench_ckks_s_small PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_score_full
        bench_ckks_score_full.cu
)
target_link_libraries(bench_ckks_score_full PRIVATE Phantom)
target_include_directories(bench_ckks_score_full PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(bench_ckks_score_full PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_value_full
        bench_ckks_value_full.cu
)
target_link_libraries(bench_ckks_value_full PRIVATE Phantom)
target_include_directories(bench_ckks_value_full PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(bench_ckks_value_full PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_ffn_full
        bench_ckks_ffn_full.cu
)
target_link_libraries(bench_ckks_ffn_full PRIVATE Phantom)
target_include_directories(bench_ckks_ffn_full PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(bench_ckks_ffn_full PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_qkv_full
        bench_ckks_qkv_full.cu
)
target_link_libraries(bench_ckks_qkv_full PRIVATE Phantom)
target_include_directories(bench_ckks_qkv_full PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(bench_ckks_qkv_full PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_ff1
        pipe_ckks_ff1.cu
)
target_link_libraries(pipe_ckks_ff1 PRIVATE Phantom)
target_include_directories(pipe_ckks_ff1 PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(pipe_ckks_ff1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_ff2
        pipe_ckks_ff2.cu
)
target_link_libraries(pipe_ckks_ff2 PRIVATE Phantom)
target_include_directories(pipe_ckks_ff2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(pipe_ckks_ff2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_qkv
        pipe_ckks_qkv.cu
)
target_link_libraries(pipe_ckks_qkv PRIVATE Phantom)
target_include_directories(pipe_ckks_qkv PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(pipe_ckks_qkv PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_value_out
        pipe_ckks_value_out.cu
)
target_link_libraries(pipe_ckks_value_out PRIVATE Phantom)
target_include_directories(pipe_ckks_value_out PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(pipe_ckks_value_out PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_qkv_score
        pipe_ckks_qkv_score.cu
)
target_link_libraries(pipe_ckks_qkv_score PRIVATE Phantom)
target_include_directories(pipe_ckks_qkv_score PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(pipe_ckks_qkv_score PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_attn
        pipe_ckks_attn.cu
)
target_link_libraries(pipe_ckks_attn PRIVATE Phantom)
target_include_directories(pipe_ckks_attn PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(pipe_ckks_attn PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_qkv_full_bert_base
        bench_ckks_qkv_full.cu
)
target_link_libraries(bench_ckks_qkv_full_bert_base PRIVATE Phantom)
target_include_directories(bench_ckks_qkv_full_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_qkv_full_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_ckks_qkv_full_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_score_full_bert_base
        bench_ckks_score_full.cu
)
target_link_libraries(bench_ckks_score_full_bert_base PRIVATE Phantom)
target_include_directories(bench_ckks_score_full_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_score_full_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_ckks_score_full_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_value_full_bert_base
        bench_ckks_value_full.cu
)
target_link_libraries(bench_ckks_value_full_bert_base PRIVATE Phantom)
target_include_directories(bench_ckks_value_full_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_value_full_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_ckks_value_full_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_ffn_full_bert_base
        bench_ckks_ffn_full.cu
)
target_link_libraries(bench_ckks_ffn_full_bert_base PRIVATE Phantom)
target_include_directories(bench_ckks_ffn_full_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_ffn_full_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_ckks_ffn_full_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_ff1_bert_base
        pipe_ckks_ff1.cu
)
target_link_libraries(pipe_ckks_ff1_bert_base PRIVATE Phantom)
target_include_directories(pipe_ckks_ff1_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_ff1_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_ff1_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_ff2_bert_base
        pipe_ckks_ff2.cu
)
target_link_libraries(pipe_ckks_ff2_bert_base PRIVATE Phantom)
target_include_directories(pipe_ckks_ff2_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_ff2_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_ff2_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_qkv_bert_base
        pipe_ckks_qkv.cu
)
target_link_libraries(pipe_ckks_qkv_bert_base PRIVATE Phantom)
target_include_directories(pipe_ckks_qkv_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_qkv_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_qkv_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_value_out_bert_base
        pipe_ckks_value_out.cu
)
target_link_libraries(pipe_ckks_value_out_bert_base PRIVATE Phantom)
target_include_directories(pipe_ckks_value_out_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_value_out_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_value_out_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_qkv_score_bert_base
        pipe_ckks_qkv_score.cu
)
target_link_libraries(pipe_ckks_qkv_score_bert_base PRIVATE Phantom)
target_include_directories(pipe_ckks_qkv_score_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_qkv_score_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_qkv_score_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_attn_bert_base
        pipe_ckks_attn.cu
)
target_link_libraries(pipe_ckks_attn_bert_base PRIVATE Phantom)
target_include_directories(pipe_ckks_attn_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_attn_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_attn_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_qkv_full_bert_large
        bench_ckks_qkv_full.cu
)
target_link_libraries(bench_ckks_qkv_full_bert_large PRIVATE Phantom)
target_include_directories(bench_ckks_qkv_full_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_qkv_full_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(bench_ckks_qkv_full_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_score_full_bert_large
        bench_ckks_score_full.cu
)
target_link_libraries(bench_ckks_score_full_bert_large PRIVATE Phantom)
target_include_directories(bench_ckks_score_full_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_score_full_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(bench_ckks_score_full_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_value_full_bert_large
        bench_ckks_value_full.cu
)
target_link_libraries(bench_ckks_value_full_bert_large PRIVATE Phantom)
target_include_directories(bench_ckks_value_full_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_value_full_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(bench_ckks_value_full_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_ffn_full_bert_large
        bench_ckks_ffn_full.cu
)
target_link_libraries(bench_ckks_ffn_full_bert_large PRIVATE Phantom)
target_include_directories(bench_ckks_ffn_full_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_ffn_full_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(bench_ckks_ffn_full_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_ff1_bert_large
        pipe_ckks_ff1.cu
)
target_link_libraries(pipe_ckks_ff1_bert_large PRIVATE Phantom)
target_include_directories(pipe_ckks_ff1_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_ff1_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(pipe_ckks_ff1_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_ff2_bert_large
        pipe_ckks_ff2.cu
)
target_link_libraries(pipe_ckks_ff2_bert_large PRIVATE Phantom)
target_include_directories(pipe_ckks_ff2_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_ff2_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(pipe_ckks_ff2_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_qkv_bert_large
        pipe_ckks_qkv.cu
)
target_link_libraries(pipe_ckks_qkv_bert_large PRIVATE Phantom)
target_include_directories(pipe_ckks_qkv_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_qkv_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(pipe_ckks_qkv_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_value_out_bert_large
        pipe_ckks_value_out.cu
)
target_link_libraries(pipe_ckks_value_out_bert_large PRIVATE Phantom)
target_include_directories(pipe_ckks_value_out_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_value_out_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(pipe_ckks_value_out_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_qkv_score_bert_large
        pipe_ckks_qkv_score.cu
)
target_link_libraries(pipe_ckks_qkv_score_bert_large PRIVATE Phantom)
target_include_directories(pipe_ckks_qkv_score_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_qkv_score_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(pipe_ckks_qkv_score_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_attn_bert_large
        pipe_ckks_attn.cu
)
target_link_libraries(pipe_ckks_attn_bert_large PRIVATE Phantom)
target_include_directories(pipe_ckks_attn_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_attn_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(pipe_ckks_attn_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_qkv_full_gpt2_base
        bench_ckks_qkv_full.cu
)
target_link_libraries(bench_ckks_qkv_full_gpt2_base PRIVATE Phantom)
target_include_directories(bench_ckks_qkv_full_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_qkv_full_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_ckks_qkv_full_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_score_full_gpt2_base
        bench_ckks_score_full.cu
)
target_link_libraries(bench_ckks_score_full_gpt2_base PRIVATE Phantom)
target_include_directories(bench_ckks_score_full_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_score_full_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_ckks_score_full_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_value_full_gpt2_base
        bench_ckks_value_full.cu
)
target_link_libraries(bench_ckks_value_full_gpt2_base PRIVATE Phantom)
target_include_directories(bench_ckks_value_full_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_value_full_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_ckks_value_full_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_ckks_ffn_full_gpt2_base
        bench_ckks_ffn_full.cu
)
target_link_libraries(bench_ckks_ffn_full_gpt2_base PRIVATE Phantom)
target_include_directories(bench_ckks_ffn_full_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_ckks_ffn_full_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_ckks_ffn_full_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_ff1_gpt2_base
        pipe_ckks_ff1.cu
)
target_link_libraries(pipe_ckks_ff1_gpt2_base PRIVATE Phantom)
target_include_directories(pipe_ckks_ff1_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_ff1_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_ff1_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_ff2_gpt2_base
        pipe_ckks_ff2.cu
)
target_link_libraries(pipe_ckks_ff2_gpt2_base PRIVATE Phantom)
target_include_directories(pipe_ckks_ff2_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_ff2_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_ff2_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_qkv_gpt2_base
        pipe_ckks_qkv.cu
)
target_link_libraries(pipe_ckks_qkv_gpt2_base PRIVATE Phantom)
target_include_directories(pipe_ckks_qkv_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_qkv_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_qkv_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_value_out_gpt2_base
        pipe_ckks_value_out.cu
)
target_link_libraries(pipe_ckks_value_out_gpt2_base PRIVATE Phantom)
target_include_directories(pipe_ckks_value_out_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_value_out_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_value_out_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_qkv_score_gpt2_base
        pipe_ckks_qkv_score.cu
)
target_link_libraries(pipe_ckks_qkv_score_gpt2_base PRIVATE Phantom)
target_include_directories(pipe_ckks_qkv_score_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_qkv_score_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_qkv_score_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_attn_gpt2_base
        pipe_ckks_attn.cu
)
target_link_libraries(pipe_ckks_attn_gpt2_base PRIVATE Phantom)
target_include_directories(pipe_ckks_attn_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(pipe_ckks_attn_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_attn_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- pipe_ckks_layer: Full transformer layer with bridge protocol ---
# Path B target (binary builds; F_0/F_1 polynomial pre-evaluation in
# pipe_ckks_common.h is still placeholder, so the deployed Table VIII / Table XI
# numbers reflect the minimal-GELU code path, not the expanded-GELU Path B
# pre-evaluation).

add_executable(pipe_ckks_layer
        pipe_ckks_layer.cu
)
target_link_libraries(pipe_ckks_layer PRIVATE Phantom)
target_include_directories(pipe_ckks_layer PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(pipe_ckks_layer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- bench_f0f1: microbenchmark for expanded-GELU F_0/F_1 CKKS pre-evaluation ---
# Standalone timing harness for compute_f0_f1_in_ckks (Path B Step B).
add_executable(bench_f0f1
        bench_f0f1.cu
)
target_compile_definitions(bench_f0f1 PRIVATE ENCFORMER_MODEL_BERT_BASE)
target_link_libraries(bench_f0f1 PRIVATE Phantom)
target_include_directories(bench_f0f1 PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(bench_f0f1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_layer_bert_base
        pipe_ckks_layer.cu
)
target_link_libraries(pipe_ckks_layer_bert_base PRIVATE Phantom)
target_include_directories(pipe_ckks_layer_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(pipe_ckks_layer_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_layer_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_layer_bert_large
        pipe_ckks_layer.cu
)
target_link_libraries(pipe_ckks_layer_bert_large PRIVATE Phantom)
target_include_directories(pipe_ckks_layer_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(pipe_ckks_layer_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(pipe_ckks_layer_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pipe_ckks_layer_gpt2_base
        pipe_ckks_layer.cu
)
target_link_libraries(pipe_ckks_layer_gpt2_base PRIVATE Phantom)
target_include_directories(pipe_ckks_layer_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(pipe_ckks_layer_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(pipe_ckks_layer_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_f0f1_bert_base
        bench_f0f1.cu
)
target_link_libraries(bench_f0f1_bert_base PRIVATE Phantom)
target_include_directories(bench_f0f1_bert_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_f0f1_bert_base PRIVATE ENCFORMER_M=128 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_f0f1_bert_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_f0f1_bert_large
        bench_f0f1.cu
)
target_link_libraries(bench_f0f1_bert_large PRIVATE Phantom)
target_include_directories(bench_f0f1_bert_large PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_f0f1_bert_large PRIVATE ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16 ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)
set_target_properties(bench_f0f1_bert_large PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_f0f1_gpt2_base
        bench_f0f1.cu
)
target_link_libraries(bench_f0f1_gpt2_base PRIVATE Phantom)
target_include_directories(bench_f0f1_gpt2_base PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_f0f1_gpt2_base PRIVATE ENCFORMER_M=64 ENCFORMER_D=768 ENCFORMER_H=12 ENCFORMER_D_FF=3072 ENCFORMER_NSLOTS=16384)
set_target_properties(bench_f0f1_gpt2_base PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
