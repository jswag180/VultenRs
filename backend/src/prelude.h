#define FLOAT_NUM 0
#define INT_NUM 1
#define UINT_NUM 2
#define INT64_NUM 3
#define UINT64_NUM 4

#ifdef USE_INT64
    #extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
    #extension GL_EXT_shader_subgroup_extended_types_int64 : enable
#endif