#pragma once
// pipe_io.h — File I/O for the native pipe CKKS pipeline.
// Data is raw float64 arrays (no headers). Dimensions are compile-time constants.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace pipe_io {

inline std::string get_pipe_dir() {
    const char *d = std::getenv("PIPE_DIR");
    if (!d || d[0] == '\0')
        throw std::runtime_error("PIPE_DIR environment variable not set");
    return std::string(d);
}

inline std::string pipe_path(const std::string &name) {
    return get_pipe_dir() + "/" + name;
}

inline void read_f64(const char *path, double *buf, size_t n) {
    FILE *f = std::fopen(path, "rb");
    if (!f)
        throw std::runtime_error(std::string("pipe_io::read_f64: cannot open ") + path);
    size_t got = std::fread(buf, sizeof(double), n, f);
    std::fclose(f);
    if (got != n)
        throw std::runtime_error(
            std::string("pipe_io::read_f64: expected ") + std::to_string(n) +
            " doubles, got " + std::to_string(got) + " from " + path);
}

inline void write_f64(const char *path, const double *buf, size_t n) {
    FILE *f = std::fopen(path, "wb");
    if (!f)
        throw std::runtime_error(std::string("pipe_io::write_f64: cannot open ") + path);
    size_t wrote = std::fwrite(buf, sizeof(double), n, f);
    std::fclose(f);
    if (wrote != n)
        throw std::runtime_error(
            std::string("pipe_io::write_f64: expected to write ") + std::to_string(n) +
            " doubles, wrote " + std::to_string(wrote) + " to " + path);
}

inline std::vector<double> read_f64_vec(const char *path, size_t n) {
    std::vector<double> v(n);
    read_f64(path, v.data(), n);
    return v;
}

inline void write_f64_vec(const char *path, const std::vector<double> &v) {
    write_f64(path, v.data(), v.size());
}

} // namespace pipe_io
