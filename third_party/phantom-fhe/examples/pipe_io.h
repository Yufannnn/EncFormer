#pragma once

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace pipe_io {

inline std::string get_pipe_dir() {
    const char *d = std::getenv("PIPE_DIR");
    if (!d || d[0] == '\0')
        throw std::runtime_error("PIPE_DIR environment variable not set");
    return std::string(d);
}

inline std::string get_pipe_dir_or(const char *env_name) {
    const char *d = std::getenv(env_name);
    if (!d || d[0] == '\0')
        throw std::runtime_error(std::string(env_name) + " environment variable not set");
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

inline const pid_t parent_at_start = getppid();

inline bool file_present(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

[[noreturn]] inline void abort_wait(const std::string &why, const std::string &path) {
    std::cerr << "[pipe_io] giving up waiting for " << path << ": " << why << "\n";
    std::exit(3);
}

inline void exit_if_orphaned(const std::string &path) {
    if (getppid() != parent_at_start) abort_wait("parent process exited", path);
}

inline double pipe_timeout_s() {
    const char *e = std::getenv("ENCFORMER_PIPE_TIMEOUT_S");
    double v = (e && *e) ? std::atof(e) : 600.0;
    return v > 0 ? v : 600.0;
}

inline void wait_for_file(const std::string &path, useconds_t poll_us = 10000) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(pipe_timeout_s());
    while (!file_present(path)) {
        exit_if_orphaned(path);
        if (std::chrono::steady_clock::now() > deadline) abort_wait("timeout", path);
        usleep(poll_us);
    }
}

}
