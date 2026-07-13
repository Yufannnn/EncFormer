// binding.cpp — pybind11 module exposing EzPC/SCI 2PC protocols to Python.
//
// This produces the `ezpc_sci` module that mpc_engine_ezpc.py imports.
// Interface:
//   ctx = ezpc_sci.SCIContext(role=0)
//   out = ezpc_sci.softmax_2pc(ctx, x, scale_bits=13, ring_bits=43)
//   out = ezpc_sci.layer_norm_2pc(ctx, x, eps=1e-5, scale_bits=13, ring_bits=43)
//   out = ezpc_sci.gelu_2pc(ctx, x, scale_bits=13, ring_bits=43)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ezpc_sci/config.h"
#include "ezpc_sci/context.h"
#include "ezpc_sci/fixedpoint.h"
#include "ezpc_sci/bpmax.h"
#include "ezpc_sci/gelu.h"
#include "ezpc_sci/layernorm.h"
#include "ezpc_sci/mbnorm.h"
#include "ezpc_sci/probe.h"
#include "ezpc_sci/softmax.h"

namespace py = pybind11;

using ezpc_sci::ProtocolConfig;
using ezpc_sci::SCIContext;

// ---------------------------------------------------------------------------
// Helper: build config from kwargs
// ---------------------------------------------------------------------------

static ProtocolConfig make_config(int scale_bits, int ring_bits, double threshold = 2.7) {
    return {ring_bits, scale_bits, threshold};
}

// ---------------------------------------------------------------------------
// Softmax 2PC
// ---------------------------------------------------------------------------

static py::array_t<double> softmax_2pc(
    SCIContext &ctx,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    int scale_bits, int ring_bits)
{
    auto buf = x.request();
    if (buf.ndim < 1 || buf.ndim > 2)
        throw std::invalid_argument("softmax_2pc: input must be 1D or 2D");

    size_t rows, cols;
    if (buf.ndim == 1) {
        rows = 1;
        cols = static_cast<size_t>(buf.shape[0]);
    } else {
        rows = static_cast<size_t>(buf.shape[0]);
        cols = static_cast<size_t>(buf.shape[1]);
    }

    auto cfg = make_config(scale_bits, ring_bits);
    auto result = py::array_t<double>(buf.shape, buf.strides);
    auto res_buf = result.request();

#ifdef EZPC_HAS_SCI
    ezpc_sci::softmax_rows_sci(
        ctx,
        static_cast<const double *>(buf.ptr),
        static_cast<double *>(res_buf.ptr),
        rows, cols, cfg);
#else
    ezpc_sci::softmax_rows(
        static_cast<const double *>(buf.ptr),
        static_cast<double *>(res_buf.ptr),
        rows, cols, cfg);
#endif

    return result;
}

// ---------------------------------------------------------------------------
// Layer norm 2PC
// ---------------------------------------------------------------------------

static py::array_t<double> layer_norm_2pc(
    SCIContext &ctx,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    double eps, int scale_bits, int ring_bits)
{
    auto buf = x.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("layer_norm_2pc: input must be 2D");

    size_t rows = static_cast<size_t>(buf.shape[0]);
    size_t cols = static_cast<size_t>(buf.shape[1]);

    auto cfg = make_config(scale_bits, ring_bits);
    auto result = py::array_t<double>(buf.shape, buf.strides);
    auto res_buf = result.request();

#ifdef EZPC_HAS_SCI
    ezpc_sci::layer_norm_sci(
        ctx,
        static_cast<const double *>(buf.ptr),
        static_cast<double *>(res_buf.ptr),
        rows, cols, eps, cfg);
#else
    ezpc_sci::layer_norm(
        static_cast<const double *>(buf.ptr),
        static_cast<double *>(res_buf.ptr),
        rows, cols, eps, cfg);
#endif

    return result;
}

// ---------------------------------------------------------------------------
// GELU 2PC
// ---------------------------------------------------------------------------

static py::array_t<double> gelu_2pc(
    SCIContext &ctx,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    int scale_bits, int ring_bits)
{
    auto buf = x.request();
    size_t n = static_cast<size_t>(buf.size);

    auto cfg = make_config(scale_bits, ring_bits);
    auto result = py::array_t<double>(buf.shape, buf.strides);
    auto res_buf = result.request();

    // GELU uses our Algorithm 4 implementation in both modes.
    // With SCI, the comparison/mux ops inside gelu.h would use OT protocols.
    // Currently gelu.h implements the full fixed-point algorithm in C++.
    ezpc_sci::gelu_vec(
        static_cast<const double *>(buf.ptr),
        static_cast<double *>(res_buf.ptr),
        n, cfg);

    return result;
}

// ---------------------------------------------------------------------------
// Probe: 2PC round-trip (input SERVER -> output PUBLIC). Diagnostic.
// ---------------------------------------------------------------------------

static py::array_t<double> probe_2pc(
    SCIContext &ctx,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    int scale_bits, int ring_bits)
{
    auto buf = x.request();
    size_t n = static_cast<size_t>(buf.size);
    auto cfg = make_config(scale_bits, ring_bits);
    auto result = py::array_t<double>(buf.shape, buf.strides);
    auto res_buf = result.request();
#ifdef EZPC_HAS_SCI
    ezpc_sci::probe_roundtrip_sci(
        ctx, static_cast<const double *>(buf.ptr),
        static_cast<double *>(res_buf.ptr), n, cfg);
#else
    ezpc_sci::probe_roundtrip(
        static_cast<const double *>(buf.ptr),
        static_cast<double *>(res_buf.ptr), n, cfg);
#endif
    return result;
}

// ---------------------------------------------------------------------------
// BPMax 2PC — paper softmax surrogate: clamp(x+c,0)^p * inv_rd (public per row)
// ---------------------------------------------------------------------------

static py::array_t<double> bpmax_2pc(
    SCIContext &ctx,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> inv_rd,
    double c, int p, int scale_bits, int ring_bits)
{
    auto buf = x.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("bpmax_2pc: x must be 2D (rows x cols)");
    size_t rows = static_cast<size_t>(buf.shape[0]);
    size_t cols = static_cast<size_t>(buf.shape[1]);
    auto rbuf = inv_rd.request();
    if (static_cast<size_t>(rbuf.size) != rows)
        throw std::invalid_argument("bpmax_2pc: inv_rd length must equal rows");

    auto cfg = make_config(scale_bits, ring_bits);
    auto result = py::array_t<double>(buf.shape, buf.strides);
    auto res_buf = result.request();
#ifdef EZPC_HAS_SCI
    ezpc_sci::bpmax_rows_sci(
        ctx, static_cast<const double *>(buf.ptr),
        static_cast<const double *>(rbuf.ptr),
        static_cast<double *>(res_buf.ptr), rows, cols, c, p, cfg);
#else
    ezpc_sci::bpmax_rows(
        static_cast<const double *>(buf.ptr),
        static_cast<const double *>(rbuf.ptr),
        static_cast<double *>(res_buf.ptr), rows, cols, c, p, cfg);
#endif
    return result;
}

// ---------------------------------------------------------------------------
// MBNorm 2PC — paper BatchLayerNorm: ((x-mean)/R_d)*gamma+beta (R_d,g,b public)
// ---------------------------------------------------------------------------

static py::array_t<double> mbnorm_2pc(
    SCIContext &ctx,
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> inv_rd,
    py::array_t<double, py::array::c_style | py::array::forcecast> gamma,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta,
    int scale_bits, int ring_bits)
{
    auto buf = x.request();
    if (buf.ndim != 2)
        throw std::invalid_argument("mbnorm_2pc: x must be 2D (rows x cols)");
    size_t rows = static_cast<size_t>(buf.shape[0]);
    size_t cols = static_cast<size_t>(buf.shape[1]);
    auto rbuf = inv_rd.request();
    auto gbuf = gamma.request();
    auto bbuf = beta.request();
    if (static_cast<size_t>(rbuf.size) != rows)
        throw std::invalid_argument("mbnorm_2pc: inv_rd length must equal rows");
    if (static_cast<size_t>(gbuf.size) != cols || static_cast<size_t>(bbuf.size) != cols)
        throw std::invalid_argument("mbnorm_2pc: gamma/beta length must equal cols");

    auto cfg = make_config(scale_bits, ring_bits);
    auto result = py::array_t<double>(buf.shape, buf.strides);
    auto res_buf = result.request();
#ifdef EZPC_HAS_SCI
    ezpc_sci::mbnorm_rows_sci(
        ctx, static_cast<const double *>(buf.ptr),
        static_cast<const double *>(rbuf.ptr),
        static_cast<const double *>(gbuf.ptr),
        static_cast<const double *>(bbuf.ptr),
        static_cast<double *>(res_buf.ptr), rows, cols, cfg);
#else
    ezpc_sci::mbnorm_rows(
        static_cast<const double *>(buf.ptr),
        static_cast<const double *>(rbuf.ptr),
        static_cast<const double *>(gbuf.ptr),
        static_cast<const double *>(bbuf.ptr),
        static_cast<double *>(res_buf.ptr), rows, cols, cfg);
#endif
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(ezpc_sci, m) {
    m.doc() = "EzPC/SCI 2PC protocols for EncFormer nonlinear layers";

    py::class_<SCIContext>(m, "SCIContext")
        .def(py::init<int, std::string, int>(),
             py::arg("role") = 0,
             py::arg("address") = "127.0.0.1",
             py::arg("port") = 32000)
        .def_property_readonly("role", &SCIContext::role)
#ifdef EZPC_HAS_SCI
        .def("get_comm", &SCIContext::get_comm,
             "Total bytes sent by this party on the SCI channels so far")
        .def("get_rounds", &SCIContext::get_rounds,
             "Total communication rounds on the SCI channels so far")
#endif
        ;

    m.def("softmax_2pc", &softmax_2pc,
          "Secure 2PC softmax (row-wise)",
          py::arg("ctx"),
          py::arg("x"),
          py::arg("scale_bits") = 13,
          py::arg("ring_bits") = 43);

    m.def("layer_norm_2pc", &layer_norm_2pc,
          "Secure 2PC layer normalization (row-wise)",
          py::arg("ctx"),
          py::arg("x"),
          py::arg("eps") = 1e-5,
          py::arg("scale_bits") = 13,
          py::arg("ring_bits") = 43);

    m.def("gelu_2pc", &gelu_2pc,
          "Secure 2PC GELU activation (Algorithm 4)",
          py::arg("ctx"),
          py::arg("x"),
          py::arg("scale_bits") = 13,
          py::arg("ring_bits") = 43);

    m.def("probe_2pc", &probe_2pc,
          "2PC round-trip probe (input SERVER -> output PUBLIC)",
          py::arg("ctx"), py::arg("x"),
          py::arg("scale_bits") = 13, py::arg("ring_bits") = 43);

    m.def("bpmax_2pc", &bpmax_2pc,
          "Secure 2PC BPMax attention weights: clamp(x+c,0)^p * inv_rd (public per-row)",
          py::arg("ctx"), py::arg("x"), py::arg("inv_rd"),
          py::arg("c") = 5.0, py::arg("p") = 5,
          py::arg("scale_bits") = 13, py::arg("ring_bits") = 43);

    m.def("mbnorm_2pc", &mbnorm_2pc,
          "Secure 2PC MBNorm/BatchLayerNorm: ((x-mean)/R_d)*gamma+beta (R_d,gamma,beta public)",
          py::arg("ctx"), py::arg("x"), py::arg("inv_rd"),
          py::arg("gamma"), py::arg("beta"),
          py::arg("scale_bits") = 13, py::arg("ring_bits") = 43);

    // Version/build info
#ifdef EZPC_HAS_SCI
    m.attr("HAS_NATIVE_SCI") = true;
#else
    m.attr("HAS_NATIVE_SCI") = false;
#endif
}
