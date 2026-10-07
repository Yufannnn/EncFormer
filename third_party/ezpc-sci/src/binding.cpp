#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ezpc_sci/config.h"
#include "ezpc_sci/context.h"
#include "ezpc_sci/shares.h"

namespace py = pybind11;

using ezpc_sci::SCIContext;

using U64Array = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static py::dict phase_dict(const ezpc_sci::PhaseStats &p) {
    py::dict d;
    d["seconds"] = p.seconds;
    d["bytes_sent"] = p.bytes_sent;
    d["bytes_recv"] = p.bytes_recv;
    d["rounds"] = p.rounds;
    d["ot_ext_bytes_sent"] = p.ext_bytes_sent;
    d["ot_ext_choice_bits"] = p.ext_choice_bits;
    d["ot_ext_ots"] = p.ext_ots;
    d["ot_ext_seconds"] = p.ext_seconds;
    d["ot_derand_bytes_sent"] = p.derand_bytes;
    d["ot_counts"] = p.ot_counts;
    return d;
}

static py::dict stats_dict(const ezpc_sci::CallStats &st) {
    py::dict d;
    d["offline"] = phase_dict(st.offline);
    d["online"] = phase_dict(st.online);
    d["threads"] = st.threads;
    d["n"] = st.n;
    d["planned"] = st.planned;
    return d;
}

static U64Array new_like(const py::buffer_info &buf) {
    std::vector<py::ssize_t> shape(buf.shape.begin(), buf.shape.end());
    return U64Array(shape);
}

static const double *opt_f64(py::object o, F64Array &hold, size_t &len) {
    if (o.is_none()) { len = 0; return nullptr; }
    hold = o.cast<F64Array>();
    len = static_cast<size_t>(hold.size());
    return hold.data();
}

#ifdef EZPC_HAS_SCI
using ezpc_sci::shares::TruncMode;

static TruncMode parse_trunc(const std::string &t) {
    if (t == "local") return TruncMode::LOCAL;
    if (t == "faithful") return TruncMode::FAITHFUL;
    throw std::invalid_argument("trunc must be 'local' or 'faithful'");
}

static U64Array bpmax_rows_shares(SCIContext &ctx, U64Array x, py::object rd, double c, int p,
                                  bool prescaled, int scale_bits, int a, int cmp_bits,
                                  int scale_frac, int scale_width, std::string scale_trunc) {
    auto buf = x.request();
    if (buf.ndim < 1) throw std::invalid_argument("bpmax_rows_shares: x must be at least 1-D");
    size_t cols = static_cast<size_t>(buf.shape[buf.ndim - 1]);
    size_t n = static_cast<size_t>(buf.size);
    size_t rows = cols ? n / cols : 0;
    F64Array rhold;
    size_t rlen = 0;
    const double *rp = opt_f64(rd, rhold, rlen);
    if (!prescaled && rlen != rows)
        throw std::invalid_argument("bpmax_rows_shares: rd must have one entry per row (x.size / x.shape[-1])");
    U64Array out = new_like(buf);
    auto obuf = out.request();
    ezpc_sci::shares::BPMaxOpts o;
    o.c = c; o.p = p; o.F = scale_bits; o.a = a; o.cmp_bits = cmp_bits;
    o.scale_frac = scale_frac; o.scale_width = scale_width; o.prescaled = prescaled;
    o.scale_trunc = parse_trunc(scale_trunc);
    {
        py::gil_scoped_release nogil;
        ezpc_sci::shares::bpmax_rows_shares(ctx, static_cast<const uint64_t *>(buf.ptr), rp,
                                            static_cast<uint64_t *>(obuf.ptr), rows, cols, o);
    }
    return out;
}

static U64Array layernorm_shares(SCIContext &ctx, U64Array x, py::object rd, py::object gamma, py::object beta,
                                 double eps, double ln_l, int scale_bits, int coef_bits,
                                 int coef_width, std::string trunc) {
    auto buf = x.request();
    if (buf.ndim < 1) throw std::invalid_argument("layernorm_shares: x must be at least 1-D");
    size_t cols = static_cast<size_t>(buf.shape[buf.ndim - 1]);
    size_t n = static_cast<size_t>(buf.size);
    size_t rows = cols ? n / cols : 0;
    F64Array rh, gh, bh;
    size_t rlen = 0, glen = 0, blen = 0;
    const double *rp = opt_f64(rd, rh, rlen);
    const double *gp = opt_f64(gamma, gh, glen);
    const double *bp = opt_f64(beta, bh, blen);
    if (rlen != rows && rlen != 1) throw std::invalid_argument("layernorm_shares: rd must have length rows or 1");
    if (gp && glen != cols) throw std::invalid_argument("layernorm_shares: gamma length != cols");
    if (bp && blen != cols) throw std::invalid_argument("layernorm_shares: beta length != cols");
    U64Array out = new_like(buf);
    auto obuf = out.request();
    ezpc_sci::shares::LNOpts o;
    o.F = scale_bits; o.coef_bits = coef_bits; o.coef_width = coef_width;
    o.trunc = parse_trunc(trunc);
    {
        py::gil_scoped_release nogil;
        ezpc_sci::shares::layernorm_shares(ctx, static_cast<const uint64_t *>(buf.ptr), rp, rlen, gp, bp,
                                           static_cast<uint64_t *>(obuf.ptr), rows, cols, eps, ln_l, o);
    }
    return out;
}

static U64Array gelu_select_shares(SCIContext &ctx, U64Array x, U64Array f0, U64Array f1, int scale_bits,
                                   int cmp_bits, double threshold) {
    auto bx = x.request(), b0 = f0.request(), b1 = f1.request();
    if (b0.size != bx.size || b1.size != bx.size)
        throw std::invalid_argument("gelu_select_shares: x, f0, f1 must have the same size");
    U64Array out = new_like(bx);
    auto obuf = out.request();
    ezpc_sci::shares::GeluSelectOpts o;
    o.F = scale_bits; o.cmp_bits = cmp_bits; o.threshold = threshold;
    {
        py::gil_scoped_release nogil;
        ezpc_sci::shares::gelu_select_shares(ctx, static_cast<const uint64_t *>(bx.ptr),
                                             static_cast<const uint64_t *>(b0.ptr),
                                             static_cast<const uint64_t *>(b1.ptr),
                                             static_cast<uint64_t *>(obuf.ptr), static_cast<size_t>(bx.size), o);
    }
    return out;
}

static void prepare_bpmax(SCIContext &ctx, size_t rows, size_t cols, py::object rd, double c, int p,
                          bool prescaled, int scale_bits, int a, int cmp_bits, int scale_frac,
                          int scale_width, std::string scale_trunc) {
    F64Array rh;
    size_t rlen = 0;
    const double *rp = opt_f64(rd, rh, rlen);
    if (!prescaled && rlen != rows)
        throw std::invalid_argument("prepare_bpmax: rd must have one entry per row");
    ezpc_sci::shares::BPMaxOpts o;
    o.c = c; o.p = p; o.F = scale_bits; o.a = a; o.cmp_bits = cmp_bits; o.scale_frac = scale_frac;
    o.scale_width = scale_width; o.prescaled = prescaled;
    o.scale_trunc = parse_trunc(scale_trunc);
    py::gil_scoped_release nogil;
    ezpc_sci::shares::prepare_bpmax(ctx, rows, cols, rp, o);
}

static void prepare_layernorm(SCIContext &ctx, size_t rows, size_t cols, int scale_bits, int coef_bits,
                              int coef_width, std::string trunc) {
    ezpc_sci::shares::LNOpts o;
    o.F = scale_bits; o.coef_bits = coef_bits; o.coef_width = coef_width;
    o.trunc = parse_trunc(trunc);
    ezpc_sci::shares::prepare_layernorm(ctx, rows, cols, o);
}

static void prepare_gelu_select(SCIContext &ctx, size_t n, int scale_bits, int cmp_bits, double threshold) {
    ezpc_sci::shares::GeluSelectOpts o;
    o.F = scale_bits; o.cmp_bits = cmp_bits; o.threshold = threshold;
    ezpc_sci::shares::prepare_gelu_select(ctx, n, o);
}

#else
[[noreturn]] static void no_sci(const char *fn) {
    throw std::runtime_error(std::string(fn) + ": ezpc_sci was built without SCI (emulated-only); "
                             "the share API needs the native SCI build");
}
static U64Array bpmax_rows_shares(SCIContext &, U64Array, py::object, double, int, bool, int, int, int, int, int, std::string) { no_sci("bpmax_rows_shares"); }
static U64Array layernorm_shares(SCIContext &, U64Array, py::object, py::object, py::object, double, double, int, int, int, std::string) { no_sci("layernorm_shares"); }
static U64Array gelu_select_shares(SCIContext &, U64Array, U64Array, U64Array, int, int, double) { no_sci("gelu_select_shares"); }
#endif

PYBIND11_MODULE(ezpc_sci, m) {
    m.doc() = "EzPC/SCI two-party protocols for EncFormer";

    py::class_<SCIContext>(m, "SCIContext")
        .def(py::init([](int role, std::string address, int port, int threads) {
                 py::gil_scoped_release nogil;
                 return new SCIContext(role, address, port, ezpc_sci::default_config(), threads);
             }),
             py::arg("role") = 0,
             py::arg("address") = "127.0.0.1",
             py::arg("port") = 32000,
             py::arg("threads") = 1)
        .def_property_readonly("role", &SCIContext::role)
        .def_property_readonly("threads", &SCIContext::threads)
        .def_property_readonly("setup_seconds", &SCIContext::setup_seconds)
        .def_property_readonly("last_stats", [](const SCIContext &c) { return stats_dict(c.last_stats); })
#ifdef EZPC_HAS_SCI
        .def("get_comm", &SCIContext::get_comm)
        .def("get_rounds", &SCIContext::get_rounds)
        .def("comm", [](const SCIContext &c) {
                 auto s = c.comm();
                 py::dict d;
                 d["bytes_sent"] = s.bytes_sent; d["bytes_recv"] = s.bytes_recv; d["rounds"] = s.rounds;
                 return d;
             })
#ifdef EZPC_SCI_HAS_OT_POOL
        .def("ot_counts", &SCIContext::ot_counts)
        .def("pool_left", &SCIContext::pool_left)
        .def("pool_fill", [](SCIContext &c, std::vector<uint64_t> counts) {
                 py::gil_scoped_release nogil;
                 c.pool_fill(counts);
             })
        .def("begin_online", [](SCIContext &c) { c.planned = true; c.pool_set_on(true); })
        .def("end_plan", [](SCIContext &c) {
                 py::dict d;
                 d["material_left"] = c.plan_queue.size();
                 d["pool_left"] = c.pool_left();
                 c.planned = false;
                 c.pool_set_on(false);
                 c.pool_clear();
                 c.plan_queue.clear();
                 return d;
             })
        .def_property_readonly("planned", [](const SCIContext &c) { return c.planned; })
        .def_property_readonly("plan_queue_len", [](const SCIContext &c) { return c.plan_queue.size(); })
        .def_property_readonly_static("OT_INSTANCES", [](py::object) { return SCIContext::OT_INSTANCES; })
#endif
#endif
        ;

    m.def("bpmax_rows_shares", &bpmax_rows_shares,
          py::arg("ctx"), py::arg("x"), py::arg("rd") = py::none(), py::arg("c") = 5.0, py::arg("p") = 5,
          py::arg("prescaled") = false, py::arg("scale_bits") = 13,
          py::arg("a") = 3, py::arg("cmp_bits") = 21, py::arg("scale_frac") = 16, py::arg("scale_width") = 21,
          py::arg("scale_trunc") = "faithful");

    m.def("layernorm_shares", &layernorm_shares,
          py::arg("ctx"), py::arg("x"), py::arg("rd") = py::none(), py::arg("gamma") = py::none(),
          py::arg("beta") = py::none(), py::arg("eps") = 1e-12, py::arg("ln_l") = 1.0,
          py::arg("scale_bits") = 13, py::arg("coef_bits") = 16,
          py::arg("coef_width") = 24, py::arg("trunc") = "local");

    m.def("gelu_select_shares", &gelu_select_shares,
          py::arg("ctx"), py::arg("x"), py::arg("f0"), py::arg("f1"), py::arg("scale_bits") = 13,
          py::arg("cmp_bits") = 22, py::arg("threshold") = 2.7);

#ifdef EZPC_HAS_SCI
    m.def("prepare_bpmax", &prepare_bpmax,
          py::arg("ctx"), py::arg("rows"), py::arg("cols"), py::arg("rd") = py::none(), py::arg("c") = 5.0,
          py::arg("p") = 5, py::arg("prescaled") = false, py::arg("scale_bits") = 13,
          py::arg("a") = 3, py::arg("cmp_bits") = 21, py::arg("scale_frac") = 16, py::arg("scale_width") = 21,
          py::arg("scale_trunc") = "faithful");
    m.def("prepare_layernorm", &prepare_layernorm,
          py::arg("ctx"), py::arg("rows"), py::arg("cols"), py::arg("scale_bits") = 13, py::arg("coef_bits") = 16,
          py::arg("coef_width") = 24, py::arg("trunc") = "local");
    m.def("prepare_gelu_select", &prepare_gelu_select,
          py::arg("ctx"), py::arg("n"), py::arg("scale_bits") = 13, py::arg("cmp_bits") = 22, py::arg("threshold") = 2.7);
#endif
#if defined(EZPC_HAS_SCI) && defined(EZPC_SCI_HAS_OT_POOL)
    m.attr("HAS_OT_POOL") = true;
#else
    m.attr("HAS_OT_POOL") = false;
#endif

#ifdef EZPC_HAS_SCI
    m.attr("HAS_NATIVE_SCI") = true;
#else
    m.attr("HAS_NATIVE_SCI") = false;
#endif
#ifdef EZPC_SCI_COMMIT
    m.attr("EZPC_COMMIT") = EZPC_SCI_COMMIT;
#else
    m.attr("EZPC_COMMIT") = "";
#endif
    m.attr("SHARE_RING_BITS") = 64;
}
