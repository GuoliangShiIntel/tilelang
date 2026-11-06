// npu_run -- runs one TileLang-generated Intel NPU kernel on the device with
// parameters read from files, times it and writes the outputs back.
//
// usage: npu_run --kernel NAME --dir DIR --tiles T --shaves S [--repeat N] [--warmup N] [--profile out.json]
// DIR/params.txt: one line per kernel tensor parameter, in kernel order:
//     <mode> <bytes> <file>      mode: in (read DIR/file) | out (write DIR/file) | inout | zero
// The per-SHAVE status area (32 B each, T*S SHAVEs) is appended as the last
// parameter and written to DIR/status.bin. Inputs are re-copied before every
// run. Prints "run <i> ms <t>" per timed run and "median_ms <t>" (host wall
// time around submit+wait; a kernel-only number needs the profiler trace).
// --profile enables the runtime profiler and saves its JSON: a profiled run is
// an observability run, never a timing claim.
#include <npu_runtime.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kStatusBytesPerShave = 32;
constexpr std::uint32_t kStatusOk = 1;

[[noreturn]] void die(const std::string &m) {
  std::cerr << "ERROR: " << m << std::endl;
  std::exit(1);
}

struct Param {
  std::string mode, file;
  std::uint64_t bytes = 0;
  void *dev = nullptr;
  std::vector<char> init;
};

int run(int argc, char **argv) {
  std::string kernel, dir, profile;
  int repeat = 5, warmup = 1, tiles = 0, shaves = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() {
      if (i + 1 >= argc)
        die("missing value for " + a);
      return std::string(argv[++i]);
    };
    if (a == "--kernel")
      kernel = next();
    else if (a == "--dir")
      dir = next();
    else if (a == "--repeat")
      repeat = std::stoi(next());
    else if (a == "--warmup")
      warmup = std::stoi(next());
    else if (a == "--tiles")
      tiles = std::stoi(next());
    else if (a == "--shaves")
      shaves = std::stoi(next());
    else if (a == "--profile")
      profile = next();
    else
      die("unknown argument " + a);
  }
  if (kernel.empty() || dir.empty() || tiles <= 0 || shaves <= 0)
    die("usage: npu_run --kernel NAME --dir DIR --tiles T --shaves S [--repeat N] [--warmup N] [--profile out.json]");

  auto device = npu::Device();
  if ((std::size_t)tiles > device.getTileCount())
    die("kernel wants " + std::to_string(tiles) + " tiles, device has " + std::to_string(device.getTileCount()));
  std::vector<Param> ps;
  std::ifstream spec(dir + "/params.txt");
  if (!spec)
    die("no " + dir + "/params.txt");
  std::string line;
  while (std::getline(spec, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;
    std::istringstream ls(line);
    Param p;
    ls >> p.mode >> p.bytes >> p.file;
    p.dev = device.mallocShared(std::max<std::uint64_t>(p.bytes, 64));
    if (!p.dev)
      die("mallocShared " + std::to_string(p.bytes));
    if (p.mode == "in" || p.mode == "inout") {
      std::ifstream f(dir + "/" + p.file, std::ios::binary);
      if (!f)
        die("cannot read " + p.file);
      p.init.resize(p.bytes);
      f.read(p.init.data(), (std::streamsize)p.bytes);
      if ((std::uint64_t)f.gcount() != p.bytes)
        die("short file " + p.file);
    }
    ps.push_back(std::move(p));
  }
  const std::uint32_t workers = (std::uint32_t)(tiles * shaves);
  // Sized for every tile the device has, so a status written under an unexpected tile id is still visible.
  const std::uint32_t statusSlots = (std::uint32_t)device.getTileCount() * (std::uint32_t)shaves;
  const std::uint64_t statusBytes = (std::uint64_t)statusSlots * kStatusBytesPerShave;
  auto *status = static_cast<std::uint32_t *>(device.mallocShared(statusBytes));
  auto dumpStatus = [&]() {
    std::ofstream(dir + "/status.bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(status), (std::streamsize)statusBytes);
  };
  auto dumpOutputs = [&]() {
    for (auto &p : ps)
      if (p.mode == "out" || p.mode == "inout") {
        std::ofstream f(dir + "/" + p.file, std::ios::binary);
        f.write(static_cast<const char *>(p.dev), (std::streamsize)p.bytes);
      }
  };
  auto &k = device.createKernel(kernel, npu::Device::Tiles(tiles), npu::Device::ShavesPerTile(shaves));
  if (!profile.empty())
    k.setProfilingEnabled(true);
  auto &stream = device.createStream();
  std::vector<void *> params;
  for (auto &p : ps)
    params.push_back(p.dev);
  params.push_back(status);

  std::vector<double> times;
  for (int r = 0; r < warmup + repeat; ++r) {
    for (auto &p : ps) {
      if (!p.init.empty())
        std::memcpy(p.dev, p.init.data(), p.bytes);
      else if (p.mode == "zero" || p.mode == "out")
        std::memset(p.dev, 0, p.bytes);
    }
    std::memset(status, 0xFF, statusBytes);
    auto t0 = std::chrono::steady_clock::now();
    k.submit(params, stream);
    stream.wait();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::string health;
    if (!device.isHealthy(&health)) {
      dumpStatus();
      dumpOutputs();
      die("device unhealthy after run " + std::to_string(r) + ": " + health);
    }
    for (std::uint32_t w = 0; w < workers; ++w) {
      volatile std::uint32_t *word = status + w * 8;
      if (*word == 0xFFFFFFFFu) {
        // Diagnostic for an intermittent unwritten record: is it late, or never written?
        int waited_ms = 0;
        while (*word == 0xFFFFFFFFu && waited_ms < 50) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          ++waited_ms;
        }
        if (*word != 0xFFFFFFFFu)
          std::cout << "WARNING run " << r << ": SHAVE " << w << " status arrived " << waited_ms
                    << " ms after stream.wait() returned" << std::endl;
      }
      if (*word != kStatusOk) {
        dumpStatus();
        dumpOutputs();
        die("SHAVE " + std::to_string(w) + " status " + std::to_string(*word) + " detail " +
            std::to_string(status[w * 8 + 1]) + " (status.bin and outputs written)");
      }
    }
    if (r >= warmup) {
      times.push_back(ms);
      std::cout << "run " << r - warmup << " ms " << ms << std::endl;
    }
  }
  if (!profile.empty()) {
    std::ofstream f(profile);
    k.saveProfilingJson(f);
  }
  for (auto &p : ps)
    if (p.mode == "out" || p.mode == "inout") {
      std::ofstream f(dir + "/" + p.file, std::ios::binary);
      f.write(static_cast<const char *>(p.dev), (std::streamsize)p.bytes);
    }
  dumpStatus();
  std::sort(times.begin(), times.end());
  if (!times.empty())
    std::cout << "median_ms " << times[times.size() / 2] << std::endl;
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: exception: " << e.what() << std::endl;
    return 1;
  }
}
