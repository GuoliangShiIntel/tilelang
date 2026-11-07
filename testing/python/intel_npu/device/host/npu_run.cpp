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
// npu_run --session --dir DIR --tiles T --shaves S: the persistent mode (see `session` below).
#include <npu_runtime.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kStatusBytesPerShave = 64;
constexpr std::uint32_t kStatusWords = kStatusBytesPerShave / 4;
// Status words 8 and 9: NCE_TIME free-running counter at kernel entry and at finish (19.2 MHz on this box).
constexpr std::uint32_t kStampStart = 8, kStampEnd = 9;
constexpr double kPerfClkMHz = 19.2;
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

  std::vector<double> times, kernelTimes;
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
      volatile std::uint32_t *word = status + w * kStatusWords;
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
            std::to_string(status[w * kStatusWords + 1]) + " (status.bin and outputs written)");
      }
    }
    // Kernel-only span: earliest entry stamp to latest finish stamp over every SHAVE, in
    // perf-clock ticks (32-bit counter; differences are taken modulo 2^32).
    std::uint32_t start0 = status[kStampStart];
    for (std::uint32_t w = 1; w < workers; ++w)
      if ((std::int32_t)(status[w * kStatusWords + kStampStart] - start0) < 0)
        start0 = status[w * kStatusWords + kStampStart];
    std::uint32_t spanTicks = 0;
    for (std::uint32_t w = 0; w < workers; ++w)
      spanTicks = std::max(spanTicks, status[w * kStatusWords + kStampEnd] - start0);
    const double kernelMs = spanTicks / kPerfClkMHz / 1000.0;
    if (r >= warmup) {
      times.push_back(ms);
      kernelTimes.push_back(kernelMs);
      std::cout << "run " << r - warmup << " ms " << ms << " kernel_ms " << kernelMs << std::endl;
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
  std::sort(kernelTimes.begin(), kernelTimes.end());
  if (!times.empty())
    std::cout << "median_ms " << times[times.size() / 2] << " median_kernel_ms "
              << kernelTimes[kernelTimes.size() / 2] << std::endl;
  return 0;
}

// Session mode: one process, kernels loaded once, device buffers resident, launches driven over stdin
// (one command per line, one reply line per command, flushed). A model's decode loop is a sequence of
// launches over resident weights with the host reading back a few KB per token.
//   alloc NAME BYTES              shared device buffer, zeroed
//   load NAME FILE [OFFSET]       DIR/FILE -> buffer at OFFSET
//   zero NAME
//   save NAME FILE [OFFSET BYTES] buffer -> DIR/FILE
//   kernel ALIAS ELFNAME [profile]
//   launch ALIAS NAME...          -> "ok ms <host> kernel_ms <k>" or "error <why>"
//   profile ALIAS FILE            the profiler JSON of a kernel created with `profile`
//   quit
int session(const std::string &dir, int tiles, int shaves) {
  auto device = npu::Device();
  if ((std::size_t)tiles > device.getTileCount())
    die("session wants " + std::to_string(tiles) + " tiles, device has " + std::to_string(device.getTileCount()));
  struct Buf {
    void *dev;
    std::uint64_t bytes;
  };
  std::map<std::string, Buf> bufs;
  std::map<std::string, npu::Kernel *> kernels;
  const std::uint32_t workers = (std::uint32_t)(tiles * shaves);
  const std::uint32_t statusSlots = (std::uint32_t)device.getTileCount() * (std::uint32_t)shaves;
  const std::uint64_t statusBytes = (std::uint64_t)statusSlots * kStatusBytesPerShave;
  auto *status = static_cast<std::uint32_t *>(device.mallocShared(statusBytes));
  auto &stream = device.createStream();
  std::cout << "session ready tiles " << tiles << " shaves " << shaves << std::endl;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    std::istringstream ls(line);
    std::string cmd;
    ls >> cmd;
    auto reply = [&](const std::string &s) { std::cout << s << std::endl; };
    try {
      if (cmd.empty())
        continue;
      if (cmd == "quit")
        break;
      if (cmd == "alloc") {
        std::string name;
        std::uint64_t bytes = 0;
        ls >> name >> bytes;
        if (bufs.count(name)) {
          reply("error buffer exists " + name);
          continue;
        }
        void *p = device.mallocShared(std::max<std::uint64_t>(bytes, 64));
        if (!p) {
          reply("error mallocShared " + std::to_string(bytes));
          continue;
        }
        std::memset(p, 0, bytes);
        bufs[name] = Buf{p, bytes};
        reply("ok");
      } else if (cmd == "load" || cmd == "save") {
        std::string name, file;
        std::uint64_t off = 0, n = 0;
        ls >> name >> file >> off >> n;
        auto it = bufs.find(name);
        if (it == bufs.end()) {
          reply("error no buffer " + name);
          continue;
        }
        if (cmd == "load") {
          std::ifstream f(dir + "/" + file, std::ios::binary | std::ios::ate);
          if (!f) {
            reply("error cannot read " + file);
            continue;
          }
          const std::uint64_t size = (std::uint64_t)f.tellg();
          if (off + size > it->second.bytes) {
            reply("error " + file + " (" + std::to_string(size) + " B at " + std::to_string(off) + ") exceeds " + name);
            continue;
          }
          f.seekg(0);
          f.read(static_cast<char *>(it->second.dev) + off, (std::streamsize)size);
          reply("ok " + std::to_string(size));
        } else {
          if (n == 0)
            n = it->second.bytes - off;
          if (off + n > it->second.bytes) {
            reply("error range exceeds " + name);
            continue;
          }
          std::ofstream f(dir + "/" + file, std::ios::binary);
          f.write(static_cast<const char *>(it->second.dev) + off, (std::streamsize)n);
          reply("ok " + std::to_string(n));
        }
      } else if (cmd == "zero") {
        std::string name;
        ls >> name;
        auto it = bufs.find(name);
        if (it == bufs.end()) {
          reply("error no buffer " + name);
          continue;
        }
        std::memset(it->second.dev, 0, it->second.bytes);
        reply("ok");
      } else if (cmd == "kernel") {
        std::string alias, elf, opt;
        ls >> alias >> elf >> opt;
        auto &k = device.createKernel(elf, npu::Device::Tiles(tiles), npu::Device::ShavesPerTile(shaves));
        if (opt == "profile")
          k.setProfilingEnabled(true);
        kernels[alias] = &k;
        reply("ok");
      } else if (cmd == "profile") {
        std::string alias, file;
        ls >> alias >> file;
        auto it = kernels.find(alias);
        if (it == kernels.end()) {
          reply("error no kernel " + alias);
          continue;
        }
        std::ofstream f(file);
        it->second->saveProfilingJson(f);
        reply("ok");
      } else if (cmd == "launch") {
        std::string alias, name;
        ls >> alias;
        auto it = kernels.find(alias);
        if (it == kernels.end()) {
          reply("error no kernel " + alias);
          continue;
        }
        std::vector<void *> params;
        bool bad = false;
        while (ls >> name) {
          auto b = bufs.find(name);
          if (b == bufs.end()) {
            reply("error no buffer " + name);
            bad = true;
            break;
          }
          params.push_back(b->second.dev);
        }
        if (bad)
          continue;
        params.push_back(status);
        std::memset(status, 0xFF, statusBytes);
        auto t0 = std::chrono::steady_clock::now();
        it->second->submit(params, stream);
        stream.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::string health;
        if (!device.isHealthy(&health)) {
          reply("fatal device unhealthy after " + alias + ": " + health);
          return 1;
        }
        std::string err;
        for (std::uint32_t w = 0; w < workers; ++w) {
          volatile std::uint32_t *word = status + w * kStatusWords;
          int waited_ms = 0;
          while (*word == 0xFFFFFFFFu && waited_ms < 50) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++waited_ms;
          }
          if (*word != kStatusOk) {
            err += " shave " + std::to_string(w) + " status " + std::to_string(*word) + " detail " +
                   std::to_string(status[w * kStatusWords + 1]);
          }
        }
        if (!err.empty()) {
          reply("error " + alias + ":" + err);
          continue;
        }
        std::uint32_t start0 = status[kStampStart];
        for (std::uint32_t w = 1; w < workers; ++w)
          if ((std::int32_t)(status[w * kStatusWords + kStampStart] - start0) < 0)
            start0 = status[w * kStatusWords + kStampStart];
        std::uint32_t spanTicks = 0;
        for (std::uint32_t w = 0; w < workers; ++w)
          spanTicks = std::max(spanTicks, status[w * kStatusWords + kStampEnd] - start0);
        std::ostringstream os;
        os << "ok ms " << ms << " kernel_ms " << spanTicks / kPerfClkMHz / 1000.0;
        reply(os.str());
      } else {
        reply("error unknown command " + cmd);
      }
    } catch (const std::exception &e) {
      reply(std::string("error exception ") + e.what());
    }
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--session") {
      std::string dir;
      int tiles = 0, shaves = 0;
      for (int i = 2; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--dir")
          dir = v;
        else if (a == "--tiles")
          tiles = std::stoi(v);
        else if (a == "--shaves")
          shaves = std::stoi(v);
        else
          die("unknown session argument " + a);
      }
      if (dir.empty() || tiles <= 0 || shaves <= 0)
        die("usage: npu_run --session --dir DIR --tiles T --shaves S");
      return session(dir, tiles, shaves);
    }
    return run(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: exception: " << e.what() << std::endl;
    return 1;
  }
}
