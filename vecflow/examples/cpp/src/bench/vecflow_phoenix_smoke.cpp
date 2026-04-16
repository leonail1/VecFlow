#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "phoenix.h"

namespace fs = std::filesystem;

namespace {

constexpr size_t kAlignment = 4096;
constexpr size_t kTransferBytes = 64 * 1024;

void check_cuda(cudaError_t status, const char* message)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

void check_posix(bool condition, const std::string& message)
{
  if (!condition) { throw std::runtime_error(message + ": " + std::strerror(errno)); }
}

void write_exact(int fd, const void* buffer, size_t bytes)
{
  size_t written = 0;
  auto* ptr = static_cast<const uint8_t*>(buffer);
  while (written < bytes) {
    auto ret = ::pwrite(fd, ptr + written, bytes - written, static_cast<off_t>(written));
    check_posix(ret >= 0, "pwrite failed");
    written += static_cast<size_t>(ret);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto file_path = std::string(argc >= 2 ? argv[1] : "./vecflow_phoenix_smoke.bin");
    auto device_id = argc >= 3 ? std::stoi(argv[2]) : 0;

    if (!fs::path(file_path).parent_path().empty()) {
      fs::create_directories(fs::path(file_path).parent_path());
    }

    std::cout << "Phoenix smoke file: " << file_path << "\n";
    std::cout << "Device: " << device_id << "\n";
    std::cout << "Transfer bytes: " << kTransferBytes << "\n";

    check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");

    int fd = ::open(file_path.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_DIRECT, 0644);
    check_posix(fd >= 0, "open failed");

    void* host_buffer = nullptr;
    check_posix(::posix_memalign(&host_buffer, kAlignment, kTransferBytes) == 0,
                "posix_memalign failed");
    auto* host_bytes = static_cast<uint8_t*>(host_buffer);
    for (size_t i = 0; i < kTransferBytes; ++i) {
      host_bytes[i] = static_cast<uint8_t>(i % 251);
    }
    write_exact(fd, host_buffer, kTransferBytes);
    check_posix(::fsync(fd) == 0, "fsync failed");

    check_posix(phxfs_open(device_id) == 0, "phxfs_open failed");

    void* gpu_buffer = nullptr;
    void* target_addr = nullptr;
    check_cuda(cudaMalloc(&gpu_buffer, kTransferBytes), "cudaMalloc failed");
    check_cuda(cudaMemset(gpu_buffer, 0, kTransferBytes), "cudaMemset failed");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    check_posix(phxfs_regmem(device_id, gpu_buffer, kTransferBytes, &target_addr) == 0,
                "phxfs_regmem failed");

    phxfs_fileid_t file_id{};
    file_id.fd = fd;
    file_id.deviceID = device_id;
    auto bytes_read = phxfs_read(file_id, gpu_buffer, 0, kTransferBytes, 0);
    if (bytes_read != static_cast<ssize_t>(kTransferBytes)) {
      throw std::runtime_error("phxfs_read read " + std::to_string(bytes_read) + " bytes");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after phxfs_read failed");

    std::vector<uint8_t> roundtrip(kTransferBytes);
    check_cuda(cudaMemcpy(roundtrip.data(), gpu_buffer, kTransferBytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy failed");

    if (std::memcmp(roundtrip.data(), host_buffer, kTransferBytes) != 0) {
      throw std::runtime_error("Phoenix smoke validation failed: data mismatch");
    }

    std::cout << "Phoenix C++ smoke test passed. First 8 bytes: ";
    for (int i = 0; i < 8; ++i) {
      std::cout << static_cast<int>(roundtrip[static_cast<size_t>(i)]) << " ";
    }
    std::cout << "\n";

    check_posix(phxfs_deregmem(device_id, gpu_buffer, kTransferBytes) == 0,
                "phxfs_deregmem failed");
    check_cuda(cudaFree(gpu_buffer), "cudaFree failed");
    check_posix(phxfs_close(device_id) == 0, "phxfs_close failed");
    check_posix(::close(fd) == 0, "close failed");
    std::free(host_buffer);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
