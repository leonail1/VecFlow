# Troubleshooting Kernel Module Installation

Since Phoenix relies on ZONE_DEVICE to provide memory mapping services, the system must ensure that no other applications or kernel drivers are using the PCIe BAR resources when installing the kernel module (phxfs.ko).

If you encounter the following error during installation:
```shell
insmod: ERROR: could not insert module phoenixfs.ko: Operation not permitted
```
it typically indicates that the PCIe BAR resources required by Phoenix are already in use.
Follow the steps below to identify and resolve the conflict.

## 1.Check whether any user-space applications are using GPU memory
First, verify whether any user-space applications are currently accessing NVIDIA device resources:
```shell
sudo lsof /dev/nvidia*
```
If any processes are listed, terminate them and retry installing the kernel module.

## 2.Check whether any kernel drivers are using GPU memory
Next, inspect the set of loaded NVIDIA-related kernel modules:
```shell
sudo lsmod | grep nvidia
```
In most cases, you will see that `nvidia_drm` is used by other DRM-related modules, which results in partial occupation of GPU memory resources.

To resolve this issue, you may either:
- Add `nvidia_drm` to the system blacklist to prevent it from being loaded automatically, or
- Manually unload `nvidia_drm` and its dependent modules using rmmod, then retry the installation.