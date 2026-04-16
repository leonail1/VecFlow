/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef CUVS_VECFLOW_TAGORE_DEVICE_HELPERS_DEFINED
#define CUVS_VECFLOW_TAGORE_DEVICE_HELPERS_DEFINED

__device__ inline void bitonic_sort_id_new2(unsigned* shared_arr, unsigned len)
{
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (unsigned stride = 1; stride < len; stride <<= 1) {
    for (unsigned step = stride; step > 0; step >>= 1) {
      for (unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y) {
        unsigned a = 2 * step * (k / step);
        unsigned b = k % step;
        unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
        unsigned d = a + b + step;
        if (d < len && shared_arr[u] > shared_arr[d]) {
          auto tmp      = shared_arr[u];
          shared_arr[u] = shared_arr[d];
          shared_arr[d] = tmp;
        }
      }
      __syncthreads();
    }
  }
}

__device__ inline void bitonic_sort_id_by_dis(float* shared_arr,
                                              unsigned* ids,
                                              bool* visit,
                                              unsigned len)
{
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (unsigned stride = 1; stride < len; stride <<= 1) {
    for (unsigned step = stride; step > 0; step >>= 1) {
      for (unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y) {
        unsigned a = 2 * step * (k / step);
        unsigned b = k % step;
        unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
        unsigned d = a + b + step;
        if (d < len && shared_arr[u] > shared_arr[d]) {
          auto tmp_dis  = shared_arr[u];
          shared_arr[u] = shared_arr[d];
          shared_arr[d] = tmp_dis;

          auto tmp_id = ids[u];
          ids[u]      = ids[d];
          ids[d]      = tmp_id;

          auto tmp_visit = visit[u];
          visit[u]       = visit[d];
          visit[d]       = tmp_visit;
        }
      }
      __syncthreads();
    }
  }
}

__device__ inline float distance_filter(float res, float dis, float cur_dis, float threshold)
{
  (void)cur_dis;
  return ((threshold * res) - dis);
}

#endif
