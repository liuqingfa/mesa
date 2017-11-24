/*
 * Copyright © 2017 Gert Wollny
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef MESA_GLSL_TO_TGSI_ARRAY_MERGE_H
#define MESA_GLSL_TO_TGSI_ARRAY_MERGE_H

#include "st_glsl_to_tgsi_private.h"

namespace tgsi_array_remap {

class array_remapping {
public:
   array_remapping(int tid, int res_swizzle, int old_swizzle);

   int writemask(int original_swizzle) const;
   int read_swizzle(int original_swizzle) const;
   int new_array_id() const;

   int target_id;
   int writemask_map[4];
   int read_swizzle_map[4];
};

}

struct array_lifetime {
  int begin;
  int end;
  int access_swizzle;
};

int merge_arrays(void *mem_ctx,
                 int narrays,
                 unsigned *array_sizes,
                 exec_list *instructions,
                 const struct array_lifetime *arr_lifetimes);

#endif