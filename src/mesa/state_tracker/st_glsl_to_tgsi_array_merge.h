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
#include <iosfwd>

class array_lifetime {

public:
   array_lifetime();
   array_lifetime(unsigned id, unsigned length);

   array_lifetime(unsigned id, unsigned length, int begin, int end, int sw);
   void set_lifetime(int begin, int end);
   void set_begin(int _begin){begin = _begin;}
   void set_end(int _end){end = _end;}
   void set_swizzle(int s){access_swizzle = s;}
   int get_begin() const { return begin;}
   int get_end() const { return end;}
   int get_swizzle() const { return access_swizzle;}

   bool has_equal_access(const array_lifetime& other) const;
   bool contains_access_range(const array_lifetime& other) const;

private:
   unsigned array_id;
   unsigned array_length;
   int begin;
   int end;
   int access_swizzle;
};

namespace tgsi_array_remap {

class array_remapping {
public:
   array_remapping():valid(false) {}
   array_remapping(int tid);
   array_remapping(int tid, int res_swizzle, int old_swizzle);

   int writemask(int original_bits) const;
   int read_swizzle(int original_bits) const;
   int new_array_id() const {return target_id;}
   bool is_valid() const ;

   friend bool operator == (const array_remapping& lhs,
                            const array_remapping& rhs);

   void print(std::ostream& os) const;

private:
   int target_id;
   uint8_t writemask_map[4];
   uint8_t read_swizzle_map[4];
   bool reswizzle;
   bool valid;

#ifndef NDEBUG
   int original_writemask;
#endif
};

inline
std::ostream& operator << (std::ostream& os, const array_remapping& am)
{
   am.print(os);
   return os;
}

bool get_array_remapping(array_lifetime *arr_lifetimes,
                         array_remapping *remapping);

}

int merge_arrays(void *mem_ctx,
                 int narrays,
                 unsigned *array_sizes,
                 exec_list *instructions,
                 const struct array_lifetime *arr_lifetimes);

#endif