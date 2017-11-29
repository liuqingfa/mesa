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

#include "st_glsl_to_tgsi_array_merge.h"
#include <util/u_math.h>
#include <ostream>
#include <cassert>
#include <algorithm>

#include <iostream>

class array_access_record {
public:
   int begin;
   int end;
   int array_id;
   int length;
   bool erase;
   int swizzle;

   array_access_record(struct array_lifetime *lt, int id, int l);

   bool operator < (const array_access_record& rhs) const {
      return begin < rhs.begin;
   }

   void merge(array_access_record& other);
};


array_lifetime::array_lifetime():
   array_id(0),
   array_length(0),
   begin(0),
   end(0),
   access_swizzle(0)
{
}

array_lifetime::array_lifetime(unsigned id, unsigned length):
   array_id(id),
   array_length(length),
   begin(0),
   end(0),
   access_swizzle(0),
   ncomponents(0)
{
}

array_lifetime::array_lifetime(unsigned id, unsigned length, int begin,
                               int end, int sw):
   array_id(id),
   array_length(length),
   begin(begin),
   end(end),
   access_swizzle(sw),
   ncomponents(util_bitcount(sw))
{
}

void array_lifetime::set_lifetime(int _begin, int _end)
{
   set_begin(_begin);
   set_end(_end);
}

void array_lifetime::set_swizzle(int sw)
{
   access_swizzle = sw;
   ncomponents = util_bitcount(sw);
}

void array_lifetime::augment_lifetime(int b, int e)
{
   if (b < begin)
      begin = b;
   if (e > end)
      end = e;
}

int array_lifetime::get_ncomponents() const
{
   return ncomponents;
}

void array_lifetime::print(std::ostream& os) const
{
   os << "[id:" << array_id
      << ", length:" << array_length
      << ", (b:" << begin
      << ", e:" << end
      << "), sw:" << access_swizzle
      << ", nc:" << ncomponents
      << "]";
}

bool array_lifetime::can_merge_with(const array_lifetime& other) const
{
   return (other.end < begin || end < other.begin);
}

bool array_lifetime::has_equal_access(const array_lifetime& other) const
{
   return begin == other.begin &&
         end == other.end &&
         access_swizzle == other.access_swizzle;
}

bool array_lifetime::contains_access_range(const array_lifetime& other) const
{
   return begin < other.begin &&
         end > other.end &&
         (access_swizzle | other.access_swizzle) == access_swizzle;
}

namespace tgsi_array_remap {

array_remapping::array_remapping(int tid):
   target_id(tid),
   reswizzle(false),
   valid(true)
{
}

array_remapping::array_remapping(int tid, int reserved_component_bits,
                                 int orig_component_bits):
   target_id(tid),
   reswizzle(true),
   valid(true)
{
#ifndef NDEBUG
   original_writemask = orig_component_bits;
#endif

   for (int i = 0; i < 4; ++i) {
      read_swizzle_map[i] = -1;
   }

   int src_swizzle = 1;
   int free_swizzle = 1;
   int k = 0;
   for (int i = 0; i < 4; ++i, src_swizzle <<= 1) {
      writemask_map[i] = 0;
      if (!(src_swizzle & orig_component_bits))
         continue;

      while (reserved_component_bits & free_swizzle) {
         free_swizzle <<= 1;
         ++k;
      }
      read_swizzle_map[i] = k;
      writemask_map[i] = free_swizzle;
      reserved_component_bits |= free_swizzle;
      assert(k < 4);
   }
   swizzle_sum = reserved_component_bits;
}

int array_remapping::writemask(int writemask_to_map) const
{
   assert(valid);

   if (!reswizzle)
      return writemask_to_map;

   assert(original_writemask & writemask_to_map);

   int result = 0;
   for (int i = 0; i < 4; ++i) {
      if (1 << i & writemask_to_map) {
         result |= writemask_map[i];
      }
   }
   return result;
}

int array_remapping::read_swizzle(int swizzle_to_map) const
{
   assert(valid);

   if (!reswizzle)
      return swizzle_to_map;

   assert(read_swizzle_map[swizzle_to_map] >= 0);
   return read_swizzle_map[swizzle_to_map];
}

void array_remapping::print(std::ostream& os) const
{
   static const char xyzw[] = "xyzw";
   if (valid) {
      os << "[aid: " << target_id;

      if (reswizzle) {
         os << " write-swz: ";
         for (int i = 0; i < 4; ++i) {
            switch (writemask_map[i]) {
            case 1: os << "x"; break;
            case 2: os << "y"; break;
            case 4: os << "z"; break;
            case 8: os << "w"; break;
            default: os << "_";
            }
         }
         os << " ";
         for (int i = 0; i < 4; ++i) {
            if (read_swizzle_map[i] >= 0)
               os << xyzw[read_swizzle_map[i]];
            else
            os << "_";
         }
      }
      os << "]";
   } else {
      os << "[unused]";
   }
}

void array_remapping::concat(const array_remapping& map)
{
   // we remap a valid remapping
   assert(valid);
   target_id = map.target_id;
}


bool operator == (const array_remapping& lhs, const array_remapping& rhs)
{
   if (!lhs.valid)
      return !rhs.valid;

   if (!rhs.valid)
      return false;

   if (lhs.target_id != rhs.target_id)
      return false;

   if (lhs.reswizzle) {
      return (rhs.reswizzle &&
           (memcmp(lhs.writemask_map, rhs.writemask_map,
                   4 * sizeof(uint8_t)) == 0) &&
           (memcmp(lhs.read_swizzle_map, rhs.read_swizzle_map,
                   4 * sizeof(uint8_t)) == 0));
   } else {
      return !rhs.reswizzle;
   }
}

bool sort_by_begin(const array_lifetime& lhs, const array_lifetime& rhs) {
   return lhs.get_begin() < rhs.get_begin();
}

static int merge_arrays_with_equal_swizzle(int narrays, array_lifetime *alt,
                                           array_remapping *remapping)
{
   int remaps = 0;
   std::sort(alt, alt + narrays, sort_by_begin);

   for (int i = 0; i < narrays; ++i) {
      array_lifetime& ai = alt[i];
      std::cerr << "Analysing: " << ai << "\n";

      if (remapping[ai.get_id()].is_valid())
         continue;

      for (int j = 0; j < narrays; ++j) {
         array_lifetime& aj = alt[j];
         std::cerr << "    vs: " << aj << "\n";

         if (i == j || remapping[aj.get_id()].is_valid())
            continue;

         if ((ai.get_array_length() < aj.get_array_length()) ||
             (ai.get_swizzle() !=  aj.get_swizzle()) ||
             !ai.can_merge_with(aj))
            continue;

         /* ai is a longer array then aj, they both have the same swizzle and
          * the life ranges don't overlap, hence they can be merged.
          */
         remapping[aj.get_id()] = array_remapping(ai.get_id());
         ai.augment_lifetime(aj.get_begin(), aj.get_end());
         std::cerr << "       --> Merge \n";

         ++remaps;
      }
   }
   return remaps;
}

static int interleave_arrays(int narrays, array_lifetime *alt,
                             array_remapping *remapping)
{
   int remaps = 0;
   for (int i = 0; i < narrays; ++i) {
      array_lifetime& ai = alt[i];
      std::cerr << "Analysing: " << ai << "\n";
      if (remapping[ai.get_id()].is_valid())
         continue;

      for (int j = 0; j < narrays; ++j) {
         array_lifetime& aj = alt[j];
         std::cerr << "    vs: " << aj << "\n";
         if (i == j || remapping[aj.get_id()].is_valid())
            continue;

         if ((ai.get_array_length() < aj.get_array_length()) ||
             (ai.get_ncomponents() + aj.get_ncomponents() > 4))
            continue;

         /* ai is a longer array then aj, and together they don
          *
          */
         remapping[aj.get_id()] = array_remapping(ai.get_id(), ai.get_swizzle(),
                                        aj.get_swizzle());
         ai.augment_lifetime(aj.get_begin(), aj.get_end());
         ai.set_swizzle(remapping[aj.get_id()].combined_swizzle());

         std::cerr << "      --> Interleave\n";

         ++remaps;
      }
   }
   return remaps;

}

bool get_array_remapping(int narrays, array_lifetime *arr_lifetimes,
                         array_remapping *remapping)
{
   int remapped_arrays;

   do {
      remapped_arrays = merge_arrays_with_equal_swizzle(narrays,
                                                        arr_lifetimes,
                                                        remapping);

      remapped_arrays += interleave_arrays(narrays,
                                           arr_lifetimes,
                                           remapping);

   } while (remapped_arrays > 0);

   return true;
}

}


int  merge_arrays(void *mem_ctx,
                  int narrays,
                  unsigned *array_sizes,
                  exec_list *instructions,
                  const struct array_lifetime *arr_lifetimes)
{


   return narrays;
}
