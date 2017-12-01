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

#include <program/prog_instruction.h>
#include <util/u_math.h>
#include <ostream>
#include <cassert>
#include <algorithm>

#include <iostream>

array_lifetime::array_lifetime():
   id(0),
   length(0),
   first_access(0),
   last_access(0),
   component_access_mask(0),
   used_component_count(0)
{
}

array_lifetime::array_lifetime(unsigned aid, unsigned alength):
   id(aid),
   length(alength),
   first_access(0),
   last_access(0),
   component_access_mask(0),
   used_component_count(0)
{
}

array_lifetime::array_lifetime(unsigned aid, unsigned alength, int begin,
                               int end, int sw):
   id(aid),
   length(alength),
   first_access(begin),
   last_access(end),
   component_access_mask(sw),
   used_component_count(util_bitcount(sw))
{
}

void array_lifetime::set_lifetime(int _begin, int _end)
{
   set_begin(_begin);
   set_end(_end);
}

void array_lifetime::set_access_mask(int mask)
{
   component_access_mask = mask;
   used_component_count = util_bitcount(mask);
}

void array_lifetime::merge_lifetime(int _begin, int _end)
{
   if (_begin < first_access)
      first_access = _begin;
   if (_end > last_access)
      last_access = _end;
}

void array_lifetime::print(std::ostream& os) const
{
   os << "[id:" << id
      << ", length:" << length
      << ", (b:" << first_access
      << ", e:" << last_access
      << "), sw:" << component_access_mask
      << ", nc:" << used_component_count
      << "]";
}

bool array_lifetime::time_doesnt_overlap(const array_lifetime& other) const
{
   return (other.last_access < first_access || last_access < other.first_access);
}

namespace tgsi_array_merge {

array_remapping::array_remapping():
   target_id(0)
{
}

array_remapping::array_remapping(int tid):
   target_id(tid),
   reswizzle(false)
{
}

array_remapping::array_remapping(int tid, int reserved_component_bits,
                                 int orig_component_bits):
   target_id(tid),
   original_writemask(orig_component_bits),
   reswizzle(true)
{
   evaluate_swizzle_map(reserved_component_bits, orig_component_bits);
}

void array_remapping::evaluate_swizzle_map(uint8_t reserved_component_mask,
                                           uint8_t orig_component_mask)
{
   for (int i = 0; i < 4; ++i) {
      read_swizzle_map[i] = -1;
      writemask_map[i] = 0;
   }

   int src_swizzle_bit = 1;
   int next_free_swizzle_bit = 1;
   int k = 0;

   for (int i = 0; i < 4; ++i, src_swizzle_bit <<= 1) {

      if (!(src_swizzle_bit & orig_component_mask))
         continue;

      while (reserved_component_mask & next_free_swizzle_bit) {
         next_free_swizzle_bit <<= 1;
         ++k;
         assert(k < 4);
      }

      read_swizzle_map[i] = k;
      writemask_map[i] = next_free_swizzle_bit;
      reserved_component_mask |= next_free_swizzle_bit;
   }

   summary_component_mask = reserved_component_mask;
}

int array_remapping::map_writemask(int writemask_to_map) const
{
   assert(is_valid());

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

int array_remapping::map_one_swizzle(int swizzle_to_map) const
{
   assert(is_valid());

   if (!reswizzle)
      return swizzle_to_map;

   assert(read_swizzle_map[swizzle_to_map] >= 0);
   return read_swizzle_map[swizzle_to_map];
}

uint16_t array_remapping::map_swizzles(uint16_t old_swizzle) const
{
   if (!reswizzle)
      return old_swizzle;

   uint16_t out_swizzle = 0;
   for (int idx = 0; idx < 4; ++idx) {
     uint16_t swz = map_one_swizzle(GET_SWZ(old_swizzle, idx));
     out_swizzle |= swz << 3 * idx;
   }
   return out_swizzle;
}

void array_remapping::print(std::ostream& os) const
{
   static const char xyzw[] = "xyzw";
   if (is_valid()) {
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

void array_remapping::set_target_id(int new_tid)
{
   assert(is_valid());
   target_id = new_tid;
}

void array_remapping::propagate_remapping(const array_remapping& map)
{
   assert(is_valid());
   target_id = map.target_id;
   memcpy(read_swizzle_map, map.read_swizzle_map, 4 * sizeof(int));
   memcpy(writemask_map, map.writemask_map, 4 * sizeof(int));
}

bool operator == (const array_remapping& lhs, const array_remapping& rhs)
{
   if (lhs.target_id != rhs.target_id)
      return false;

   if (lhs.target_id == 0)
      return true;

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
   return lhs.begin() < rhs.begin();
}

static int merge_arrays_with_equal_swizzle(int narrays, array_lifetime *alt,
                                           array_remapping *remapping)
{
   int remaps = 0;
   std::sort(alt, alt + narrays, sort_by_begin);

   for (int i = 0; i < narrays; ++i) {
      array_lifetime& ai = alt[i];

      if (remapping[ai.array_id()].is_valid())
         continue;

      for (int j = 0; j < narrays; ++j) {
         array_lifetime& aj = alt[j];

         if (i == j || remapping[aj.array_id()].is_valid())
            continue;

         if ((ai.array_length() < aj.array_length()) ||
             (ai.access_mask() !=  aj.access_mask()) ||
             !ai.time_doesnt_overlap(aj))
            continue;

         /* ai is a longer array then aj, they both have the same swizzle and
          * the life ranges don't overlap, hence they can be merged.
          */
         std::cerr << "Merge " << aj << " with " << ai << "\n";
         remapping[aj.array_id()] = array_remapping(ai.array_id());
         ai.merge_lifetime(aj.begin(), aj.end());

         for (int k = 1; k <= narrays; ++k) {
            if (remapping[k].target_array_id() == aj.array_id()) {
               std::cerr << "Remap propagate id " << k << " -> " << aj.array_id() << "\n";
               remapping[k].set_target_id(remapping[aj.array_id()].target_array_id());
            }
         }

         ++remaps;
      }
   }
   return remaps;
}

static int merge_arrays(int narrays, array_lifetime *alt,
                        array_remapping *remapping)
{
   int remaps = 0;

   for (int i = 0; i < narrays; ++i) {
      array_lifetime& ai = alt[i];

      if (remapping[ai.array_id()].is_valid())
         continue;

      for (int j = 0; j < narrays; ++j) {
         array_lifetime& aj = alt[j];

         if (i == j || remapping[aj.array_id()].is_valid())
            continue;

         if ((ai.array_length() < aj.array_length()) ||
             !ai.time_doesnt_overlap(aj))
            continue;

         /* ai is a longer array then aj, they both have the same swizzle and
          * the life ranges don't overlap, hence they can be merged.
          */
         std::cerr << "Merge " << aj << " with " << ai << "\n";
         remapping[aj.array_id()] = array_remapping(ai.array_id());
         ai.merge_lifetime(aj.begin(), aj.end());

         for (int k = 1; k <= narrays; ++k) {
            if (remapping[k].target_array_id() == aj.array_id()) {
               std::cerr << "Remap propagate id " << k << " -> " << aj.array_id() << "\n";
               remapping[k].set_target_id(remapping[aj.array_id()].target_array_id());
            }
         }

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
      if (remapping[ai.array_id()].is_valid())
         continue;

      for (int j = 0; j < narrays; ++j) {
         array_lifetime& aj = alt[j];

         if (i == j || remapping[aj.array_id()].is_valid())
            continue;

         if ((ai.array_length() < aj.array_length()) ||
             (ai.used_components() + aj.used_components() > 4) ||
             ai.time_doesnt_overlap(aj))
            continue;

         /* ai is a longer array then aj, and together they don't occupy at
          * most all four components
          */
         std::cerr << "Interleave " << aj << " with " << ai << "\n";
         remapping[aj.array_id()] = array_remapping(ai.array_id(), ai.access_mask(),
                                        aj.access_mask());
         ai.merge_lifetime(aj.begin(), aj.end());
         ai.set_access_mask(remapping[aj.array_id()].combined_access_mask());

         /* If any array was merged with aj this one before, we need to propagate
          * the swizzle changes
          */
         for (int k = 1; k <= narrays; ++k) {
            if (remapping[k].target_array_id() == aj.array_id()) {
               std::cerr << "Remap propagate " << k << " -> " << aj.array_id() << "\n";
               std::cerr << "   remap was: " << remapping[k] << "\n";
               remapping[k].propagate_remapping(remapping[aj.array_id()]);
               std::cerr << "   now: " << remapping[k] << "\n";
            }
         }

         ++remaps;
      }
   }
   return remaps;

}

bool get_array_remapping(int narrays, array_lifetime *arr_lifetimes,
                         array_remapping *remapping)
{
   int total_remapped_arrays = 0;
   int remapped_arrays;

   do {
      remapped_arrays = merge_arrays_with_equal_swizzle(narrays,
                                                        arr_lifetimes,
                                                        remapping);

      remapped_arrays += interleave_arrays(narrays,
                                           arr_lifetimes,
                                           remapping);

      total_remapped_arrays += remapped_arrays;
   } while (remapped_arrays > 0);

   total_remapped_arrays += merge_arrays(narrays, arr_lifetimes, remapping);

   return total_remapped_arrays > 0;
}

int remap_arrays(int narrays, unsigned *array_sizes,
                 exec_list *instructions,
                 array_remapping *map)
{
   /* re-calculate arrays */
   int *idx_map = new int[narrays + 1];
   unsigned *old_sizes = new unsigned[narrays + 1];
   memcpy(old_sizes, array_sizes, sizeof(unsigned) * narrays);

   int new_narrays = 0;
   for (int i = 1; i <= narrays; ++i) {
      if (!map[i].is_valid()) {
         ++new_narrays;
         idx_map[i] = new_narrays;
         array_sizes[new_narrays] = old_sizes[i];
         std::cerr << "Array " << i << " is now " << new_narrays << "\n";
      }
   }

   for (int i = 1; i <= narrays; ++i)
      if (map[i].is_valid()) {
         std::cerr << "Propagate mapping " << i << "("<< map[i].target_array_id()
                   <<  ") to " << idx_map[map[i].target_array_id()] << "\n";
         map[i].set_target_id(idx_map[map[i].target_array_id()]);
      } else {
         std::cerr << "Set array mapping " << i << "to " << idx_map[i] << "\n";
         map[i].set_target_id(idx_map[i]);
      }

   foreach_in_list(glsl_to_tgsi_instruction, inst, instructions) {
      for (unsigned j = 0; j < num_inst_src_regs(inst); j++) {
         st_src_reg& src = inst->src[j];
         if (src.file == PROGRAM_ARRAY) {
            array_remapping& m = map[src.array_id];
            if (m.is_valid()) {
               src.array_id = m.target_array_id();
               src.swizzle = m.map_swizzles(src.swizzle);
            }
         }
      }
      for (unsigned j = 0; j < inst->tex_offset_num_offset; j++) {
         st_src_reg& src = inst->tex_offsets[j];
         if (src.file == PROGRAM_ARRAY) {
            array_remapping& m = map[src.array_id];
            if (m.is_valid()) {
               src.array_id = m.target_array_id();
               src.swizzle = m.map_swizzles(src.swizzle);
            }
         }
      }
      for (unsigned j = 0; j < num_inst_dst_regs(inst); j++) {
         st_dst_reg& dst = inst->dst[j];
         if (dst.file == PROGRAM_ARRAY) {
            array_remapping& m = map[dst.array_id];
            if (m.is_valid()) {
               dst.array_id = m.target_array_id();
               dst.writemask = m.map_writemask(dst.writemask);
            }
         }
      }
   }

   delete[] old_sizes;
   delete[] idx_map;

   return new_narrays;
}

}

using namespace tgsi_array_merge;


int  merge_arrays(int narrays,
                  unsigned *array_sizes,
                  exec_list *instructions,
                  struct array_lifetime *arr_lifetimes)
{

   array_remapping *map= new array_remapping[narrays + 1];

   if (get_array_remapping(narrays, arr_lifetimes, map))
      narrays = remap_arrays(narrays, array_sizes, instructions, map);

   delete[] map;
   return narrays;
}
