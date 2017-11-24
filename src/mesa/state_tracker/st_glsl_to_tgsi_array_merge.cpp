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
#include <ostream>

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


namespace tgsi_array_remap {

array_remapping::array_remapping(int tid, int reserved_component_bits,
                                 int orig_component_bits):
   target_id(tid),
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
}

int array_remapping::writemask(int writemask_to_map) const
{
   assert(valid);
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
   assert(read_swizzle_map[swizzle_to_map] >= 0);
   return read_swizzle_map[swizzle_to_map];
}

void array_remapping::print(std::ostream& os) const
{
   static const char xyzw[] = "xyzw";
   if (valid) {
      os << "[aid: " << target_id
         << "write-swz: ";
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
      os << "]";
   } else {
      os << "[unused]";
   }
}

bool operator == (const array_remapping& lhs, const array_remapping& rhs)
{
   if (!lhs.valid)
      return !rhs.valid;

   if (!rhs.valid)
      return false;

   if (lhs.target_id != rhs.target_id)
      return false;

   return ((memcmp(lhs.writemask_map, rhs.writemask_map,
                   4 * sizeof(uint8_t)) == 0) &&
           (memcmp(lhs.read_swizzle_map, rhs.read_swizzle_map,
                   4 * sizeof(uint8_t)) == 0));
}

bool get_array_remapping(void *mem_ctx, int narrays, int *array_length,
                         struct array_lifetime *arr_lifetimes,
                         array_remapping *remapping)
{
   remapping[1] = array_remapping(1, 1, 1);
   return true;
}

}

#if 0
namespace {


array_access_record::array_access_record(struct array_lifetime *lt, int id,
                                         int l):
   begin(lt->begin),
   end(lt->end),
   array_id(id),
   length(l),
   erase(false),
   swizzle(lt->access_swizzle)
{
}

void array_access_record::merge(array_access_record& other)
{
   assert(length >= other.length);

   if (begin > other.begin)
      begin = other.begin;

   if (end > other.end)
      end = other.end;

   swizzle |= other.swizzle;

   other.erase = true;

}

void get_array_interleave_mapping(void *mem_ctx, int narrays,
                                  const int *array_length,
                                  struct array_lifetime *lifetimes,
                                  struct array_remap_pair *result)
{
   struct fuse_distance {
      int apply(const array_access_record& s1, const array_access_record& s2)
      {
         int l1 = s1->length - s2->length;
         int l2 = s1->begin - s2->begin;
         int l3 = s1->end - s2->end;
         return l1 * l1 + l2 * l2 + l3 * l3;
      }
   };

   struct  aar_length_gt {
      bool operator < (const array_access_record& s1,
                       const array_access_record& s2) const
      {
         return s1.length > s2.length;
      }
   };

   static const int standard_swizzle_map[4] = {0,1,2,3};

   array_access_record  *acc =  ralloc_array(mem_ctx, array_access_record, narrays);

   int used_arrays = 0;
   for (int i = 0; i < narrays; ++i) {
      if (lifetimes[i].begin >= 0) {
         acc[used_arrays].begin = lifetimes[i].begin;
         acc[used_arrays].end = lifetimes[i].end;
         acc[used_arrays].array_id = i;
         acc[used_arrays].erase = false;
         acc[used_arrays].length = array_length[i];
         acc[used_arrays].swizzle = lifetimes[i].access_swizzle;
         ++used_arrays;
      }
   }

   /* The number of arrays is usually low (< 20), so we do
    * this brute force
    */

   std::sort(acc, acc+used_arrays, aar_length_gt);

   int i = 0;
   int j = 1;
   while (i < used_arrays) {

      int best_fuse_distance = numeric_limits<int>::max();
      int best_fuse_candidate = -1;

      while (j < used_arrays) {

         if (acc[j].erase ||
             (acc[i].swizzle & acc[j].swizzle  != 0))
            break;

         assert(acc[i].length > acc[j].length);

         int new_fuse_distance = fuse_distance::apply(acc[i], acc[j]);
         if (new_fuse_distance < best_fuse_distance) {
            best_fuse_distance = new_fuse_distance;
            best_fuse_candidate = j;
         }
         ++j;
      }

      if (best_fuse_candidate > 0) {
         merge_into::apply(acc[i], acc[j]);
      }

   }


}

void get_array_remapping(void *mem_ctx, int narrays,
                         const int *array_length,
                         const struct array_lifetime *lifetimes,
                         struct array_remap_pair *result)
{
   static const int standard_swizzle_map[4] = {0,1,2,3};

   /* First try to obtain an array renaming */
   array_access_record  *acc =  ralloc_array(mem_ctx, array_access_record, narrays);

   int used_arrays = 0;
   for (int i = 0; i < narrays; ++i) {
      if (lifetimes[i].begin >= 0) {
         acc[used_arrays].begin = lifetimes[i].begin;
         acc[used_arrays].end = lifetimes[i].end;
         acc[used_arrays].array_id = i;
         acc[used_arrays].erase = false;
         acc[used_arrays].length = array_length[i];
         acc[used_arrays].swizzle = lifetimes[i].access_swizzle;
         ++used_arrays;
      }
   }

#ifdef USE_STL_SORT
   std::sort(acc, acc + used_arrays);
#else
   std::qsort(acc, used_arrays, sizeof(array_access_record),
              array_access_record_compare);
#endif
   array_access_record *trgt = acc;
   array_access_record *access_end = acc + used_arrays;
   array_access_record *first_erase = access_end;
   array_access_record *search_start = trgt + 1;

   while (trgt != access_end) {
      array_access_record *src = find_next_rename(search_start, access_end,
                                                  trgt->end);
      if (src != access_end) {
         array_access_record *eliminated;

         /* Always map to the larger array */
         if (src->length < trgt->length) {
            result[src->array_id].target_array_id = trgt->array_id;
            result[src->array_id].valid = true;
            memcpy(result[src->array_id].rename_swizzle, standard_swizzle_map,
                   4 * sizeof(int));
            trgt->end = src->end;
            trgt->swizzle |= src->swizzle;
            eliminated = src;
         }else{
            result[trgt->array_id].target_array_id = src->array_id;
            result[trgt->array_id].valid = true;
            memcpy(result[src->array_id].rename_swizzle, standard_swizzle_map,
                   4 * sizeof(int));
            src->end = trgt->end;
            src->swizzle |= trgt->swizzle;
            eliminated = trgt;
            trgt = src;
         }

         --used_arrays;

         /* Since we only search forward, don't remove the renamed
          * array just now, only mark it. */
         eliminated->erase = true;

         if (first_erase == access_end)
            first_erase = eliminated;

         search_start = src + 1;
      } else {
         /* Moving to the next target array it is time to remove
          * the already merged array from the search range */
         if (first_erase != access_end) {
            array_access_record *outp = first_erase;
            array_access_record *inp = first_erase + 1;

            while (inp != access_end) {
               if (!inp->erase)
                  *outp++ = *inp;
               ++inp;
            }

            access_end = outp;
            first_erase = access_end;
         }
         ++trgt;
         search_start = trgt + 1;
      }
   }

   if (used_arrays > 1) {
      /* Try to interleave arrays */
      for (int i = 0; i < used_arrays; ++i) {
         if (acc[i].erase)
            continue;

         for (int j = 0; j < used_arrays; ++j) {
            if (acc[j].erase)
               continue;

            if ((acc[i].swizzle & acc[j].swizzle) == 0) {
               if (acc[i].length > acc[j].length) {
                  result[j].target_array_id = acc[i].array_id;
                  result[j].valid = true;
                  get_swizzle_map(acc[i].swizzle, acc[j].swizzle, result[j].rename_swizzle);
                  acc[i].swizzle |= acc[j].swizzle;
                  acc[j].erase = true;
               } else {
                  result[i].target_array_id = acc[j].array_id;
                  result[i].valid = true;
                  get_swizzle_map(acc[j].swizzle, acc[i].swizzle, result[i].rename_swizzle);
                  acc[j].swizzle |= acc[i].swizzle;
                  acc[i].erase = true;
               }
            }
         }
      }
   }

   ralloc_free(acc);
}


}

#endif


int  merge_arrays(void *mem_ctx,
                  int narrays,
                  unsigned *array_sizes,
                  exec_list *instructions,
                  const struct array_lifetime *arr_lifetimes)
{


   return narrays;
}
