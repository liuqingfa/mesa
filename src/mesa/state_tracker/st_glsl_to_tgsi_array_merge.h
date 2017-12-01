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

/* Helper class to evalaute the required lifetime of an array. */

class array_lifetime {
public:
   array_lifetime();
   array_lifetime(unsigned aid, unsigned alength);
   array_lifetime(unsigned aid, unsigned alength, int first_access,
                  int last_access, int mask);

   void set_lifetime(int first_access, int last_access);
   void set_begin(int _begin){first_access = _begin;}
   void set_end(int _end){last_access = _end;}
   void set_access_mask(int s);
   void merge_lifetime(int _begin, int _end);

   unsigned array_id() const {return id;}
   int array_length() const { return length;}
   int begin() const { return first_access;}
   int end() const { return last_access;}
   int access_mask() const { return component_access_mask;}
   int used_components() const {return used_component_count;}

   bool time_doesnt_overlap(const array_lifetime& other) const;

   void print(std::ostream& os) const;

private:
   unsigned id;
   unsigned length;
   int first_access;
   int last_access;
   int component_access_mask;
   int used_component_count;
};

inline
std::ostream& operator << (std::ostream& os, const array_lifetime& lt) {
   lt.print(os);
   return os;
}


namespace tgsi_array_merge {

/* Helper class to merge and interleave arrays.
 * The interface is exposed here to make unit tests possible.
 */

class array_remapping {
public:
   array_remapping();

   /* Simple remapping that is done when the lifetimes do not
    * overlap.
    */
   array_remapping(int target_array_id);

   /* Component interleaving of arrays.
    */
   array_remapping(int target_array_id, int target_component_mask,
                   int original_component_mask);

   /* Defines a valid remapping */
   bool is_valid() const {return target_id > 0;}

   /* Set a new target ID for array merging */
   void set_target_id(int new_tid);

   /* Propagate the target ID and a possible swizzle mapping */
   void propagate_remapping(const array_remapping& map);

   /* Translates the write mask to the new, interleaved component
    * position
    */
   int map_writemask(int original_writemask) const;

   /* Translates one read swizzle to the new, interleaved component
    * swizzle
    */
   int map_one_swizzle(int original_swizzle) const;

   /* Translates all read swizzles to the new, interleaved component
    * swizzles
    */
   uint16_t map_swizzles(uint16_t original_swizzle) const;

   unsigned target_array_id() const {return target_id;}

   int combined_access_mask() const {return summary_component_mask;}

   void print(std::ostream& os) const;

   friend bool operator == (const array_remapping& lhs,
                            const array_remapping& rhs);
private:
   void evaluate_swizzle_map(uint8_t reserved_component_bits,
                             uint8_t orig_component_bits);

   unsigned target_id;
   uint8_t writemask_map[4];
   int8_t read_swizzle_map[4];
   unsigned summary_component_mask:4;
   unsigned original_writemask:4;
   int reswizzle:1;
};

inline
std::ostream& operator << (std::ostream& os, const array_remapping& am)
{
   am.print(os);
   return os;
}

bool get_array_remapping(int narrays, array_lifetime *arr_lifetimes,
                         array_remapping *remapping);

}

int merge_arrays(int narrays,
                 unsigned *array_sizes,
                 exec_list *instructions,
                 struct array_lifetime *arr_lifetimes);

#endif